from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI, OpenAI
import time
import numpy as np
import asyncio
from collections import defaultdict, deque
import tempfile
import os
import json
import chromadb
from pydantic import BaseModel
from datetime import datetime
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import atexit

# ========================================
# 環境変数・API設定
# ========================================
DB_URL = os.getenv("DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
sync_openai_client = OpenAI(api_key=OPENAI_API_KEY)

print(f"[起動時] DB_URL設定: {'あり' if DB_URL else 'なし'}")
print(f"[起動時] OpenAI API: {'設定済み' if OPENAI_API_KEY else '未設定'}")

# ========================================
# ChromaDB接続 & 自動構築 (ICFナレッジベース)
# ========================================
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def init_and_build_icf_collection(force_rebuild: bool = False):
    global chroma_client
    collection_name = "icf"
    
    if force_rebuild:
        try:
            chroma_client.delete_collection(collection_name)
            print("🗑️ 既存のICFコレクションを削除しました。")
        except Exception:
            pass

    try:
        collection = chroma_client.get_collection(collection_name)
        if collection.count() > 0 and not force_rebuild:
            print(f"✅ [ChromaDB] ICFコレクション接続成功 (登録件数: {collection.count()})")
            return collection
    except Exception:
        pass

    print("🔄 [ChromaDB] ICFコレクションを新規構築します...")
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    json_path = "icf.json"
    if not os.path.exists(json_path):
        print(f"⚠️ [ChromaDB] {json_path} が見つからないためDB構築をスキップしました。")
        return collection

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            icf_data = json.load(f)
        
        print(f"📄 [ChromaDB] {len(icf_data)} 件のICFデータを登録開始...")
        for i, item in enumerate(icf_data):
            text = f"""コード: {item['code']}
項目名: {item['name']}
説明: {item['description']}
分類: {item['category']}
レベル: {item['level']}"""

            response = sync_openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            embedding = response.data[0].embedding

            collection.add(
                ids=[str(item.get("id", f"icf_{i}"))],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{
                    "code": item["code"],
                    "name": item["name"],
                    "category": item.get("category", ""),
                    "level": item.get("level", "")
                }]
            )
        print(f"✅ [ChromaDB] ICF登録完了 (登録件数: {collection.count()})")
    except Exception as e:
        print(f"❌ [ChromaDB] ICFデータ構築中にエラー: {e}")
    
    return collection

icf_collection = init_and_build_icf_collection(force_rebuild=False)

# ========================================
# データベース接続プールの作成
# ========================================
try:
    db_url = DB_URL
    if db_url and "pooler.supabase.com" in db_url:
        if "sslmode=" not in db_url:
            db_url += ("&" if "?" in db_url else "?") + "sslmode=require"
    
    pg_pool = psycopg2.pool.SimpleConnectionPool(
        1, 10, db_url,
        cursor_factory=RealDictCursor,
        keepalives=1, keepalives_idle=30, keepalives_interval=10,
        keepalives_count=5, connect_timeout=10
    ) if db_url else None
    
    if pg_pool:
        print("✅ [DB接続プール] 作成成功")
except Exception as e:
    print(f"❌ [DB接続プール] 作成失敗: {e}")
    pg_pool = None

def get_db_connection():
    if not pg_pool:
        return None
    try:
        conn = pg_pool.getconn()
        if conn:
            conn.autocommit = True
            return conn
    except Exception as e:
        print(f"❌ [DB接続] 取得失敗: {e}")
        return None

def release_db_connection(conn):
    if not conn or not pg_pool:
        return
    try:
        pg_pool.putconn(conn)
    except Exception as e:
        print(f"⚠️ [DB接続] 解放エラー: {e}")

@atexit.register
def cleanup_pool():
    if pg_pool:
        pg_pool.closeall()
        print("✅ [DB接続プール] クローズ完了")

# ========================================
# グローバル変数 & セッション対話管理
# ========================================
active_session = {}
conversation_history = defaultdict(lambda: deque(maxlen=20))
latest_health = "Normal"
proactive_message_counts = defaultdict(int)

session_state = {
    "current_topic": None,
    "chat_history": [],
    "turn_count": 0
}

class CONFIG:
    PROFILE_ID = 1

# ========================================
# FastAPIアプリ初期化
# ========================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]  
)

# ========================================
# ICF分散取得 ＆ 対話ゴール生成ロジック
# ========================================
class SetupTopicRequest(BaseModel):
    antecedent: str  # 先行事象 A
    behavior: str    # 標的行動 B
    memo: str = ""   # 補足メモ

async def generate_insight_goal(antecedent: str, behavior: str, memo: str, icf_coll, llm_client) -> tuple[list[str], str]:
    query_text = f"場面: {antecedent} 行動: {behavior} 詳細: {memo}"
    
    emb_resp = await llm_client.embeddings.create(
        model="text-embedding-3-small",
        input=query_text
    )
    query_embedding = emb_resp.data[0].embedding

    retrieved_factors = []
    category_patterns = [
        ["b", "body", "心身機能", "身体機能"],
        ["d", "activity", "活動", "活動と参加"],
        ["e", "environment", "環境", "環境因子"]
    ]

    for cat_group in category_patterns:
        for cat in cat_group:
            try:
                res = icf_coll.query(
                    query_embeddings=[query_embedding],
                    n_results=1,
                    where={"category": cat}
                )
                if res and res.get("documents") and len(res["documents"][0]) > 0:
                    retrieved_factors.append(res["documents"][0][0])
                    break
            except Exception:
                continue

    if len(retrieved_factors) < 3:
        try:
            fallback_res = icf_coll.query(
                query_embeddings=[query_embedding],
                n_results=3
            )
            if fallback_res and fallback_res.get("documents") and len(fallback_res["documents"][0]) > 0:
                retrieved_factors = fallback_res["documents"][0]
        except Exception:
            pass

    if not retrieved_factors:
        retrieved_factors = [
            "b140 注意機能: 集中による刺激の見落とし・切り替えにくさ",
            "d240 ストレス対処: 活動中断への抵抗感・悔しさ",
            "e330 指導者の関係: 指示や合図の届きにくさ・曖昧さ"
        ]

    synthesis_prompt = f"""
以下の指導者の記録とICF要因をもとに、児童が対話を通じて自分で気づくべき「原因の自覚」と「具体的な対策」を作成してください。

【指導者記録】
- 場面(A): {antecedent}
- 行動(B): {behavior}
- メモ: {memo}

【抽出されたICF背景要因】
{chr(10).join(retrieved_factors)}

【出力形式】
目標の気づき: （児童が自覚すべき感覚やきっかけ）
目指す対策: （児童が自分で言える具体的な行動や工夫）
"""
    goal_res = await llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": synthesis_prompt}],
        temperature=0.3
    )
    generated_goal = goal_res.choices[0].message.content

    return retrieved_factors, generated_goal

# ========================================
# 画面・APIルート
# ========================================
@app.get("/")
async def read_index():
    return FileResponse('index.html', media_type='text/html')

@app.get("/survey")
async def read_survey():
    return FileResponse('survey.html', media_type='text/html')

@app.post("/api/setup_topic")
async def setup_topic(req: SetupTopicRequest):
    global icf_collection
    if icf_collection is None:
        icf_collection = init_and_build_icf_collection(force_rebuild=False)
    
    factors, goal = await generate_insight_goal(
        antecedent=req.antecedent,
        behavior=req.behavior,
        memo=req.memo,
        icf_coll=icf_collection,
        llm_client=openai_client
    )

    topic_data = {
        "antecedent": req.antecedent,
        "behavior": req.behavior,
        "memo": req.memo,
        "icf_factors": factors,
        "goal": goal
    }
    session_state["current_topic"] = topic_data
    session_state["chat_history"] = []
    session_state["turn_count"] = 0

    print(f"🎯 [対話ゴール設定完了]:\n{goal}")
    return {"status": "success", "topic": topic_data}

@app.post("/api/reset")
async def reset_session():
    session_state["current_topic"] = None
    session_state["chat_history"] = []
    session_state["turn_count"] = 0
    return {"status": "reset"}

# ========================================
# 過去の類似会話ベクトル検索 (conversationsテーブル)
# ========================================
async def find_similar_conversation(user_input: str, development_stage: str = "stage_1", similarity_threshold: float = 0.88):
    resp = await openai_client.embeddings.create(
        input=[user_input],
        model="text-embedding-3-small"
    )
    query_vector = resp.data[0].embedding
    
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return None

        with conn.cursor() as cur:
            # 登録されている会話例からベクトル距離で最も近いものを取得
            cur.execute("""
                SELECT text, fish_text, children_reply_1, children_reply_2,
                       user_embedding <-> %s::vector as distance
                FROM conversations
                ORDER BY distance
                LIMIT 1;
            """, (query_vector,))
            
            result = cur.fetchone()
            if result and result['distance'] < similarity_threshold:
                print(f"[類似会話ヒット] '{result['text']}' (スコア: {result['distance']:.4f})")
                return result
            return None
    except Exception as e:
        print(f"❌ [類似検索エラー] {e}")
        return None
    finally:
        if conn:
            release_db_connection(conn)

# ========================================
# メダカ対話生成（過去会話参照 ＋ ターン誘導型）
# ========================================
async def get_medaka_reply(user_input: str, health_status="不明", similar_example=None, profile_info=None):
    start = time.time()
    
    medaka_state = "元気" if health_status == "Active" else ("元気ない" if health_status == "Lethargic" else "休憩中")
    profile_name = profile_info.get('name', 'お友達') if profile_info else 'お友達'

    current_topic = session_state.get("current_topic")
    goal_context = current_topic["goal"] if current_topic else "児童と楽しく雑談する。"

    session_state["turn_count"] += 1
    turn = session_state["turn_count"]

    # ターン数による対話フェーズの明確な切り替え
    if turn == 1:
        phase_instruction = """
【対話フェーズ1: 雑談・共感】
・まずは児童の発話を受け止めて楽しく共感・会話を広げてください。
・まだ課題や指導内容には直接触れないでください。
"""
    elif turn == 2:
        phase_instruction = f"""
【対話フェーズ2: 自己開示からの問いかけ（重要）】
・児童の返答を受け止めつつ、メダカ自身の体験として自己開示してください（例: 「シロちゃんもエサに夢中だと周りの声聞こえないことあるよ〜」など）。
・その流れで、【目標の気づき】について「〇〇くんも〜なことある？」と優しく問いかけてください。
"""
    else:
        phase_instruction = f"""
【対話フェーズ3: 対策の引き出し】
・児童の気づきを肯定した上で、「じゃあ次はどうしたら気づけそうかな？」と問いかけ、【目指す対策】を児童自身の言葉で言えるように促してください。
"""

    similar_context = ""
    if similar_example:
        similar_context = f"""
【参考とする過去の会話例】
児童:「{similar_example['text']}」
メダカ:「{similar_example['fish_text']}」
※このトーンや雰囲気を参考にしてください。
"""

    system_prompt = f"""
あなたは水槽に住むかわいいメダカ「シロちゃん」です。
口調: やさしいタメ口（〜だよ、〜かな？、シロちゃんもそうだよ〜）。
相手: {profile_name}さん / メダカの状態: {medaka_state}

【対話ゴール】
{goal_context}

{similar_context}

{phase_instruction}

【ルール】
1. 児童を問い詰めたり、命令口調にならないこと。
2. 35文字以内で短く、セリフのみ出力すること。
"""

    messages = [{"role": "system", "content": system_prompt}]
    for hist in session_state["chat_history"][-4:]:
        messages.append(hist)
    messages.append({"role": "user", "content": f"児童:「{user_input}」\nメダカ:"})

    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=80
    )
    reply = response.choices[0].message.content.strip()

    session_state["chat_history"].append({"role": "user", "content": user_input})
    session_state["chat_history"].append({"role": "assistant", "content": reply})

    print(f"[メダカ応答(ターン{turn})] {reply} ({time.time() - start:.2f}秒)")
    return reply

# ========================================
# 音声対話エンドポイント
# ========================================
@app.post("/talk_with_fish_text")
async def talk_with_fish_text(file: UploadFile):
    # 1. 音声認識 (Whisper)
    audio_content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(audio_content)
        temp_audio_path = temp_audio.name
    with open(temp_audio_path, "rb") as audio_file:
        transcript = await openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ja",
            response_format="text"
        )
    os.unlink(temp_audio_path)
    user_input = transcript
    print(f"児童の発話: {user_input}")

    # 2. プロファイル取得
    profile = await get_profile_async(CONFIG.PROFILE_ID)
    child_name = profile["name"]

    # 3. 過去の類似対話検索（ベクトル検索）
    similar_example = await find_similar_conversation(user_input)

    # 4. DB保存（児童発話）
    save_conversation_to_db(
        profile_id=CONFIG.PROFILE_ID,
        speaker=child_name,
        message=user_input,
        health_status=latest_health,
        similar_example_used=(similar_example is not None),
        similar_example_text=similar_example['text'] if similar_example else None,
        similarity_score=similar_example['distance'] if similar_example else None
    )

    # 5. メダカ応答生成（類似対話 ＋ ICFゴール ＋ ターン制御）
    reply_text = await get_medaka_reply(user_input, latest_health, similar_example, profile)

    # 6. DB保存（メダカ発話）
    save_conversation_to_db(
        profile_id=CONFIG.PROFILE_ID,
        speaker='medaka',
        message=reply_text,
        health_status=latest_health
    )

    # 7. 音声ストリーミング返却 (TTS)
    async def audio_stream():
        async with openai_client.audio.speech.with_streaming_response(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=reply_text,
            response_format="mp3",
        ) as response:
            async for chunk in response.iter_bytes():
                yield chunk

    return StreamingResponse(
        audio_stream(),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=reply.mp3"}
    )

# ========================================
# 補助関数・プロフィール・健康状態連携
# ========================================
async def get_profile_async(profile_id: int):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_profile_sync, profile_id)

def get_profile_sync(profile_id: int):
    conn = get_db_connection()
    if conn is None:
        return {"id": profile_id, "name": "ゲスト", "age": 8}
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM profiles WHERE id = %s;", (profile_id,))
            profile = cur.fetchone()
            if not profile:
                return {"id": profile_id, "name": "ゲスト", "age": 8}
            return profile
    finally:
        release_db_connection(conn)

def save_conversation_to_db(profile_id: int, speaker: str, message: str, health_status: str = None,
                            similar_example_used: bool = False, similar_example_text: str = None, similarity_score: float = None):
    conn = get_db_connection()
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversation_history (
                    profile_id, speaker, message, health_status,
                    similar_example_used, similar_example_text, similarity_score
                ) VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;
            """, (profile_id, speaker, message, health_status, similar_example_used, similar_example_text, similarity_score))
            return cur.fetchone()['id']
    except Exception as e:
        print(f"[会話履歴DB] 保存エラー: {e}")
        return None
    finally:
        release_db_connection(conn)

@app.get("/best.onnx")
async def serve_onnx_model():
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        raise HTTPException(404, f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        content = f.read()
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Expose-Headers": "*",
            "Cache-Control": "public, max-age=31536000",
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(content))
        }
    )

@app.post("/update_health")
async def update_health(request: Request):
    global latest_health
    data = await request.json()
    latest_health = data.get("status", "Unknown")
    return {"status": "success", "current_health": latest_health}

@app.post("/set_current_profile")
async def set_current_profile(request: Request):
    data = await request.json()
    profile_id = data.get("profile_id")
    if not profile_id:
        raise HTTPException(400, "profile_id is required")
    CONFIG.PROFILE_ID = profile_id
    return {"success": True, "current_profile_id": CONFIG.PROFILE_ID}

@app.get("/profiles")
async def get_profiles():
    conn = get_db_connection()
    if not conn:
        return [{"id": 1, "name": "ゲスト", "age": 8}]
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, age FROM profiles ORDER BY id;")
            return cur.fetchall()
    finally:
        release_db_connection(conn)

@app.post("/profiles")
async def create_profile(request: Request):
    data = await request.json()
    name = data.get("name")
    age = data.get("age")
    conn = get_db_connection()
    if not conn:
        raise HTTPException(503, "DB接続不可")
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO profiles (name, age, created_at, updated_at)
                VALUES (%s, %s, NOW(), NOW())
                RETURNING id, name, age;
            """, (name, age))
            return cur.fetchone()
    finally:
        release_db_connection(conn)

@app.post("/check_session_status")
async def check_session_status(request: Request):
    return {
        "has_active_session": False,
        "conversation_count": session_state.get("turn_count", 0),
        "proactive_enabled": False
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

