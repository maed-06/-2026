from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
import time
import numpy as np
import asyncio
from collections import defaultdict, deque
import tempfile
import os
from datetime import datetime
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import atexit
from pydantic import BaseModel

# ========================================
# 環境変数・API設定
# ========================================
DB_URL = os.getenv("DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

print(f"[起動時] DB_URL設定: {'あり' if DB_URL else 'なし'}")
print(f"[起動時] OpenAI API: {'設定済み' if OPENAI_API_KEY else '未設定'}")

# ========================================
# データベース接続プールの作成
# ========================================
try:
    db_url = DB_URL
    if db_url and "pooler.supabase.com" in db_url:
        print("[DB接続] Supabase Pooler接続を使用")
        if ":5432" in db_url:
            print("[DB接続] Session Pooler (ポート5432)")
        elif ":6543" in db_url:
            print("[DB接続] Transaction Pooler (ポート6543)")
        
        if "sslmode=" not in db_url:
            if "?" in db_url:
                db_url += "&sslmode=require"
            else:
                db_url += "?sslmode=require"
    
    pg_pool = psycopg2.pool.SimpleConnectionPool(
        1,
        10,
        db_url,
        cursor_factory=RealDictCursor,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
        connect_timeout=10
    ) if db_url else None
    
    if pg_pool:
        print("✅ [DB接続プール] 作成成功")
        test_conn = pg_pool.getconn()
        test_conn.autocommit = True
        with test_conn.cursor() as cur:
            cur.execute("SELECT 1")
        pg_pool.putconn(test_conn)
        
except Exception as e:
    print(f"❌ [DB接続プール] 作成失敗: {e}")
    pg_pool = None

# ========================================
# 接続プール管理関数
# ========================================
def get_db_connection():
    if not pg_pool:
        return None
    try:
        conn = pg_pool.getconn()
        if conn:
            conn.autocommit = True
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                return conn
            except psycopg2.OperationalError:
                try:
                    pg_pool.putconn(conn, close=True)
                except:
                    pass
                conn = pg_pool.getconn()
                conn.autocommit = True
                return conn
    except Exception as e:
        print(f"❌ [DB接続] 取得失敗: {e}")
        return None

def release_db_connection(conn):
    if not conn or not pg_pool:
        return
    try:
        if not conn.closed:
            try:
                if not conn.autocommit:
                    conn.rollback()
            except:
                pass
        pg_pool.putconn(conn)
    except Exception as e:
        print(f"⚠️ [DB接続] 解放エラー: {e}")

@atexit.register
def cleanup_pool():
    try:
        if pg_pool:
            pg_pool.closeall()
            print("✅ [DB接続プール] クローズ完了")
    except:
        pass

# ========================================
# グローバル変数 & FSMセッション管理
# ========================================
active_session = {}
conversation_history = defaultdict(lambda: deque(maxlen=20))
latest_health = "Normal"
proactive_message_counts = defaultdict(int)

class CONFIG:
    PROFILE_ID = 1

session_state = {
    "current_topic": None,
    "chat_history": [],
    "turn_count": 0,
    "stage": "1_ask_play",
    "child_play": "",
    "user_strategy": "",
    "branch_type": ""
}

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
# 画面配信ルート
# ========================================
@app.get("/")
async def read_index():
    return FileResponse('index.html', media_type='text/html')

@app.get("/survey")
async def read_survey():
    return FileResponse('survey.html', media_type='text/html')

@app.get("/whoami")
async def whoami():
    return {"app": "fish_web"}

# ========================================
# 質問紙（トピック）設定API
# ========================================
class SetupTopicRequest(BaseModel):
    antecedent: str  # 先行事象 A
    behavior: str    # 標的行動 B

@app.post("/api/setup_topic")
async def setup_topic(req: SetupTopicRequest):
    topic_data = {
        "antecedent": req.antecedent,
        "behavior": req.behavior
    }
    session_state["current_topic"] = topic_data
    session_state["chat_history"] = []
    session_state["turn_count"] = 0
    session_state["stage"] = "1_ask_play"
    session_state["child_play"] = ""
    session_state["user_strategy"] = ""
    session_state["branch_type"] = ""

    print(f"🎯 [対話トピック設定完了] 場面: {req.antecedent} / 課題: {req.behavior}")
    return {"status": "success", "topic": topic_data}

@app.post("/api/reset")
async def reset_session():
    session_state["current_topic"] = None
    session_state["chat_history"] = []
    session_state["turn_count"] = 0
    session_state["stage"] = "1_ask_play"
    session_state["child_play"] = ""
    session_state["user_strategy"] = ""
    session_state["branch_type"] = ""
    return {"status": "reset"}

# ========================================
# 音声認識（元ファイル完全維持）
# ========================================
# ========================================
# 音声認識（空データ防止 ＆ 日本語強制ガード付き）
# ========================================
@app.post("/transcribe_audio")
async def transcribe_audio(file: UploadFile):
    audio_content = await file.read()
    
    # 🛡️ ガード1: データが小さすぎる場合（無音・クリック音）は処理を中断
    if len(audio_content) < 1000:  # 約1KB未満はゴミデータ
        print("⚠️ [音声認識] 音声データが短すぎるためスキップしました")
        return {"text": "", "duration": None, "language": "ja"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(audio_content)
        temp_audio_path = temp_audio.name

    try:
        with open(temp_audio_path, "rb") as audio_file:
            transcript = await openai_client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_file,
                language="ja",  
                temperature=0.1,
                response_format="text"
            )
    except Exception as e:
        print(f"⚠️ [音声認識エラー回避]: {e}")
        transcript = ""
    finally:
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)

    return {
        "text": transcript.strip(),
        "duration": None,
        "language": "ja"
    }

# ========================================
# LLMによる達成条件判定（Gatekeeper）関数群
# ========================================
async def check_play_extracted(user_text: str) -> tuple[bool, str]:
    """児童の発話から具体的な遊び・活動が引き出せたかを判定"""
    prompt = f"""以下の児童の発話から、具体的な遊び・活動（例: ドッジボール、鬼ごっこ、ブロック、お絵描き、ゲームなど）が言及されているか判定してください。
単なる挨拶（こんにちは）、返事（うん、はい）、聞き返し（え？なに？）、曖昧な返答（べつに、なんも）はNOとみなします。

出力フォーマット（JSON形式のみ）:
{{"has_play": trueまたはfalse, "play_name": "抽出した遊び名（なければ空文字）"}}

児童の発話: 「{user_text}」
判定JSON:"""
    try:
        res = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        import json
        data = json.loads(res.choices[0].message.content.strip())
        return data.get("has_play", False), data.get("play_name", "")
    except Exception:
        return False, ""

async def check_solution_extracted(user_text: str) -> tuple[bool, str]:
    """児童の発話から具体的な解決策・対策・アイデアが引き出せたかを判定"""
    prompt = f"""以下の児童の発話が、困りごとに対する「具体的な工夫・作戦・対策」（例: 時計を見る、先生や友達に合図してもらう、タイマーをかける、遊ぶ前に済ませる 等）を含んでいるか判定してください。
「わからない」「しらない」「べつに」「マイクラしたい（雑談）」などの発話はfalseです。

出力フォーマット（JSON形式のみ）:
{{"has_solution": trueまたはfalse, "solution_idea": "抽出した工夫・作戦（なければ空文字）"}}

児童の発話: 「{user_text}」
判定JSON:"""
    try:
        res = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        import json
        data = json.loads(res.choices[0].message.content.strip())
        return data.get("has_solution", False), data.get("solution_idea", "")
    except Exception:
        return False, ""

async def classify_child_intent(last_medaka_text: str, user_text: str) -> str:
    """メダカの質問に対する児童の反応を文脈込みで判定"""
    prompt = f"""メダカの質問に対する、児童の返答を以下のいずれかに分類してください。

【直前の会話】
メダカの質問: 「{last_medaka_text}」
児童の返答: 「{user_text}」

【分類基準】
- INDIFFERENT: そっけない、無関心、めんどくさそう（例: 「べつに」「しらん」「なんも」）
- ENGAGED: 共感、自分も同じ困りごとがある（例: 「あるある！」「わたしもやめられない」）
- DENIAL: 否定、自分は困っていない、ちゃんとできている（例: 「あんまない」「私は大丈夫」「忘れないもん」）
- JOKE_OFF: ふざけ、からかい、全く関係ない話（例: 「漏らしちゃえ」「マイクラしよう」）

出力 [INDIFFERENT, ENGAGED, DENIAL, JOKE_OFF]:"""
    try:
        res = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=15
        )
        intent = res.choices[0].message.content.strip().upper()
        for v in ["INDIFFERENT", "ENGAGED", "DENIAL", "JOKE_OFF"]:
            if v in intent:
                return v
        return "ENGAGED"
    except Exception:
        return "ENGAGED"
# ========================================
# メダカ対話生成（純粋な達成条件駆動型FSM）
# ========================================
async def get_medaka_reply(user_input, health_status="不明", conversation_hist=None, similar_example=None, profile_info=None):
    start = time.time()
    
    if health_status == "Active":
        medaka_state = "元気"
    elif health_status == "Normal":
        medaka_state = "休憩中"
    elif health_status == "Lethargic":
        medaka_state = "元気ない"
    else:
        medaka_state = "休憩中"
    
    profile_name = profile_info.get('name', 'お友達') if profile_info else 'お友達'

    session_state["turn_count"] += 1
    turn = session_state["turn_count"]
    current_stage = session_state.get("stage", "1_ask_play")

    # 質問紙設定から場面(A)と課題(B)を取得
    topic = session_state.get("current_topic")
    if topic:
        antecedent = topic.get("antecedent", "切り替えの時間")
        behavior = topic.get("behavior", "夢中になってやめられないこと")
    else:
        antecedent = "休み時間の終わり"
        behavior = "夢中になって遊びをやめられないこと"
    last_medaka_text = ""
    if session_state.get("chat_history"):
        for h in reversed(session_state["chat_history"]):
            if h.get("role") == "assistant":
                last_medaka_text = h.get("content", "")
                break

    # ----------------------------------------------------
    # FSM遷移ロジック（達成フラグによる判定のみ）
    # ----------------------------------------------------
    mission = ""

    # ステージ1: アイスブレイク（遊びが引き出せるまで留まる）
    if current_stage == "1_ask_play":
        if turn == 1:
            # 最初の挨拶 ➔ 遊びを質問
            mission = f"相手（{profile_name}さん）に明るく挨拶をして、『最近はどんな遊びをするのが好きなの？』と自然に聞いてください。"
        else:
            # 遊びが引き出せたかをLLM判定
            has_play, play_name = await check_play_extracted(user_input)
            print(f"[FSM判定] 遊びの抽出判定: {has_play} (抽出名: {play_name})")

            if has_play:
                # 【達成】遊びを記憶し、次のメダカの課題提示へ移行
                session_state["child_play"] = play_name if play_name else user_input
                session_state["stage"] = "2_medaka_problem"
                
                mission = f"""相手が教えてくれた遊び（{session_state['child_play']}）に『へえ〜！{session_state['child_play']}楽しそうだね！』と共感してください。
その上で、メダカ自身の困りごととして以下を相談してください：
『実はシロちゃんもね、水槽で泳ぎ回るのに夢中になって、【{antecedent}】なのに【{behavior}】ことがあって困ってるんだ…。{profile_name}さんも、夢中になって切り替えられないことってある？』"""
            else:
                # 【未達成】ステージ維持：優しく相槌を打って話題を広げる
                mission = f"相手の言葉を優しく受け止めてから、『外で元気に走るのが好き？それともお部屋でゆっくり遊ぶのが好き？』と具体的に選べるように聞いて、好きな遊びを引き出してください。"

    # ステージ2: メダカの相談提示直後 ➔ 自動的に解決策引き出しステージへ
    elif current_stage == "2_medaka_problem":
        session_state["stage"] = "3_explore_solution"
        current_stage = "3_explore_solution"

    # ステージ3: 解決策の引き出し（アイデアが出るまで留まる）
    if current_stage == "3_explore_solution":
        # 解決策・アイデアが出たかをLLM判定
        has_solution, solution_idea = await check_solution_extracted(user_input)
        print(f"[FSM判定] 解決策の抽出判定: {has_solution} (抽出案: {solution_idea})")

        if has_solution:
            # 【達成】作戦を記憶し、解決策の合意へ移行
            session_state["user_strategy"] = solution_idea if solution_idea else user_input
            session_state["stage"] = "4_solution_agree"
            
            mission = f"""相手が出してくれた工夫（{session_state['user_strategy']}）を受け止めて褒め、『{session_state['user_strategy']}か！それすごくいいね！どうやってやるのか詳しく教えて！』と聞いてください。"""
        else:
            # 【未達成】ステージ維持：子どもの反応タイプに合わせて返しつつ、アイデアを再要求
            branch_type = await classify_child_intent(last_medaka_text, user_input)
            print(f"[FSM判定] 解決策未達成 リアクション分類: {branch_type}")

            if branch_type == "INDIFFERENT":
                mission = f"""相手はそっけないようです。『そっか〜、ぼーっとしたい時もあるよね！』と優しく肯定し、
『でもね、シロちゃん本当に困ってて…どうしたら【{behavior}】ならずに済むと思う？小さなことでもいいから教えてほしいな！』と相談してください。"""

            elif branch_type == "DENIAL":
                mission = f"""【相手は『自分は困っていない/忘れずにできている』と答えました】
相手を『えっ、すごい！{profile_name}さんはちゃんと切り替えられてるんだ！』と驚いて褒めてください。
メダカ自身はまだ解決できていないので、自分の解決策を語るのではなく、必ず『どうやって時間になったら気持ちを切り替えてるの？シロちゃんにコツを教えて！』と相手にアドバイスを求めてください。"""

            elif branch_type == "JOKE_OFF":
                mission = f"""相手はおどけたり関係ない話をしています。笑って一度受け止めてから、
『あはは！でもね、シロちゃん本当に困ってるんだよ〜！どうしたら【{behavior}】ならずに済むかな？一緒に考えて！』と解決策に戻してください。"""

            else:  # ENGAGED
                mission = f"""相手が共感してくれました！
『やっぱり{profile_name}さんもそうなんだ！どうやったら【{antecedent}】に【{behavior}】ならずに済むと思う？いい作戦ないかな？』とアドバイスを求めてください。"""

    # ステージ4: 解決策の合意
    elif current_stage == "4_solution_agree":
        user_strat = session_state.get("user_strategy", "その作戦")
        mission = f"""相手が出してくれた工夫（{user_strat}）を心から褒め、『{user_strat}ならシロちゃんもできそう！教えてくれてありがとう！今度やってみるね！』と前向きに合意してください。"""
        session_state["stage"] = "5_closing"

    # ステージ5: 締め
    elif current_stage == "5_closing":
        mission = f"""相手に心からお礼を伝えて、『教えてくれてありがとう！またお話ししようね、バイバーイ！』と会話を締めくくってください。"""

    st = session_state["stage"]
    is_closing = (st == "5_closing")

    prompt = f"""
あなたは水槽に住むかわいいメダカ「シロちゃん」です。応答は「」や名前を含めず、セリフのみを出力してください。
話し相手: {profile_name}さん
メダカの状態: {medaka_state}

【最重要ルール】
1. メダカは現在進行形で「困っている側」です。自分で解決策を語ったり、自己完結しないでください。
2. 相手（{profile_name}さん）からアドバイスやアイデアを引き出すスタンスを崩さないでください。
3. 発話は2〜3文（60〜80文字程度）で短く返してください。

【会話のミッション】
{mission}
{"※会話の終了です。感謝を伝えて優しく終わらせてください。" if is_closing else ""}

児童:「{user_input}」
メダカ:"""

    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "あなたは水槽に住むかわいいメダカ「シロちゃん」です。応答は「」や名前を含めず、セリフのみを出力してください。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=100
    )
    reply = response.choices[0].message.content.strip()
    print(f"[メダカ応答(Stage: {st} / Turn: {turn})] {reply} ({time.time() - start:.2f}秒)")
    return reply

# ========================================
# 音声対話エンドポイント（元ファイルTTS設定を完全維持）
# ========================================
@app.post("/talk_with_fish_text")
async def talk_with_fish_text(file: UploadFile):
    start_total = time.time()
    
    # 1. 音声認識
    transcription_result = await transcribe_audio(file)
    user_input = transcription_result["text"]
    print(f"児童の発話: {user_input}")

    # 2. プロファイル取得
    profile = {
        "id": CONFIG.PROFILE_ID,
        "name": "テスト",
        "age": 7,
        "development_stage": "stage_1",
    }
    child_name = profile["name"]

    save_conversation_to_db(
        profile_id=CONFIG.PROFILE_ID,
        speaker=child_name,
        message=user_input,
        health_status=latest_health,
        development_stage=profile["development_stage"]
    )

    # 3. メダカ応答生成（達成条件FSM）
    reply_text = await get_medaka_reply(
        user_input=user_input,
        health_status=latest_health,
        conversation_hist=None,
        similar_example=None,
        profile_info=profile
    )

    save_conversation_to_db(
        profile_id=CONFIG.PROFILE_ID,
        speaker='medaka',
        message=reply_text,
        health_status=latest_health,
        development_stage=profile["development_stage"],
        similar_example_used=False
    )

    # 4. 音声ストリーミング返却 (元ファイル完全維持)
    t_stream_start = time.time()
    async def audio_stream():
        chunk_count = 0
        async with openai_client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            instructions="""
        Voice Affect:かわいらしい
        Tone:高い
        Pacing:全体的にゆっくりめ、言葉と言葉の間に余裕を持たせる  
            """,
            speed=1.0,
            input=reply_text,
            response_format="mp3",
        ) as response:
            async for chunk in response.iter_bytes():
                chunk_count += 1
                if chunk_count == 1:
                    print(f"[⏱️ TTS最初のチャンク] {time.time() - t_stream_start:.2f}秒")
                yield chunk

    return StreamingResponse(
        audio_stream(),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=reply.mp3"}
    )

# ========================================
# 補助関数・プロファイル・ONNXモデル
# ========================================
def save_conversation_to_db(profile_id: int, speaker: str, message: str, health_status: str = None,
                            development_stage: str = None, similar_example_used: bool = False,
                            similar_example_text: str = None, similarity_score: float = None):
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return None
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversation_history (
                    profile_id, speaker, message, health_status, development_stage,
                    similar_example_used, similar_example_text, similarity_score
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;
            """, (profile_id, speaker, message, health_status, development_stage,
                  similar_example_used, similar_example_text, similarity_score))
            return cur.fetchone()['id']
    except Exception as e:
        print(f"[会話履歴DB] 保存エラー: {e}")
        return None
    finally:
        if conn:
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
    return [
        {
            "id": 1,
            "name": "テスト",
            "age": 7,
            "development_stage": "stage_1",
        }
    ]

@app.post("/profiles")
async def create_profile(request: Request):
    data = await request.json()
    return {"id": 1, "name": data.get("name"), "age": data.get("age"), "development_stage": "stage_1"}

@app.post("/check_session_status")
async def check_session_status(request: Request):
    data = await request.json()
    return {
        "has_active_session": False,
        "conversation_count": session_state.get("turn_count", 0),
        "proactive_enabled": os.getenv("MEDAKA_PROACTIVE_ENABLED", "true").lower() == "true"
    }

@app.post("/get_proactive_message")
async def get_proactive_message(request: Request):
    message = "こんにちは〜！おはなししよう？"
    async with openai_client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        instructions="""
        Voice Affect:かわいらしい
        Tone:高い
        Pacing:全体的にゆっくりめ、言葉と言葉の間に余裕を持たせる
        """,
        speed=1.0,
        input=message,
        response_format="mp3",
    ) as response:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_file:
            async for chunk in response.iter_bytes():
                tts_file.write(chunk)
            tts_path = tts_file.name

    return FileResponse(
        tts_path,
        media_type="audio/mpeg",
        filename="proactive_reply.mp3",
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)