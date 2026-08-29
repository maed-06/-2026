import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# =========================
# 1. APIキー読み込み
# =========================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# =========================
# 2. ChromaDB接続
# =========================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection("icf")


# =========================
# 3. 登録件数確認
# =========================

print(f"ICF登録件数：{collection.count()}")


# =========================
# 4. 児童の事例を入力
# =========================

user_input = input("\n児童の事例を入力してください：")


# =========================
# 5. 入力文をEmbedding
# =========================

embedding_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=user_input
)

query_embedding = embedding_response.data[0].embedding


# =========================
# 6. RAGでICF Top 5を検索
# =========================

TOP_K = 5

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=TOP_K,
    include=[
        "documents",
        "metadatas",
        "distances"
    ]
)


# =========================
# 7. RAG検索結果を表示
# =========================

print("\n==============================")
print("RAG検索結果 Top 5")
print("==============================")

for i in range(TOP_K):

    code = results["metadatas"][0][i]["code"]
    name = results["metadatas"][0][i]["name"]
    document = results["documents"][0][i]
    distance = results["distances"][0][i]

    print(f"\n【{i + 1}位】")
    print(f"ICFコード：{code}")
    print(f"項目名：{name}")
    print(f"距離：{distance}")

    print("説明：")

    # documentの中から説明などを表示
    for line in document.strip().split("\n"):
        if "コード:" not in line and "項目名:" not in line:
            print(line)

    print("------------------------------")


# =========================
# 8. GPTに渡すICF候補を作成
# =========================

candidate_text = ""

for i in range(TOP_K):

    code = results["metadatas"][0][i]["code"]
    name = results["metadatas"][0][i]["name"]
    document = results["documents"][0][i]

    candidate_text += f"""
【ICF候補{i + 1}】
コード：{code}
項目名：{name}
説明：
{document}

"""


# =========================
# 9. 気づきを生成するプロンプト
# =========================

prompt = f"""
あなたは、児童の日常生活について振り返る対話を支援するAIです。

以下の「児童の事例」と「RAGによって検索されたICF候補」をもとに、
児童自身が自分の行動や状況について考えるきっかけとなる
「気づきにつながる対話」を3つ作成してください。

【児童の事例】
{user_input}

【ICF候補】
{candidate_text}

### 重要なルール

1. 児童を診断したり、障害特性を断定したりしないでください。

2. 「あなたは○○が苦手です」のような断定的な表現は使わないでください。

3. ICFコードそのものを児童に伝える必要はありません。

4. 児童自身が「そういえば自分はこういうことがあるかも」と
   振り返れるような内容にしてください。

5. 正解を押し付けるのではなく、児童が自分で考えられる
   開かれた問いにしてください。

6. 児童の年齢を考慮し、できるだけ簡単な言葉を使ってください。

7. 児童の事例に直接関係しない内容は避けてください。

8. ICF候補は「気づきを考えるための参考情報」として使用してください。
   ICF候補に無理に合わせる必要はありません。

9. 問題点を指摘するだけではなく、
   「どんなときに起こるか」
   「そのとき自分はどうしているか」
   「どうしたら気づけそうか」
   などを考えられる問いにしてください。

10. 3つの問いは、できるだけ異なる視点から作ってください。

### 出力形式

必ず以下のJSON形式だけで出力してください。

{{
    "awareness_candidates": [
        {{
            "question": "",
            "intention": ""
        }},
        {{
            "question": "",
            "intention": ""
        }},
        {{
            "question": "",
            "intention": ""
        }}
    ]
}}
"""


# =========================
# 10. GPTで気づき候補を生成
# =========================

response = client.responses.create(
    model="gpt-4.1-mini",
    input=prompt
)


# =========================
# 11. 結果表示
# =========================

print("\n==============================")
print("気づきにつながる対話候補")
print("==============================")

print(response.output_text)