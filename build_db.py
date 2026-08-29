import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# =========================
# 初期設定
# =========================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ChromaDB
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

# 既存のコレクションを削除
try:
    chroma_client.delete_collection("icf")
    print("既存のICFデータを削除しました。")
except Exception:
    pass

# 新しく作成
collection = chroma_client.create_collection(
    name="icf"
)


# =========================
# icf.json読み込み
# =========================

with open("icf.json", "r", encoding="utf-8") as f:
    icf_data = json.load(f)

print(f"ICFデータ数: {len(icf_data)}")


# =========================
# Embeddingして登録
# =========================

for i, item in enumerate(icf_data):

    # Embeddingする文章
    text = f"""
コード: {item['code']}
項目名: {item['name']}
説明: {item['description']}
分類: {item['category']}
レベル: {item['level']}
"""

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    embedding = response.data[0].embedding

    collection.add(
        ids=[item["id"]],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{
            "code": item["code"],
            "name": item["name"],
            "category": item["category"],
            "level": item["level"]
        }]
    )

    print(f"{i + 1}/{len(icf_data)} 登録: {item['code']}")


print("\n==============================")
print("✅ ChromaDBの作成が完了しました！")
print(f"登録件数: {collection.count()}")
print("==============================")