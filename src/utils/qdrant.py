# pip install pycryptodome
import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "my_collection"


def load_qdrant():
    client = QdrantClient(
        url=os.environ.get("QDRANT_CLOUD_ENDPOINT"),
        api_key=os.environ.get("QDRANT_CLOUD_API_KEY"),
    )
    # For sqlite3, local use
    # client = QdrantClient(path=QDRANT_PATH)

    # すべてのコレクション名を取得
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    print(collection_names)

    # コレクションが存在しなければ作成
    if COLLECTION_NAME not in collection_names:
        # コレクションが存在しない場合、新しく作成します
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print("collection created")

    return Qdrant(
        client=client, collection_name=COLLECTION_NAME, embeddings=OpenAIEmbeddings()
    )


def main():
    load_qdrant()


if __name__ == "__main__":
    main()
