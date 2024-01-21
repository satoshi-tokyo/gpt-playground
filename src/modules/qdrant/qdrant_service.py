import logging
import os

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

LOCAL_QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "my_collection1"
SIMILARITY_SEARCH_TYPE = "similarity"
DEFAULT_K = 4


class QdrantService:
    def __init__(self, host="localhost", port=6333):
        self.local_qdrant_path = LOCAL_QDRANT_PATH
        self.collection_name = COLLECTION_NAME
        self.similarity_search_type = SIMILARITY_SEARCH_TYPE
        self.default_k = DEFAULT_K
        self.use_sqlite = os.getenv("USE_SQLITE", "false").lower() in ["true", "1"]
        self.qdrant_cloud_endpoint = os.environ.get("QDRANT_CLOUD_ENDPOINT")
        self.qdrant_cloud_api_key = os.environ.get("QDRANT_CLOUD_API_KEY")
        self.client = self.get_qdrant_client()

    def get_qdrant_client(self) -> QdrantClient:
        """Get Qdrant client based on the environment."""

        # Handle local development case with SQLite backend
        if self.use_sqlite:
            if not self.local_qdrant_path:
                raise ValueError("Missing LOCAL_QDRANT_PATH for sqlite configuration.")
            return QdrantClient(path=self.local_qdrant_path)

        # Ensure required cloud configuration is present
        if not self.qdrant_cloud_endpoint or not self.qdrant_cloud_api_key:
            raise ValueError(
                "Missing QDRANT cloud configuration environment variables."
            )

        return QdrantClient(
            url=self.qdrant_cloud_endpoint, api_key=self.qdrant_cloud_api_key
        )

    def load_qdrant(self) -> Qdrant:
        """Get Qdrant client based on the environment."""

        try:
            # Get all collection names
            collection_names = [
                collection.name
                for collection in self.client.get_collections().collections
            ]

            # Create collection if it doesn't exist
            if self.collection_name not in collection_names:
                # Create collection if it doesn't exist
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
                logger.info("Collection created.")
            else:
                logger.info("Collection already exists.")

            return Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=OpenAIEmbeddings(),
            )
        except Exception as e:
            logger.error("An error occurred while loading Qdrant: %s", e)
            raise

    def build_qa_retriever(self):
        # TODO type hint retriever
        qdrant = self.load_qdrant()

        # Configure the retriever with predefined constants.
        return qdrant.as_retriever(
            # Other choise: "mmr", "similarity_score_threshold"
            search_type=self.similarity_search_type,
            # How many times to retreave document (default: 4)
            search_kwargs={"k": self.default_k},
        )

    def build_vector_store(self, document) -> None:
        # Initialize embeddings using OpenAIEmbeddings which likely convert the document to vector form.
        embeddings = OpenAIEmbeddings()

        # Handle local development case with SQLite backend
        if self.use_sqlite:
            if not self.local_qdrant_path:
                raise ValueError("Missing LOCAL_QDRANT_PATH for sqlite configuration.")

            # Store documents in a QDRANT collection residing at the specified SQLite path.
            Qdrant.from_documents(
                document,
                embeddings,
                path=self.local_qdrant_path,
                collection_name=self.collection_name,
                force_recreate=True,  # Recreate the collection if it already exists.
            )
        else:  # If false, configure for a cloud-based setup using QDRANT's cloud service.
            if not self.qdrant_cloud_endpoint or not self.qdrant_cloud_api_key:
                raise ValueError(
                    "Missing QDRANT_CLOUD_ENDPOINT or QDRANT_CLOUD_API_KEY for cloud configuration."
                )

            # Store documents in a QDRANT collection on the cloud endpoint using gRPC for improved performance.
            Qdrant.from_documents(
                document,
                embeddings,
                url=self.qdrant_cloud_endpoint,
                prefer_grpc=True,  # Enable gRPC over HTTP for potentially faster data transfer.
                collection_name=self.collection_name,
                force_recreate=True,  # Recreate the collection if it already exists.
                api_key=self.qdrant_cloud_api_key,
            )


# Usage example
if __name__ == "__main__":
    service = QdrantService()
