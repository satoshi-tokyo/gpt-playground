import logging
import os
from typing import List, Optional, Tuple

import streamlit as st
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import GitLoader, YoutubeLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain_core.documents import Document
from modules.streamlit.streamlit_operation import StreamlitOps
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Qdrant params
LOCAL_QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "my_collection1"
SIMILARITY_SEARCH_TYPE = "similarity"
DEFAULT_K = 4

# LancChain params
CHAIN_TYPE = "map_reduce"
RETURN_SOURCE_DOCUMENTS = True
VERBOSE_MODE = True

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def summarize(
    llm: ChatOpenAI, docs: List[Document]
) -> Tuple[Optional[str], Optional[float], Optional[int]]:
    prompt_template = """
Write a concise Japanese summary of the following transcript of Youtube Video.

{text}

ここから日本語で書いてね:
"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    try:
        with get_openai_callback() as cb:
            chain = load_summarize_chain(
                llm,
                chain_type="map_reduce",
                verbose=VERBOSE_MODE,
                map_prompt=PROMPT,
                combine_prompt=PROMPT,
            )
            response = chain(
                {
                    "input_documents": docs,
                    # token_max を指示しないと、GPT3.5など通常の
                    # モデルサイズに合わせた内部処理になってしまうので注意
                    "token_max": st.session_state.max_token,
                },
                return_only_outputs=True,
            )

        return response["output_text"], cb.total_cost, cb.total_tokens

    except Exception as e:
        logger.info(f"An error occurred: {e}")
        return None, None, None


def summarize_repo(
    llm: ChatOpenAI, docs: List[Document]
) -> Tuple[Optional[str], Optional[float], Optional[int]]:
    prompt_template = """
Write a concise summary of the following codes from repository.

{text}

"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    try:
        with get_openai_callback() as cb:
            chain = load_summarize_chain(
                llm,
                chain_type="map_reduce",
                verbose=VERBOSE_MODE,
                map_prompt=PROMPT,
                combine_prompt=PROMPT,
            )
            response = chain(
                {
                    "input_documents": docs,
                    # token_max を指示しないと、GPT3.5など通常の
                    # モデルサイズに合わせた内部処理になってしまうので注意
                    "token_max": st.session_state.max_token,
                },
                return_only_outputs=True,
            )

        return response["output_text"], cb.total_cost, cb.total_tokens

    except Exception as e:
        logger.info(f"An error occurred: {e}")
        return None, None, None


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client based on the environment."""

    use_sqlite = os.getenv("USE_SQLITE", "false").lower() in ["true", "1"]

    # Handle local development case with SQLite backend
    if use_sqlite:
        qdrant_path = os.environ.get("LOCAL_QDRANT_PATH")
        if not qdrant_path:
            raise ValueError("Missing LOCAL_QDRANT_PATH for sqlite configuration.")
        return QdrantClient(path=LOCAL_QDRANT_PATH)

    # Ensure required cloud configuration is present
    qdrant_cloud_endpoint = os.environ.get("QDRANT_CLOUD_ENDPOINT")
    qdrant_cloud_api_key = os.environ.get("QDRANT_CLOUD_API_KEY")
    if not qdrant_cloud_endpoint or not qdrant_cloud_api_key:
        raise ValueError("Missing QDRANT cloud configuration environment variables.")

    return QdrantClient(url=qdrant_cloud_endpoint, api_key=qdrant_cloud_api_key)


def load_qdrant() -> Qdrant:
    """Get Qdrant client based on the environment."""

    try:
        client = get_qdrant_client()

        # Get all collection names
        collection_names = [
            collection.name for collection in client.get_collections().collections
        ]

        # Create collection if it doesn't exist
        if COLLECTION_NAME not in collection_names:
            # Create collection if it doesn't exist
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            logger.info("Collection created.")
        else:
            logger.info("Collection already exists.")

        return Qdrant(
            client=client,
            collection_name=COLLECTION_NAME,
            embeddings=OpenAIEmbeddings(),
        )
    except Exception as e:
        logger.error("An error occurred while loading Qdrant: %s", e)
        raise


def build_vector_store(document) -> None:
    # Initialize embeddings using OpenAIEmbeddings which likely convert the document to vector form.
    embeddings = OpenAIEmbeddings()

    use_sqlite = os.getenv("USE_SQLITE", "false").lower() in ["true", "1"]

    # Handle local development case with SQLite backend
    if use_sqlite:
        qdrant_path = os.environ.get("LOCAL_QDRANT_PATH")
        if not qdrant_path:
            raise ValueError("Missing LOCAL_QDRANT_PATH for sqlite configuration.")

        # Store documents in a QDRANT collection residing at the specified SQLite path.
        Qdrant.from_documents(
            document,
            embeddings,
            path=qdrant_path,
            collection_name=COLLECTION_NAME,
            force_recreate=True,  # Recreate the collection if it already exists.
        )
    else:  # If false, configure for a cloud-based setup using QDRANT's cloud service.
        qdrant_cloud_endpoint = os.getenv("QDRANT_CLOUD_ENDPOINT")
        qdrant_cloud_api_key = os.getenv("QDRANT_CLOUD_API_KEY")

        if not qdrant_cloud_endpoint or not qdrant_cloud_api_key:
            raise ValueError(
                "Missing QDRANT_CLOUD_ENDPOINT or QDRANT_CLOUD_API_KEY for cloud configuration."
            )

        # Store documents in a QDRANT collection on the cloud endpoint using gRPC for improved performance.
        Qdrant.from_documents(
            document,
            embeddings,
            url=qdrant_cloud_endpoint,
            prefer_grpc=True,  # Enable gRPC over HTTP for potentially faster data transfer.
            collection_name=COLLECTION_NAME,
            force_recreate=True,  # Recreate the collection if it already exists.
            api_key=qdrant_cloud_api_key,
        )


def build_qa_model(llm) -> BaseRetrievalQA:
    qdrant = load_qdrant()
    # Configure the retriever with predefined constants.

    retriever = qdrant.as_retriever(
        # Other choise: "mmr", "similarity_score_threshold"
        search_type=SIMILARITY_SEARCH_TYPE,
        # How many times to retreave document (default: 4)
        search_kwargs={"k": DEFAULT_K},
    )

    # Return a configured instance of RetrievalQA.
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=CHAIN_TYPE,
        retriever=retriever,
        return_source_documents=RETURN_SOURCE_DOCUMENTS,
        verbose=VERBOSE_MODE,
    )


def ask(qa, query) -> Tuple[Optional[str], Optional[float], Optional[int]]:
    with get_openai_callback() as cb:
        # query / result / source_documents
        answer = qa(query)

    total_cost = cb.total_cost
    total_tokens = cb.total_tokens

    return answer, total_cost, total_tokens


def page_ask_my_content(stops: StreamlitOps) -> None:
    stops.title_ask_content()

    llm = stops.select_model()
    container = stops.call_container()
    with container:
        query = stops.query()

        if query:  # Only build QA model and ask if there is a query
            if "qa" not in stops.st.session_state:
                stops.st.session_state.qa = build_qa_model(llm)

            qa = stops.st.session_state.qa
            if qa:
                with stops.st.spinner("ChatGPT is typing ..."):
                    answer, cost, token = ask(qa, query)

                stops.st.session_state.costs.append(cost)
                stops.st.session_state.tokens.append(token)

                stops.answer(answer)
            else:
                stops.st.error("Failed to initialize the QA model.")

    stops.display_costs_tokens_sidebar(
        stops.st.session_state.costs, stops.st.session_state.tokens
    )


def page_youtube_summarizer(stops: StreamlitOps) -> None:
    llm = stops.select_model()

    # Create containers once at the beginning to reduce redundancy.
    container = stops.call_container()
    response_container = stops.call_container()

    with container:
        url = stops.get_url_input()

        if url:
            with stops.st.spinner("Fetching Content ..."):
                loader = YoutubeLoader.from_youtube_url(
                    url,
                    add_video_info=True,  # タイトルや再生数も取得できる
                    language=["en", "ja"],  # 英語→日本語の優先順位で字幕を取得
                )

            if loader:
                text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    model_name=stops.st.session_state.model_name,
                    chunk_size=stops.st.session_state.max_token,
                    chunk_overlap=50,
                )
                document = loader.load_and_split(text_splitter=text_splitter)

                with stops.st.spinner("Storing document ..."):
                    build_vector_store(document)

                with stops.st.spinner("ChatGPT is typing ..."):
                    output_text, cost, token = summarize(llm, document)

                if output_text:
                    with response_container:
                        stops.display_summary(output_text, document)

                        # Append costs and tokens after checking for valid output.
                        stops.st.session_state.costs.append(cost)
                        stops.st.session_state.tokens.append(token)

    # Display costs and tokens in sidebar, ensuring they're only displayed once.
    stops.display_costs_tokens_sidebar(
        stops.st.session_state.costs, stops.st.session_state.tokens
    )


def page_github_loader(stops: StreamlitOps) -> None:
    llm = stops.select_model()

    # Create containers once at the beginning to reduce redundancy.
    container = stops.call_container()
    response_container = stops.call_container()

    with container:
        url = stops.get_url_input()
        branch = "main"
        repo_path = "./repo_tmp"
        filter_ext = ".py"

        if url:
            with stops.st.spinner("Fetching Content ..."):
                git_loader = GitLoader(
                    clone_url=url,
                    branch=branch,
                    repo_path=repo_path,
                    file_filter=lambda file_path: file_path.endswith(filter_ext),
                )
            if git_loader:
                document = git_loader.load()
                logger.info(document)

                with stops.st.spinner("Storing document ..."):
                    build_vector_store(document)

                with stops.st.spinner("ChatGPT is typing ..."):
                    output_text, cost, token = summarize_repo(llm, document)

                if output_text:
                    with response_container:
                        stops.display_summary(output_text, document)

                        # Append costs and tokens after checking for valid output.
                        stops.st.session_state.costs.append(cost)
                        stops.st.session_state.tokens.append(token)

    # Display costs and tokens in sidebar, ensuring they're only displayed once.
    stops.display_costs_tokens_sidebar(
        stops.st.session_state.costs, stops.st.session_state.tokens
    )


def main() -> None:
    # TODO optimize use of embedding instance, to resuse instance
    # TODO separate modules
    stops = StreamlitOps()
    stops.init_page()
    selection = stops.selection()
    if selection == "YouTube Summarizer":
        page_youtube_summarizer(stops)
    elif selection == "GitHub Loader":
        page_github_loader(stops)
    elif selection == "Ask My Content":
        page_ask_my_content(stops)


if __name__ == "__main__":
    main()
