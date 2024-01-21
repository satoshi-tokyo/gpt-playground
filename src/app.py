import logging

from langchain.callbacks import get_openai_callback
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from modules.qdrant.qdrant_service import QdrantService
from modules.streamlit.streamlit_operation import StreamlitOps

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


def page_ask_my_content(stops: StreamlitOps, qs: QdrantService) -> None:
    stops.title_ask_content()

    llm = stops.select_model()
    container = stops.call_container()
    with container:
        query = stops.query()

        if query:  # Only build QA model and ask if there is a query
            if "qa" not in stops.st.session_state:
                stops.st.session_state.qa = qs.build_qa_model(llm)

            qa = stops.st.session_state.qa
            if qa:
                with stops.st.spinner("ChatGPT is typing ..."):
                    with get_openai_callback() as cb:
                        # query / result / source_documents
                        answer = qa(query)

                    stops.st.session_state.cost = cb.total_cost
                    stops.st.session_state.token = cb.total_tokens

                stops.st.session_state.costs.append(stops.st.session_state.cost)
                stops.st.session_state.tokens.append(stops.st.session_state.token)

                stops.answer(answer)
            else:
                stops.st.error("Failed to initialize the QA model.")

    stops.display_costs_tokens_sidebar(
        stops.st.session_state.costs, stops.st.session_state.tokens
    )


def page_youtube_summarizer(stops: StreamlitOps, qs: QdrantService) -> None:
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
                    qs.build_vector_store(document)

                with stops.st.spinner("ChatGPT is typing ..."):
                    prompt_template = """
Write a concise Japanese summary of the following transcript of Youtube Video.

{text}

ここから日本語で書いてね:
"""
                    PROMPT = PromptTemplate(
                        template=prompt_template, input_variables=["text"]
                    )
                    try:
                        with get_openai_callback() as cb:
                            chain = load_summarize_chain(
                                llm,
                                chain_type=CHAIN_TYPE,
                                verbose=VERBOSE_MODE,
                                map_prompt=PROMPT,
                                combine_prompt=PROMPT,
                            )
                            response = chain(
                                {
                                    "input_documents": document,
                                    # token_max を指示しないと、GPT3.5など通常の
                                    # モデルサイズに合わせた内部処理になってしまう
                                    "token_max": stops.st.session_state.max_token,
                                },
                                return_only_outputs=True,
                            )
                        stops.st.session_state.output_text = response["output_text"]
                        stops.st.session_state.cost = cb.total_cost
                        stops.st.session_state.token = cb.total_tokens
                    except Exception as e:
                        logger.info(f"An error occurred: {e}")

                if stops.st.session_state.output_text:
                    with response_container:
                        stops.display_summary(
                            stops.st.session_state.output_text, document
                        )

                        # Append costs and tokens after checking for valid output.
                        stops.st.session_state.costs.append(stops.st.session_state.cost)
                        stops.st.session_state.tokens.append(
                            stops.st.session_state.token
                        )

    # Display costs and tokens in sidebar, ensuring they're only displayed once.
    stops.display_costs_tokens_sidebar(
        stops.st.session_state.costs, stops.st.session_state.tokens
    )


def main() -> None:
    # TODO optimize use of embedding instance, to resuse instance
    # TODO separate modules
    stops = StreamlitOps()
    qs = QdrantService()

    stops.init_page()
    selection = stops.selection()
    if selection == "YouTube Summarizer":
        page_youtube_summarizer(stops, qs)
    elif selection == "Ask My Content":
        page_ask_my_content(stops, qs)


if __name__ == "__main__":
    main()
