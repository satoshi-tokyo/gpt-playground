import logging

from modules.langchain.langchain_client import LangChainClient
from modules.qdrant.qdrant_service import QdrantService
from modules.streamlit.streamlit_operation import StreamlitOps

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
    langchain_client = LangChainClient(
        stops.st.session_state.model_name, stops.st.session_state.max_token
    )

    container = stops.call_container()
    with container:
        query = stops.query()

        if query:  # Only build QA model and ask if there is a query
            retriever = qs.build_qa_retriever()
            langchain_client.build_qa_model(retriever, llm)

            if langchain_client.qa_model:
                with stops.st.spinner("ChatGPT is typing ..."):
                    answer, total_cost, total_tokens = langchain_client.qa(query)
                    stops.st.session_state.cost = total_cost
                    stops.st.session_state.token = total_tokens

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
    langchain_client = LangChainClient(
        stops.st.session_state.model_name, stops.st.session_state.max_token
    )
    # Create containers once at the beginning to reduce redundancy.
    container = stops.call_container()
    response_container = stops.call_container()

    with container:
        url = stops.get_url_input()

        if url:
            with stops.st.spinner("Fetching Content ..."):
                youtube_loader = langchain_client.youtube_loader(url)
            if youtube_loader:
                text_splitter = langchain_client.text_splitter()
                document = youtube_loader.load_and_split(text_splitter=text_splitter)

                with stops.st.spinner("Storing document ..."):
                    qs.build_vector_store(document)

                with stops.st.spinner("ChatGPT is typing ..."):
                    try:
                        (
                            output_text,
                            total_cost,
                            total_tokens,
                        ) = langchain_client.summarize_youtube_transcript(llm, document)

                        stops.st.session_state.output_text = output_text
                        stops.st.session_state.cost = total_cost
                        stops.st.session_state.token = total_tokens
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
    stops = StreamlitOps()
    qs = QdrantService()

    stops.init_page()
    if stops.selection == "YouTube Summarizer":
        page_youtube_summarizer(stops, qs)
    elif stops.selection == "Ask My Content":
        page_ask_my_content(stops, qs)


if __name__ == "__main__":
    main()
