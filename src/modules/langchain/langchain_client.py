import logging
from typing import Optional, Tuple

from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHAIN_TYPE = "map_reduce"
RETURN_SOURCE_DOCUMENTS = True
VERBOSE_MODE = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class LangChainClient:
    def __init__(self, model_name, max_token):
        """
        Initialize the LangChainClient with any required arguments.
        """
        self.model_name = model_name
        self.max_token = max_token
        # Initialize your client configuration here, like API keys, settings, etc.

    def text_splitter(self):
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=self.model_name,
            chunk_size=self.max_token,
            chunk_overlap=50,
        )

    def youtube_loader(self, url):
        return YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,  # タイトルや再生数も取得できる
            language=["en", "ja"],  # 英語→日本語の優先順位で字幕を取得
        )

    def summarize_youtube_transcript(
        self, llm, document
    ) -> Tuple[Optional[str], Optional[float], Optional[int]]:
        prompt_template = """
Write a concise Japanese summary of the following transcript of Youtube Video.

{text}

ここから日本語で書いてね:
"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
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
                    "token_max": self.max_token,
                },
                return_only_outputs=True,
            )
        return response["output_text"], cb.total_cost, cb.total_tokens

    def build_qa_model(self, retriever, llm) -> None:
        self.qa_model = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=CHAIN_TYPE,
            retriever=retriever,
            return_source_documents=RETURN_SOURCE_DOCUMENTS,
            verbose=VERBOSE_MODE,
        )

    def qa(self, query):
        with get_openai_callback() as cb:
            answer = self.qa_model(query)
        return answer, cb.total_cost, cb.total_tokens
