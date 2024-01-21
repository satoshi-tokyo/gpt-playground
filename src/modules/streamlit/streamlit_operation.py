from typing import Optional

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from streamlit.delta_generator import DeltaGenerator


class StreamlitOps:
    def __init__(self):
        self.st = st
        self.st.session_state.output_text = ""
        self.st.session_state.cost = None
        self.st.session_state.token = None

    def init_page(self) -> None:
        self.st.set_page_config(page_title="YouTube Summarizer")
        self.st.header("YouTube Summarizer")
        self.st.sidebar.title("Options")
        self.st.session_state.costs = []
        self.st.session_state.tokens = []

    def call_container(self) -> DeltaGenerator:
        return self.st.container()

    def selection(self) -> Optional[str]:
        return self.st.sidebar.radio("Go to", ["YouTube Summarizer", "Ask My Content"])

    def select_model(self) -> ChatOpenAI:
        model = self.st.sidebar.radio("Choose a model:", ("GPT-3.5-turbo", "GPT-4"))

        if model == "GPT-3.5-turbo":
            self.st.session_state.model_name = "gpt-3.5-turbo"
        else:
            self.st.session_state.model_name = "gpt-4"
        # 300: 本文以外の指示のtoken数 (以下同じ)
        self.st.session_state.max_token = (
            OpenAI.modelname_to_contextsize(self.st.session_state.model_name) - 300
        )
        return ChatOpenAI(model=self.st.session_state.model_name, temperature=0)

    def get_url_input(self) -> str | None:
        url = self.st.text_input("Youtube URL: ", key="input")
        return url

    def title_ask_content(self) -> None:
        self.st.title("Ask My Content")

    def query(self) -> str:
        """Query for contents"""
        return self.st.text_input("Query: ", key="input")

    def answer(self, answer) -> Optional[str]:
        """Answer for contents"""
        self.st.markdown("## Answer")
        return self.st.write(answer)

    def display_summary(self, output_text, document) -> None:
        self.st.markdown("## Summary")
        self.st.write(output_text)
        self.st.markdown("---")
        self.st.markdown("## Original Text")
        self.st.write(document)

    def display_costs_tokens_sidebar(self, costs, tokens) -> None:
        self.st.sidebar.markdown("## Costs")
        self.st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
        for cost in costs:
            self.st.sidebar.markdown(f"- ${cost:.5f}")

        self.st.sidebar.markdown("## Tokens")
        self.st.sidebar.markdown(f"**Total count: {sum(tokens)}**")
        for token in tokens:
            self.st.sidebar.markdown(f"- {token}")


if __name__ == "__main__":
    pass
