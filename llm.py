import openai
import os
from typing import Any

class PromptTemplate:
    def demos(self):
        _demo = (
            "As abbreviations of column names from a table, "
            "c_name | pCd | dt stand for Customer Name | Product Code | Date. "
        )
        return _demo

    def sep_token(self):
        _sep_token = " | "
        return _sep_token


class OpenaiLLM:
    def __init__(
        self, model_name: str, openai_key: str = None, openai_api_key_path: str = None
    ) -> None:
        self.model_name = model_name
        self.openai_key = openai_key
        self.openai_api_key_path = openai_api_key_path
        self.prepare_model(self.openai_key, self.openai_api_key_path)

    def prepare_model(self, openai_key: str, openai_api_key_path: str) -> None:
        if openai_key is None:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key is None:
                assert openai_api_key_path is not None
                # read OpenAI key if needed
                with open(self.open_api_key_path, "r") as f:
                    openai_key = f.read().strip("\n")
        openai.api_key = openai_key

    def __call__(
        self,
        query,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        if self.model_name == "gpt-3.5-turbo" or self.model_name == 'gpt-4':
            completion = openai.ChatCompletion.create(
                model=self.model_name,
                temperature=temperature,
                messages=[{"role": "user", "content": query}],
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content