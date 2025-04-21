import base64
import time
from typing import Any

import openai
from openai import OpenAI
import numpy as np

from ..types import MessageList, SamplerBase

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        base_url=None,
        api_key=None,
        logprobs = False
    ):
        self.api_key_name = "OPENAI_API_KEY"
        if base_url and any(provider in base_url for provider in ["google", "databricks", "together"]):
            self.client = OpenAI(base_url=base_url, api_key=api_key)
        else:
            self.client = OpenAI()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"
        self.logprobs = logprobs

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                if self.logprobs:
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=message_list,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            logprobs=self.logprobs
                        )
                    except:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=message_list,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            logprobs=1
                        )
                    
                    try:
                        return response.choices[0].message.content, float(np.exp(np.array([t.logprob for t in response.choices[0].logprobs.content])).mean()), [t.logprob for t in response.choices[0].logprobs.content]
                    except:
                        return response.choices[0].message.content, float(np.exp(response.choices[0].logprobs.token_logprobs).mean()), response.choices[0].logprobs.token_logprobs
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    return response.choices[0].message.content, None, None
                
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
