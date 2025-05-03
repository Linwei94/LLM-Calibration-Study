import base64
import time
from typing import Any

import numpy as np
import os
from google import genai
from google.genai import types

from ..custom_types import MessageList, SamplerBase

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class GoogleAISampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        system_message: str | None = None,
        temperature: float = 0,
        max_tokens: int = 1024,
        base_url=None,
        api_key=os.environ["GOOGLE_AI_KEY"],
        get_logprobs = False
    ):
        self.client = genai.Client(api_key=api_key)
        self.base_url = base_url
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"
        self.get_logprobs = get_logprobs
        self.logprobs = None
        self.top_logprobs = None

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
        trial = 0
        contents = [m["content"] for m in message_list]
        while True:
            try:
                if self.get_logprobs:
                    print(self.model, "Sampling with avg logprobs")
                    response = self.client.models.generate_content(contents=contents, 
                                                                   model=self.model, config=types.GenerateContentConfig(
                                                                    system_instruction=self.system_message,
                                                                    max_output_tokens=self.max_tokens,
                                                                    temperature=0))
                    self.logprobs = [response.candidates[0].avg_logprobs]
                else:
                    print(self.model, "Sampling without avg logprobs")
                    response = self.client.models.generate_content(contents=contents, 
                                                                   model=self.model, config=types.GenerateContentConfig(
                                                                    system_instruction=self.system_message,
                                                                    max_output_tokens=self.max_tokens,
                                                                    temperature=0))
                return response.text
                
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except Exception as e:
                exception_backoff = min(2**trial, 60)  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
