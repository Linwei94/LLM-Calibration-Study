import base64
import time
from typing import Any

import openai
from openai import OpenAI
import numpy as np
import os

from ..custom_types import MessageList, SamplerBase

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
        temperature: float = 0,
        max_tokens: int = 1024,
        base_url=None,
        api_key=None,
        get_logprobs = False
    ):
        # self.api_key_name = "OPENAI_API_KEY"
        if base_url:
            if any(provider in base_url for provider in ["google", "databricks", "together"]):
                self.client = OpenAI(base_url=base_url, api_key=api_key)
        else:
            OpenAI.api_key = os.environ["OPENAI_API_KEY"]
            self.client = OpenAI()
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
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                if self.get_logprobs:
                    if self.base_url and "together" in self.base_url:
                        print(self.model, "Together API")
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=message_list,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            logprobs=5, # max 5
                            seed=42,
                        )
                        try:
                            self.top_logprobs = response.choices[0].logprobs.top_logprobs # a list of dicts each of which is a dict of possible candidates with its logprob
                        except:
                            print(self.model, "Top logprobs not found")
                        try:
                            self.logprobs = response.choices[0].logprobs.token_logprobs
                        except:
                            print(self.model, "Token logprobs not found")

                        if self.model == "meta-llama/Llama-2-70b-hf":
                            print(response)
                        
                        return response.choices[0].message.content
                    elif self.base_url and "databricks" in self.base_url:
                        print(self.model, "Databricks API")
                        response = self.client.chat.completions.create(
                            messages=message_list, 
                            model=self.model, 
                            max_tokens=self.max_tokens, 
                            logprobs=True,
                            top_logprobs=5,
                            temperature=0
                        )
                        self.top_logprobs = [t.top_logprobs for t in response.choices[0].logprobs.content]
                        self.logprobs = [t.logprob for t in response.choices[0].logprobs.content]
                        return response.choices[0].message.content
                    else:
                        print(self.model, "OpenAI API")
                        response = self.client.chat.completions.create(
                            messages=message_list, 
                            model=self.model, 
                            max_tokens=self.max_tokens, 
                            logprobs=True,
                            top_logprobs=5,
                            temperature=0,
                            seed=42
                        )
                        top_logprob_lst = []
                        for top_list in [t.top_logprobs for t in response.choices[0].logprobs.content]: 
                            top_logprob_lst.append({t.token: t. logprob for t in top_list})
                        self.top_logprobs = top_logprob_lst
                        self.logprobs = [t.logprob for t in response.choices[0].logprobs.content]
                        return response.choices[0].message.content

                else:
                    print(self.model, "No logprobs")
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=message_list,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    return response.choices[0].message.content
                
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = min(2**trial, 60)  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
