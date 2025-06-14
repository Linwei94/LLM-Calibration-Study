from typing import Any, Dict, Union
from ..classes import MessageList, SamplerBase
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os


class HFChatCompletionSampler(SamplerBase):
    def __init__(
        self,
        model: str,
        model_dir: Union[str, None] = None,
        API_TOKEN: Union[str, None] = os.environ.get("HF_TOKEN", None),
        system_message: Union[str, None] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_vllm = False,
        think = False 
    ):
        if model_dir:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        else:
            if "gguf" in model.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(model.split("@")[0].replace("-GGUF", ""), trust_remote_code=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if not use_vllm:
            self.client: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager")
        self.model = model
        self.system_message = system_message
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.logprobs = None
        self.top_logprobs = None
        self.base_url = ""
        self.get_logprobs = True
        self.think = think

    def get_hf_model(self):
        return AutoModelForCausalLM.from_pretrained(self.model, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager")

    def _pack_message(self, role: str, content: Any) -> Dict:
        return {"role": str(role), "content": str(content)}

    def _pack_message_to_string(self, message_list: MessageList) -> str:
        prompt = ""
        for msg in message_list:
            if msg["role"] == "system":
                prompt += f"{msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\nAssistant: "
            elif msg["role"] == "assistant":
                prompt += f"{msg['content']}\n"

        return prompt

    def __call__(self, message_list: MessageList) -> str:
        retry = 0
        # Format messages into prompt
        for i in range(len(message_list)):
            if not self.think and "qwen" in self.model.lower():
                message_list[i]["content"] = "/no_think " + message_list[i]["content"]

        while True:
            if self.system_message:
                message_list = [self._pack_message("system", self.system_message)] + message_list

            # Generate response
            inputs = self.tokenizer(self.tokenizer.apply_chat_template(message_list, tokenize=False, add_generation_prompt=True), return_tensors="pt").to(self.client.device)
            with torch.no_grad():
                outputs = self.client.generate(
                    inputs['input_ids'], 
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=self.max_new_tokens, 
                    return_dict_in_generate=True, 
                    output_logits=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            self.logprobs = self.client.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True).numpy(force=True)
            print(self.logprobs)
            response = "".join(self.tokenizer.batch_decode(outputs.sequences[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
            print(response)

            # Extract assistant response
            try:
                return response.split("Assistant: ")[-1].strip()
            except Exception:
                retry += 1
                print("Retry:", retry)
                continue