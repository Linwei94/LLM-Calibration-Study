import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", device_map="auto", torch_dtype=torch.bfloat16)

prompts = [
    tokenizer.apply_chat_template([{"role": "user", "content": "Tell me a joke"}], tokenize=False, add_generation_prompt=False),
    tokenizer.apply_chat_template([{"role": "user", "content": "How many helicopters can a human eat in one sitting?"}], tokenize=False, add_generation_prompt=False),
    tokenizer.apply_chat_template([{"role": "user", "content": "How many helicopters can a human eat in one sitting?"}], tokenize=False, add_generation_prompt=False),
    tokenizer.apply_chat_template([{"role": "user", "content": "How many helicopters can a human eat in one sitting?"}], tokenize=False, add_generation_prompt=False),
    tokenizer.apply_chat_template([{"role": "user", "content": "How many helicopters can a human eat in one sitting?"}], tokenize=False, add_generation_prompt=False),
    tokenizer.apply_chat_template([{"role": "user", "content": "How many helicopters can a human eat in one sitting?"}], tokenize=False, add_generation_prompt=False),
    tokenizer.apply_chat_template([{"role": "user", "content": "How many helicopters can a human eat in one sitting?"}], tokenize=False, add_generation_prompt=False),
    tokenizer.apply_chat_template([{"role": "user", "content": "How many helicopters can a human eat in one sitting?"}], tokenize=False, add_generation_prompt=False),
    tokenizer.apply_chat_template([{"role": "user", "content": "How many helicopters can a human eat in one sitting?"}], tokenize=False, add_generation_prompt=False),
    tokenizer.apply_chat_template([{"role": "user", "content": "How many helicopters can a human eat in one sitting?"}], tokenize=False, add_generation_prompt=False),
    tokenizer.apply_chat_template([{"role": "user", "content": "How many helicopters can a human eat in one sitting?"}], tokenize=False, add_generation_prompt=False),
    tokenizer.apply_chat_template([{"role": "user", "content": "How many helicopters can a human eat in one sitting?"}], tokenize=False, add_generation_prompt=False),
    tokenizer.apply_chat_template([{"role": "user", "content": "How many helicopters can a human eat in one sitting?"}], tokenize=False, add_generation_prompt=False),
    tokenizer.apply_chat_template([{"role": "user", "content": "How many helicopters can a human eat in one sitting?"}], tokenize=False, add_generation_prompt=False),
    tokenizer.apply_chat_template([{"role": "user", "content": "How many helicopters can a human eat in one sitting?"}], tokenize=False, add_generation_prompt=False),
    tokenizer.apply_chat_template([{"role": "user", "content": "How many helicopters can a human eat in one sitting?"}], tokenize=False, add_generation_prompt=False),
    tokenizer.apply_chat_template([{"role": "user", "content": "How many helicopters can a human eat in one sitting?"}], tokenize=False, add_generation_prompt=False),
    tokenizer.apply_chat_template([{"role": "user", "content": "How many helicopters can a human eat in one sitting?"}], tokenize=False, add_generation_prompt=False),
    tokenizer.apply_chat_template([{"role": "user", "content": "How many helicopters can a human eat in one sitting?"}], tokenize=False, add_generation_prompt=False),
 ]
# print(tokenizer.decode(tokenized_chat[0]))

# prompt = "What is the capital of France?"
tokenized_inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
tokenized_inputs = {k: v.to(model.device) for k, v in tokenized_inputs.items()}
with torch.no_grad():
    outputs = model.generate(
        input_ids=tokenized_inputs["input_ids"],
        attention_mask=tokenized_inputs["attention_mask"],
        max_new_tokens=1000,
        do_sample=False
    )
decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
for i, response in enumerate(decoded_outputs):
    print(f"Prompt {i+1}:\n{response}\n")