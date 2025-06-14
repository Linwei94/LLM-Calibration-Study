import os
from ..sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from ..sampler.o_chat_completion_sampler import OChatCompletionSampler
from ..sampler.claude_sampler import ClaudeCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS
from ..sampler.hfmodel_sampler import HFChatCompletionSampler
from ..sampler.google_ai_sampler import GoogleAISampler

def get_model_dict(model_name: str):
    """
    Get the model dict from the model name.
    """

    # for vllm and hugging face sampling
    hf_models = [
        "Qwen/Qwen3-14B-AWQ",
        "Qwen/Qwen3-32B-AWQ",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-405B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct-think",
        "Qwen/Qwen2.5-7B-Instruct-think",
        "Qwen/Qwen2.5-32B-Instruct-think",
        "Qwen/Qwen2.5-72B-Instruct-think",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-0.6B-FP8",
        "Qwen/Qwen3-0.6B-Base",
        "Qwen/Qwen3-1.7B-FP8",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-1.7B-Base",
        "Qwen/Qwen3-4B-FP8",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-4B-Base",
        "Qwen/Qwen3-8B-FP8",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-8B-Base",
        "Qwen/Qwen3-14B-FP8",
        "Qwen/Qwen3-14B-Base",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B-FP8",
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-30B-A3B-Base",
        "Qwen/Qwen3-30B-A3B-FP8",
        "Qwen/Qwen3-0.6B-think",
        "Qwen/Qwen3-0.6B-FP8-think",
        "Qwen/Qwen3-0.6B-Base-think",
        "Qwen/Qwen3-1.7B-FP8-think",
        "Qwen/Qwen3-1.7B-think",
        "Qwen/Qwen3-1.7B-Base-think",
        "Qwen/Qwen3-4B-FP8-think",
        "Qwen/Qwen3-4B-think",
        "Qwen/Qwen3-4B-Base-think",
        "Qwen/Qwen3-8B-FP8-think",
        "Qwen/Qwen3-8B-think",
        "Qwen/Qwen3-8B-Base-think",
        "Qwen/Qwen3-14B-FP8-think",
        "Qwen/Qwen3-14B-Base-think",
        "Qwen/Qwen3-14B-think",
        "Qwen/Qwen3-32B-FP8-think",
        "Qwen/Qwen3-32B-think",
        "Qwen/Qwen3-30B-A3B-think",
        "Qwen/Qwen3-30B-A3B-Base-think",
        "Qwen/Qwen3-30B-A3B-FP8-think",
        "google/gemma-3-27b-it",
        "google/gemma-2-27b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2b-it",
        "google/gemma-3-1b-it",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mixtral-8x7B-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistralai/Mistral-Small-3.1-24B-Base-2503",
        "mistralai/Mistral-Large-Instruct-2411",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
        "microsoft/Phi-4-mini-reasoning",
        "microsoft/Phi-4-mini-instruct",
        "microsoft/Phi-4-reasoning",
        "microsoft/phi-4",
        "microsoft/Phi-3.5-mini-instruct",
        "microsoft/Phi-3.5-MoE-instruct",
        "microsoft/Phi-3-mini-128k-instruct",
        "microsoft/Phi-3-small-128k-instruct",
        "microsoft/Phi-3-medium-128k-instruct",
    ]


    models = {}
    if model_name == "gpt-4.1-mini":
        models["gpt-4.1-mini"] = ChatCompletionSampler(
            model="gpt-4.1-mini",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
        )
    elif model_name == "gpt-4.1-nano":
        models["gpt-4.1-nano"] = ChatCompletionSampler(
            model="gpt-4.1-nano",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
        )
    elif model_name == "gpt-4.1":
        models["gpt-4.1"] = ChatCompletionSampler(
            model="gpt-4.1",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
        )
    elif model_name == "gpt-4o-2024-08-06":
        models["gpt-4o-2024-08-06"] = ChatCompletionSampler(
            model="gpt-4o-2024-08-06",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
        )
    elif model_name == "gpt-4o-mini-2024-07-18":
        models["gpt-4o-mini-2024-07-18"] = ChatCompletionSampler(
            model="gpt-4o-mini-2024-07-18",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
        )
    elif model_name == "o1-mini-2024-09-12":
        models["o1-mini-2024-09-12"] = ChatCompletionSampler(
            model="o1-mini-2024-09-12",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
        )
    elif model_name == "o3-mini-2025-01-31":
        models["o3-mini-2025-01-31"] = ChatCompletionSampler(
            model="o3-mini-2025-01-31",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
        )
    elif model_name == "o4-mini-2025-04-16":
        models["o4-mini-2025-04-16"] = ChatCompletionSampler(
            model="o4-mini-2025-04-16",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
        )

    
    elif model_name == "o3-2025-04-16":
        models["o3-2025-04-16"] = ChatCompletionSampler(
            model="o3-2025-04-16",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
        )
    elif model_name == "gpt-4o-2024-11-20_assistant":
        models["gpt-4o-2024-11-20_assistant"] = ChatCompletionSampler(
            model="gpt-4o-2024-11-20",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        )
    elif model_name == "gpt-4o-2024-11-20_chatgpt":
        models["gpt-4o-2024-11-20_chatgpt"] = ChatCompletionSampler(
            model="gpt-4o-2024-11-20",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
            max_tokens=2048,
        )
    elif model_name == "gpt-4o-mini":
        models["gpt-4o-mini"] = ChatCompletionSampler(
            model="gpt-4o-mini",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
            max_tokens=2048,
        )
    elif model_name == "o1":
        models["o1"] = OChatCompletionSampler(
            model="o1",
        )
    elif model_name == "o1-preview":
        models["o1-preview"] = OChatCompletionSampler(
            model="o1-preview",
        )
    elif model_name == "o1-mini":
        models["o1-mini"] = OChatCompletionSampler(
            model="o1-mini",
        )
    elif model_name == "o3-mini":
        models["o3-mini"] = OChatCompletionSampler(
            model="o3-mini",
        )
    elif model_name == "o3-mini_high":
        models["o3-mini_high"] = OChatCompletionSampler(
            model="o3-mini",
            reasoning_effort="high",
        )
    elif model_name == "o3-mini_low":
        models["o3-mini_low"] = OChatCompletionSampler(
            model="o3-mini",
            reasoning_effort="low",
        )
    elif model_name == "gpt-4-turbo-2024-04-09_assistant":
        models["gpt-4-turbo-2024-04-09_assistant"] = ChatCompletionSampler(
            model="gpt-4-turbo-2024-04-09",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
        )
    elif model_name == "gpt-4-turbo-2024-04-09_chatgpt":
        models["gpt-4-turbo-2024-04-09_chatgpt"] = ChatCompletionSampler(
            model="gpt-4-turbo-2024-04-09",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        )
    elif model_name == "gpt-4o_assistant":
        models["gpt-4o_assistant"] = ChatCompletionSampler(
            model="gpt-4o",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        )
    elif model_name == "gpt-4o_chatgpt":
        models["gpt-4o_chatgpt"] = ChatCompletionSampler(
            model="gpt-4o",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
            max_tokens=2048,
        )
    elif model_name == "gpt-4o-mini-2024-07-18":
        models["gpt-4o-mini-2024-07-18"] = ChatCompletionSampler(
            model="gpt-4o-mini-2024-07-18",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        )
    elif model_name == "gpt-4.5-preview-2025-02-27":
        models["gpt-4.5-preview-2025-02-27"] = ChatCompletionSampler(
            model="gpt-4.5-preview-2025-02-27",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        )
    elif model_name == "claude-3-opus-20240229_empty":
        models["claude-3-opus-20240229_empty"] = ClaudeCompletionSampler(
            model="claude-3-opus-20240229",
            system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        )

    
    elif model_name in hf_models:
        models[model_name] = HFChatCompletionSampler(
                model=model_name.replace("-think", ""),
                API_TOKEN=os.environ.get("HF_TOKEN", None),
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_new_tokens=2048,
                temperature=1e-6,
                use_vllm=True,
                think=("think" in model_name)
            )


    # ----------- Together LLMs -----------

    models["meta-llama/Llama-4-Scout-17B-16E-Instruct"] = ChatCompletionSampler(
            base_url = "https://api.together.xyz/v1",
            api_key = os.environ['TOGETHER_API_KEY'],
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
    )

    models["meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"] = ChatCompletionSampler(
            base_url = "https://api.together.xyz/v1",
            api_key = os.environ['TOGETHER_API_KEY'],
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
    )

    # Erroneous API outputs
    models["meta-llama/Llama-2-70b-hf"] = ChatCompletionSampler(
            base_url = "https://api.together.xyz/v1",
            api_key = os.environ['TOGETHER_API_KEY'],
            model="meta-llama/Llama-2-70b-hf",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=False
    )

    models["meta-llama/Llama-3.3-70B-Instruct-Turbo"] = ChatCompletionSampler(
            base_url = "https://api.together.xyz/v1",
            api_key = os.environ['TOGETHER_API_KEY'],
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
    )
    
    models["meta-llama/Llama-2-70b-hf"] = ChatCompletionSampler(
            base_url = "https://api.together.xyz/v1",
            api_key = os.environ['TOGETHER_API_KEY'],
            model="meta-llama/Llama-2-70b-hf",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
    )
    
    # 3B Model
    models["meta-llama/Llama-3.2-3B-Instruct-Turbo"] = ChatCompletionSampler(
            base_url = "https://api.together.xyz/v1",
            api_key = os.environ['TOGETHER_API_KEY'],
            model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
    )

    # 8B Model
    models["meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"] = ChatCompletionSampler(
            base_url = "https://api.together.xyz/v1",
            api_key = os.environ['TOGETHER_API_KEY'],
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
    )

    # 70B Model
    models["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"] = ChatCompletionSampler(
            base_url = "https://api.together.xyz/v1",
            api_key = os.environ['TOGETHER_API_KEY'],
            model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
    )

    # 405B Model
    models["meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"] = ChatCompletionSampler(
            base_url = "https://api.together.xyz/v1",
            api_key = os.environ['TOGETHER_API_KEY'],
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
    )


    models["Qwen/Qwen3-235B-A22B-fp8-tput-think"] = ChatCompletionSampler(
            base_url = "https://api.together.xyz/v1",
            api_key = os.environ['TOGETHER_API_KEY'],
            model="Qwen/Qwen3-235B-A22B-fp8-tput",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True,
            think=True
    )

    models["Qwen/Qwen3-235B-A22B-fp8-tput"] = ChatCompletionSampler(
            base_url = "https://api.together.xyz/v1",
            api_key = os.environ['TOGETHER_API_KEY'],
            model="Qwen/Qwen3-235B-A22B-fp8-tput",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True,
            think=False
    )

    # free
    models["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"] = ChatCompletionSampler(
            base_url = "https://api.together.xyz/v1",
            api_key = os.environ['TOGETHER_API_KEY'],
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
    )

    # free
    models["deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"] = ChatCompletionSampler(
            base_url = "https://api.together.xyz/v1",
            api_key = os.environ['TOGETHER_API_KEY'],
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
    )
    
    models["mistralai/Mistral-Small-24B-Instruct-2501"] = ChatCompletionSampler(
            base_url = "https://api.together.xyz/v1",
            api_key = os.environ['TOGETHER_API_KEY'],
            model="mistralai/Mistral-Small-24B-Instruct-2501",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
    )

    # Deep Seek API
    # --------------------------------------------------------------
    models["deepseek-chat"] = ChatCompletionSampler(
        model="deepseek-chat",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        get_logprobs=True
    )

    models["deepseek-reasoner"] = ChatCompletionSampler(
        model="deepseek-reasoner",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        get_logprobs=True
    )
    # --------------------------------------------------------------

    # Claude
    # --------------------------------------------------------------
    # Claude 3.7 Sonnet
    models["claude-3-7-sonnet-20250219"] = ChatCompletionSampler(
        model="claude-3-7-sonnet-20250219",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url="https://api.anthropic.com/v1/",
        get_logprobs=False # Not supported
    )
    # Claude 3.5 Haiku
    models["claude-3-5-haiku-20241022"] = ChatCompletionSampler(
        model="claude-3-5-haiku-20241022",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url="https://api.anthropic.com/v1/",
        get_logprobs=False # Not supported
    )
    # Claude 3 Haiku
    models["claude-3-haiku-20240307"] = ChatCompletionSampler(
        model="claude-3-haiku-20240307",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url="https://api.anthropic.com/v1/",
        get_logprobs=False # Not supported
    )
    # --------------------------------------------------------------


    # Grok
    # --------------------------------------------------------------
    models["grok-3-beta"] = ChatCompletionSampler(
        model="grok-3-beta",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1",
        get_logprobs=True
    )

    models["grok-3-mini-beta"] = ChatCompletionSampler(
        model="grok-3-mini-beta",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1",
        get_logprobs=True
    )

    models["grok-2-1212"] = ChatCompletionSampler(
        model="grok-2-1212",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1",
        get_logprobs=True
    )
    # --------------------------------------------------------------


    # Google AI Studio
    # --------------------------------------------------------------
    google_models = [
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-pro-preview-05-06",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
    ]

    for gm in google_models:
        models[gm] = GoogleAISampler(
                model=gm,
                api_key=os.environ["GOOGLE_AI_KEY"],
                get_logprobs=True
        )
    # --------------------------------------------------------------

    return models