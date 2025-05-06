import os
from ..sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from ..sampler.o_chat_completion_sampler import OChatCompletionSampler
from ..sampler.claude_sampler import ClaudeCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS
from ..sampler.hfmodel_sampler import HFChatCompletionSampler

def get_model_dict(model_name: str):
    """
    Get the model dict from the model name.
    """

    hf_models = [
        "meta-llama/Llama-3.2-3B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "meta-llama/Llama-3.1-405B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct"
        "meta-llama/Llama-3.2-1B-Instruct",
        "deepseek-ai/DeepSeek-R1",
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
        "google/gemma-3-27b-it"
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
                model=model_name,
                API_TOKEN=os.environ.get("HF_TOKEN", None),
                system_message=OPENAI_SYSTEM_MESSAGE_API,
                max_new_tokens=2048,
                temperature=1e-6,
                use_vllm=True
            )


    # ----------- Databrick LLMs -----------
    DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")

    models["databricks-llama-4-maverick"] = ChatCompletionSampler(
            api_key=DATABRICKS_TOKEN,
            base_url="https://dbc-4325dfb0-cd6d.cloud.databricks.com/serving-endpoints",
            model="databricks-llama-4-maverick",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
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


    models["Qwen/Qwen3-235B-A22B-fp8"] = ChatCompletionSampler(
            base_url = "https://api.together.xyz/v1",
            api_key = os.environ['TOGETHER_API_KEY'],
            model="Qwen/Qwen3-235B-A22B-fp8",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            get_logprobs=True
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

    # ----------- Google LLMs -----------
    # from google.auth import default
    # from google.auth.transport.requests import Request
    # credentials, _ = default()
    # auth_request = Request()
    # credentials.refresh(auth_request)
    # base_url = "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/storied-channel-368910/locations/us-central1/endpoints/openapi"
    
    # models["google-llama-3.1-405b-instruct-maas"] = ChatCompletionSampler(
    #     model="meta/llama-3.1-405b-instruct-maas",
    #     system_message=OPENAI_SYSTEM_MESSAGE_API,
    #     max_tokens=2048,
    #     base_url=base_url,
    #     api_key=credentials.token
    # )
    # models["google-llama-3.1-70b-instruct-maas"] = ChatCompletionSampler(
    #         model="meta/llama-3.1-70b-instruct-maas",
    #         system_message=OPENAI_SYSTEM_MESSAGE_API,
    #         max_tokens=2048,
    #         base_url=base_url,
    #         api_key=credentials.token
    #     )
    # models["google-llama-3.1-8b-instruct-maas"] = ChatCompletionSampler(
    #         model="meta/llama-3.1-8b-instruct-maas",
    #         system_message=OPENAI_SYSTEM_MESSAGE_API,
    #         max_tokens=2048,
    #         base_url=f"https://us-central1-aiplatform.googleapis.com/v1beta1/projects/storied-channel-368910/locations/us-central1/endpoints/openapi",
    #         api_key=credentials.token
    #     )
    # models["google-llama-3.3-70b-instruct-maas"] = ChatCompletionSampler(
    #         model="meta/llama-3.3-70b-instruct-maas",
    #         system_message=OPENAI_SYSTEM_MESSAGE_API,
    #         max_tokens=2048,
    #         base_url=f"https://us-central1-aiplatform.googleapis.com/v1beta1/projects/storied-channel-368910/locations/us-central1/endpoints/openapi",
    #         api_key=credentials.token
    #     )
    # models["google-llama-3.1-8b-instruct-maas"] = ChatCompletionSampler(
    #         model="meta/llama-3.1-8b-instruct-maas",
    #         system_message=OPENAI_SYSTEM_MESSAGE_API,
    #         max_tokens=2048,
    #         base_url=f"https://us-central1-aiplatform.googleapis.com/v1beta1/projects/storied-channel-368910/locations/us-central1/endpoints/openapi",
    #         api_key=credentials.token
    #     )
    
    # google_models = [
    #     "gemini-2.5-flash-preview-04-17",
    #     "gemini-2.5-pro-exp-03-25",
    #     "gemini-2.0-flash",
    #     "gemini-2.0-flash-lite",
    #     "gemini-1.5-flash",
    #     "gemini-1.5-flash-8b",
    #     "gemini-1.5-pro",
    # ]

    # for gm in google_models:
    #     models[gm] = GoogleAISampler(
    #             model=gm,
    #             api_key=os.environ["GOOGLE_AI_KEY"],
    #             get_logprobs=True
    #     )
    

    # # free during preview stage
    # models["meta/llama-4-maverick-17b-128e-instruct-maas"] = ChatCompletionSampler(
    #         model="meta/llama-4-maverick-17b-128e-instruct-maas",
    #         system_message=OPENAI_SYSTEM_MESSAGE_API,
    #         max_tokens=2048,
    #         base_url=f"https://{"us-east5-aiplatform.googleapis.com"}/v1beta1/projects/storied-channel-368910/locations/us-central1/endpoints/openapi",
    #         api_key=credentials.token
    #     )
    # # free during preview stage
    # models["meta/llama-4-scout-17b-16e-instruct-maas"] = ChatCompletionSampler(
    #         model="meta/llama-4-scout-17b-16e-instruct-maas",
    #         system_message=OPENAI_SYSTEM_MESSAGE_API,
    #         max_tokens=2048,
    #         base_url=f"https://{"us-east5-aiplatform.googleapis.com"}/v1beta1/projects/storied-channel-368910/locations/us-central1/endpoints/openapi",
    #         api_key=credentials.token
    #     )
    
    # # free during preview stage
    # models["meta/llama-3.2-90b-vision-instruct-maas"] = ChatCompletionSampler(
    #         model="meta/llama-3.2-90b-vision-instruct-maas",
    #         system_message=OPENAI_SYSTEM_MESSAGE_API,
    #         max_tokens=2048,
    #         base_url=f"https://{"us-central1-aiplatform.googleapis.com"}/v1beta1/projects/storied-channel-368910/locations/us-central1/endpoints/openapi",
    #         api_key=credentials.token
    #     )
    
    # # free during preview stage
    # models["meta/llama-3.2-1b-instruct"] = ChatCompletionSampler(
    #         model="meta/llama-3.2-1b-instruct",
    #         system_message=OPENAI_SYSTEM_MESSAGE_API,
    #         max_tokens=2048,
    #         base_url=f"https://{"us-central1-aiplatform.googleapis.com"}/v1beta1/projects/storied-channel-368910/locations/us-central1/endpoints/openapi",
    #         api_key=credentials.token
    #     )
    
    # models["mistral/mistral-large-2402"] = ChatCompletionSampler(
    #         model="mistral/mistral-large-2402",
    #         system_message=OPENAI_SYSTEM_MESSAGE_API,
    #         max_tokens=2048,
    #         base_url=f"https://{"us-central1-aiplatform.googleapis.com"}/v1beta1/projects/storied-channel-368910/locations/us-central1/endpoints/openapi",
    #         api_key=credentials.token
    #     )



    # # Remote VLLM
    # # --------------------------------------------------------------
    # # Lab 0
    # # --------------------------------------------------------------
    # models["Qwen/Qwen3-0.6B-Base"] = ChatCompletionSampler(
    #     model="Qwen/Qwen3-0.6B-Base",
    #     system_message=OPENAI_SYSTEM_MESSAGE_API,
    #     max_tokens=2048,
    #     api_key="ANY",
    #     base_url="http://localhost:7000/v1",
    #     get_logprobs=True
    # )

    # models["Qwen/Qwen3-0.6B-FP8"] = ChatCompletionSampler(
    #     model="Qwen/Qwen3-0.6B-FP8",
    #     system_message=OPENAI_SYSTEM_MESSAGE_API,
    #     max_tokens=2048,
    #     api_key="ANY",
    #     base_url="http://localhost:7001/v1",
    #     get_logprobs=True
    # )

    # models["Qwen/Qwen3-0.6B"] = ChatCompletionSampler(
    #     model="Qwen/Qwen3-0.6B",
    #     system_message=OPENAI_SYSTEM_MESSAGE_API,
    #     max_tokens=2048,
    #     api_key="ANY",
    #     base_url="http://localhost:7002/v1",
    #     get_logprobs=True
    # )
    # # --------------------------------------------------------------

    # # Lab 1
    # # --------------------------------------------------------------
    # models["Qwen/Qwen3-4B"] = ChatCompletionSampler(
    #     model="Qwen/Qwen3-4B",
    #     system_message=OPENAI_SYSTEM_MESSAGE_API,
    #     max_tokens=2048,
    #     api_key="ANY",
    #     base_url="http://localhost:5001/v1",
    #     get_logprobs=True
    # )

    # models["Qwen/Qwen3-4B-Base"] = ChatCompletionSampler(
    #     model="Qwen/Qwen3-4B-Base",
    #     system_message=OPENAI_SYSTEM_MESSAGE_API,
    #     max_tokens=2048,
    #     api_key="ANY",
    #     base_url="http://localhost:5002/v1",
    #     get_logprobs=True
    # )

    # models["Qwen/Qwen3-14B"] = ChatCompletionSampler(
    #     model="Qwen/Qwen3-14B",
    #     system_message=OPENAI_SYSTEM_MESSAGE_API,
    #     max_tokens=2048,
    #     api_key="ANY",
    #     base_url="http://localhost:5005/v1",
    #     get_logprobs=True
    # )

    # models["Qwen/Qwen3-1.7B-Base"] = ChatCompletionSampler(
    #     model="Qwen/Qwen3-1.7B-Base",
    #     system_message=OPENAI_SYSTEM_MESSAGE_API,
    #     max_tokens=2048,
    #     api_key="ANY",
    #     base_url="http://localhost:5000/v1",
    #     get_logprobs=True
    # )
    # # --------------------------------------------------------------

    # # Lab 2
    # # --------------------------------------------------------------
    # models["Qwen/Qwen3-1.7B"] = ChatCompletionSampler(
    #     model="Qwen/Qwen3-1.7B",
    #     system_message=OPENAI_SYSTEM_MESSAGE_API,
    #     max_tokens=2048,
    #     api_key="ANY",
    #     base_url="http://localhost:7101/v1",
    #     get_logprobs=True
    # )

    # models["Qwen/Qwen3-1.7B-FP8"] = ChatCompletionSampler(
    #     model="Qwen/Qwen3-1.7B-FP8",
    #     system_message=OPENAI_SYSTEM_MESSAGE_API,
    #     max_tokens=2048,
    #     api_key="ANY",
    #     base_url="http://localhost:7100/v1",
    #     get_logprobs=True
    # )

    # models["Qwen/Qwen3-8B-Base"] = ChatCompletionSampler(
    #     model="Qwen/Qwen3-8B-Base",
    #     system_message=OPENAI_SYSTEM_MESSAGE_API,
    #     max_tokens=2048,
    #     api_key="ANY",
    #     base_url="http://localhost:7105/v1",
    #     get_logprobs=True
    # )

    # models["Qwen/Qwen3-8B"] = ChatCompletionSampler(
    #     model="Qwen/Qwen3-8B",
    #     system_message=OPENAI_SYSTEM_MESSAGE_API,
    #     max_tokens=2048,
    #     api_key="ANY",
    #     base_url="http://localhost:7104/v1",
    #     get_logprobs=True
    # )
    # --------------------------------------------------------------
    return models