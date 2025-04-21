import os
from ..sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from ..sampler.o_chat_completion_sampler import OChatCompletionSampler
from ..sampler.claude_sampler import ClaudeCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS
from ..sampler.hfmodel_sampler import HFChatCompletionSampler

from google.auth import default
from google.auth.transport.requests import Request

def get_model_dict(model_name: str):
    """
    Get the model dict from the model name.
    """
    models = {}
    if model_name == "gpt-4o-2024-11-20_assistant":
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
    elif model_name == "Llama-3.2-3B-Instruct":
        models["Llama-3.2-3B-Instruct"] = HFChatCompletionSampler(
                model="meta-llama/Llama-3.2-3B-Instruct",
                API_TOKEN=os.environ.get("HF_TOKEN", None),
                system_message=None,
                max_tokens=2048,
                temperature=0.7,
            )
    elif model_name == "Qwen2.5-3B-Instruct":
        models["Qwen2.5-3B-Instruct"] = HFChatCompletionSampler(
                model="Qwen/Qwen2.5-3B-Instruct",
                API_TOKEN=os.environ.get("HF_TOKEN", None),
                system_message=None,
                max_tokens=2048,
                temperature=0.7,
            )
    elif model_name == "Qwen2.5-7B-Instruct":
        models["Qwen2.5-7B-Instruct"] = HFChatCompletionSampler(
                model="Qwen/Qwen2.5-7B-Instruct",
                API_TOKEN=os.environ.get("HF_TOKEN", None),
                system_message=None,
                max_tokens=2048,
                temperature=0.7,
            )
    elif model_name == "Qwen2.5-13B-Instruct":
        models["Qwen2.5-13B-Instruct"] = HFChatCompletionSampler(
                model="Qwen/Qwen2.5-32B-Instruct",
                API_TOKEN=os.environ.get("HF_TOKEN", None),
                system_message=None,
                max_tokens=2048,
                temperature=0.7,
            )
    elif model_name == "Qwen2.5-32B-Instruct":
        models["Qwen2.5-32B-Instruct"] = HFChatCompletionSampler(
                model="Qwen/Qwen2.5-72B-Instruct",
                API_TOKEN=os.environ.get("HF_TOKEN", None),
                system_message=None,
                max_tokens=2048,
                temperature=0.7,
            )
    elif model_name == "Llama-3.1-405B-Instruct":
        models["Llama-3.1-405B-Instruct"] = HFChatCompletionSampler(
                model="meta-llama/Llama-3.1-405B-Instruct",
                API_TOKEN=os.environ.get("HF_TOKEN", None),
                system_message=None,
                max_tokens=2048,
                temperature=0.7,
            )
    elif model_name == "Llama-3.3-70B-Instruct": 
        models["Llama-3.3-70B-Instruct"] = HFChatCompletionSampler(
                model="meta-llama/Llama-3.3-70B-Instruct",
                API_TOKEN=os.environ.get("HF_TOKEN", None),
                system_message=None,
                max_tokens=2048,
                temperature=0.7,
            )
    elif model_name == "Llama-3.1-8B-Instruct": 
        models["Llama-3.1-8B-Instruct"] = HFChatCompletionSampler(
                model="meta-llama/Llama-3.1-8B-Instruct",
                API_TOKEN=os.environ.get("HF_TOKEN", None),
                system_message=None,
                max_tokens=2048,
                temperature=0.7,
            )
    elif model_name == "DeepSeek-R1":
        models["DeepSeek-R1"] = HFChatCompletionSampler(
                model="deepseek-ai/DeepSeek-R1",
                API_TOKEN=os.environ.get("HF_TOKEN", None),
                system_message=None,
                max_tokens=2048,
                temperature=0.7,
            )
        
    # ----------- Google LLMs -----------
    credentials, _ = default()
    auth_request = Request()
    credentials.refresh(auth_request)
    base_url = "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/storied-channel-368910/locations/us-central1/endpoints/openapi"
    
    models["google-llama-3.1-405b-instruct-maas"] = ChatCompletionSampler(
        model="meta/llama-3.1-405b-instruct-maas",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
        base_url=base_url,
        api_key=credentials.token
    )
    models["google-llama-3.1-70b-instruct-maas"] = ChatCompletionSampler(
            model="meta/llama-3.1-70b-instruct-maas",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            base_url=base_url,
            api_key=credentials.token
        )
    models["google-llama-3.1-8b-instruct-maas"] = ChatCompletionSampler(
            model="meta/llama-3.1-8b-instruct-maas",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            base_url=f"https://us-central1-aiplatform.googleapis.com/v1beta1/projects/storied-channel-368910/locations/us-central1/endpoints/openapi",
            api_key=credentials.token
        )
    models["google-llama-3.3-70b-instruct-maas"] = ChatCompletionSampler(
            model="meta/llama-3.3-70b-instruct-maas",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            base_url=f"https://us-central1-aiplatform.googleapis.com/v1beta1/projects/storied-channel-368910/locations/us-central1/endpoints/openapi",
            api_key=credentials.token
        )
    models["google-llama-3.1-8b-instruct-maas"] = ChatCompletionSampler(
            model="meta/llama-3.1-8b-instruct-maas",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            base_url=f"https://us-central1-aiplatform.googleapis.com/v1beta1/projects/storied-channel-368910/locations/us-central1/endpoints/openapi",
            api_key=credentials.token
        )
    

    # free during preview stage
    models["meta/llama-4-maverick-17b-128e-instruct-maas"] = ChatCompletionSampler(
            model="meta/llama-4-maverick-17b-128e-instruct-maas",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            base_url=f"https://{"us-east5-aiplatform.googleapis.com"}/v1beta1/projects/storied-channel-368910/locations/us-central1/endpoints/openapi",
            api_key=credentials.token
        )
    # free during preview stage
    models["meta/llama-4-scout-17b-16e-instruct-maas"] = ChatCompletionSampler(
            model="meta/llama-4-scout-17b-16e-instruct-maas",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            base_url=f"https://{"us-east5-aiplatform.googleapis.com"}/v1beta1/projects/storied-channel-368910/locations/us-central1/endpoints/openapi",
            api_key=credentials.token
        )
    
    # free during preview stage
    models["meta/llama-3.2-90b-vision-instruct-maas"] = ChatCompletionSampler(
            model="meta/llama-3.2-90b-vision-instruct-maas",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            base_url=f"https://{"us-central1-aiplatform.googleapis.com"}/v1beta1/projects/storied-channel-368910/locations/us-central1/endpoints/openapi",
            api_key=credentials.token
        )
    
    # free during preview stage
    models["meta/llama-3.2-90b-vision-instruct-maas"] = ChatCompletionSampler(
            model="meta/llama-3.2-90b-vision-instruct-maas",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            base_url=f"https://{"us-central1-aiplatform.googleapis.com"}/v1beta1/projects/storied-channel-368910/locations/us-central1/endpoints/openapi",
            api_key=credentials.token
        )
    





    # ----------- Databrick LLMs -----------
    DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")

    models["databricks-llama-4-maverick"] = ChatCompletionSampler(
            api_key=DATABRICKS_TOKEN,
            base_url="https://dbc-1b14d1ec-f076.cloud.databricks.com/serving-endpoints",
            model="databricks-llama-4-maverick",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
    )


    # ----------- Together LLMs -----------
    from together import Together

    models["Llama-4-Scout-17B-16E-Instruct"] = ChatCompletionSampler(
            base_url = "https://api.together.xyz/v1",
            api_key = os.environ['TOGETHER_API_KEY'],
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
    )

    models["meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"] = ChatCompletionSampler(
            base_url = "https://api.together.xyz/v1",
            api_key = os.environ['TOGETHER_API_KEY'],
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
    )

    # free
    models["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"] = ChatCompletionSampler(
            base_url = "https://api.together.xyz/v1",
            api_key = os.environ['TOGETHER_API_KEY'],
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
    )

    # free
    models["deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"] = ChatCompletionSampler(
            base_url = "https://api.together.xyz/v1",
            api_key = os.environ['TOGETHER_API_KEY'],
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
    )
    
        
    return models