"""
MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark
Paper Reference: https://arxiv.org/abs/2406.01574
Dataset and code: https://github.com/TIGER-AI-Lab/MMLU-Pro?tab=readme-ov-file
"""

import random
import pickle
import os
import sys
import time

from huggingface_hub import hf_hub_download
from datasets import load_dataset
from filelock import FileLock
from . import common
from .common import *
from .custom_types import Eval, EvalResult, SamplerBase, SingleEvalResult
from .utils.pre_processing import *
from .utils.post_processing_report import *
from .utils.post_processing_answer import *
from .utils.post_processing_confidence import *
import threading
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import DataParallel


pickle_write_lock = threading.Lock()
approx_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-4-Maverick-17B-128E-Instruct", trust_remote_code=True)

# increased token threshold for linguistic confidence judging
special_token_allowance = [
    "Qwen/Qwen3-235B-A22B-fp8-tput-think",
    "Qwen/Qwen3-32B-think",
    "Qwen/Qwen3-4B-think",
    "Qwen/Qwen3-8B-think",
    "Qwen/Qwen3-30B-A3B-think",
    "Qwen/Qwen3-14B-think",
]

def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    examples = []
    [examples.extend(lst) for lst in res.values()]
    return examples

class MMLUProEval(Eval):
    def __init__(self, decisiveness_grader: SamplerBase, num_examples: int | None = None, conf_mode: str = "verbal_vanilla", regenerate=True):
        self.examples = preprocess(load_dataset("TIGER-Lab/MMLU-Pro")["test"])
        if num_examples:
            self.examples = random.Random(0).sample(self.examples, num_examples)
        self.conf_mode = conf_mode
        self.regenerate = regenerate
        self.num_examples = num_examples
        self.cache_found = False
        self.decisiveness_grader = decisiveness_grader

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):

            extracted_answer = None
            semantic_entropy_answer = None
            confidence = 0
            judge_response = ""
            logprobs = None 
            verbal_numerical_confidence = None
            verbal_linguistic_confidence = None
            logit_perplexity_confidence = None
            semantic_entropy = None
            token_length = None

            match self.conf_mode:
                
                case "verbal_numerical" | "verbal_numerical_shared_sampling" | "tmp_verbal_numerical_shared_sampling":
                    if self.cache_found:
                        response = row["response"] 
                        prompt_messages = row["prompt_messages"] 
                    else:
                        prompt_messages = [
                            sampler._pack_message(
                                content=format_multichoice_question(row, conf_mode="verbal_numerical", choices=0), role="user"
                            )
                        ]
                        response = sampler(prompt_messages)
                        row["prompt_messages"] = prompt_messages
                        row["response"] = response
                        row["logprobs"] = sampler.logprobs
                        row["top_logprobs"] = sampler.top_logprobs
                    logprobs = row["logprobs"]
                    response_text = normalize_response(response)
                    # extracted_answer, confidence = consolidated_answer_extraction(benchmark="mmlu_pro", response_text=response_text, row=row, with_verbal_confidence=True)
                    extracted_answer = extract_answer(response_text=response)
                    confidence = extract_verbal_numerical_confidence(response_text=response_text)

                
                case "verbal_linguistic" | "verbal_linguistic_shared_sampling" | "tmp_verbal_linguistic_shared_sampling":
                    if self.cache_found:
                        response = row["response"] 
                        prompt_messages = row["prompt_messages"] 
                    else:
                        prompt_messages = [
                            sampler._pack_message(
                                content=format_multichoice_question(row, conf_mode="verbal_linguistic", choices=0), role="user"
                            )
                        ]
                        response = sampler(prompt_messages)
                        row["prompt_messages"] = prompt_messages
                        row["response"] = response
                        row["logprobs"] = sampler.logprobs
                        row["top_logprobs"] = sampler.top_logprobs
                    logprobs = row["logprobs"]
                    response_text = remove_verbal_confidence(normalize_response(response)) # remove verbal confidence to avoid judgement biases
                    # extracted_answer = consolidated_answer_extraction(benchmark="mmlu_pro", response_text=response_text, row=row, with_verbal_confidence=False)
                    extracted_answer = extract_answer(response_text=response)
                    confidence, judge_response = linguistic_confidence_score(self.decisiveness_grader, format_multichoice_question(row, conf_mode="decisiveness_grading", choices=0), response_text)


                case "logit_perplexity" | "logit_perplexity_shared_sampling" | "tmp_logit_perplexity_shared_sampling":
                    if sampler.get_logprobs == False:
                        raise NotImplementedError("The selected model does not support logprobs. Force switching on by setting the model's get_logprobs to True in utils/models.py only after checking with the provider.")
                    if self.cache_found:
                        response = row["response"] 
                        prompt_messages = row["prompt_messages"] 
                    else:
                        prompt_messages = [
                            sampler._pack_message(
                                content=format_multichoice_question(row, conf_mode="logit_perplexity", choices=0), role="user"
                            )
                        ]
                        response = sampler(prompt_messages)
                        row["prompt_messages"] = prompt_messages
                        row["response"] = response
                        row["logprobs"] = sampler.logprobs
                        row["top_logprobs"] = sampler.top_logprobs

                    response_text = normalize_response(response)
                    # extracted_answer = consolidated_answer_extraction(benchmark="mmlu_pro", response_text=response_text, row=row, with_verbal_confidence=False)
                    extracted_answer = extract_answer(response_text=response)
                    logprobs = row["logprobs"]
                    confidence = calculate_logit_perplexity(logprobs)


                case "token_sar" | "token_sar_shared_sampling":
                    if sampler.get_logprobs == False:
                        raise NotImplementedError("The selected model does not support logprobs. Force switching on by setting the model's get_logprobs to True in utils/models.py only after checking with the provider.")
                    if self.cache_found:
                        response = row["response"] 
                        top_logprobs = row["top_logprobs"] 
                        prompt_messages = row["prompt_messages"] 
                    else:
                        prompt_messages = [
                            sampler._pack_message(
                                content=format_multichoice_question(row, conf_mode="token_sar", choices=0), role="user"
                            )
                        ]
                        response = sampler(prompt_messages)
                        row["prompt_messages"] = prompt_messages
                        row["response"] = response
                        row["logprobs"] = sampler.logprobs
                        row["top_logprobs"] = sampler.top_logprobs

                    response_text = normalize_response(response)
                    extracted_answer = extract_answer(response_text=response)
                    logprobs = row["logprobs"]
                    confidence = token_sar_confidence(top_logprobs)


                case "eval_all":
                    if not self.cache_found:
                        print("Please sample responses before evaluating.")
                        sys.exit()
                    response = row["response"] 
                    prompt_messages = row["prompt_messages"] 
                    logprobs = row["logprobs"]
                    top_logprobs = row["top_logprobs"]

                    # extract answer 
                    response_text = normalize_response(response)
                    extracted_answer = extract_answer(response_text=response_text)
                    verbal_numerical_confidence = extract_verbal_numerical_confidence(response_text=response_text)
                    if logprobs is not None:
                        logit_perplexity_confidence = calculate_logit_perplexity(logprobs)
                    else:
                        logit_perplexity_confidence = None 

                    # only evaluate a response when it has extracted answer and confidence and its token lenght < 10240
                    tokens = approx_tokenizer(response_text, return_tensors="pt")
                    token_length = len(tokens["input_ids"][0])
                    token_cap = 10240 if sampler.model in special_token_allowance else 1024
                    # if response is not None and verbal_numerical_confidence is not None and token_length < token_cap:
                    #     verbal_linguistic_confidence, judge_response = linguistic_confidence_score(self.decisiveness_grader, format_multichoice_question(row, conf_mode="decisiveness_grading", choices=0), remove_verbal_confidence(response_text))
                    #     print(verbal_linguistic_confidence)
                    # else:
                    #     verbal_linguistic_confidence, judge_response = None, None

                    confidence = verbal_numerical_confidence


                case "batch_eval_all":
                    if not self.cache_found:
                        print("Please sample responses before evaluating.")
                        sys.exit()
                    response = row["response"] 
                    prompt_messages = row["prompt_messages"] 
                    logprobs = row["logprobs"]
                    top_logprobs = row["top_logprobs"]

                    # extract answer 
                    response_text = normalize_response(response)
                    extracted_answer = extract_answer(response_text=response_text)
                    verbal_numerical_confidence = extract_verbal_numerical_confidence(response_text=response_text)
                    if logprobs is not None:
                        logit_perplexity_confidence = calculate_logit_perplexity(logprobs)
                    else:
                        logit_perplexity_confidence = None 
                    verbal_linguistic_confidence = row["linguistic_confidence"]
                    
                    confidence = verbal_numerical_confidence
                    judge_response = row["linguistic_judge_response"]


                case "sampling":
                    if sampler.get_logprobs == False:
                        print("Sampling without logprobs.")
                    else:
                        print("Sampling with logprobs.")
                    prompt_messages = [
                        sampler._pack_message(
                            content=format_multichoice_question(row, conf_mode="sampling", choices=0), role="user"
                        )
                    ]
                    response = sampler(prompt_messages)
                    row["prompt_messages"] = prompt_messages
                    row["response"] = response
                    row["logprobs"] = sampler.logprobs
                    row["top_logprobs"] = sampler.top_logprobs
                    if sampler.logprobs is None:
                        print("Token log probs not available.")
                    if sampler.top_logprobs is None:
                        print("Top log probs not available.")


                    # save current progress
                    current_progress = self.examples.copy()
                    current_progress = [eg for eg in self.examples if "logprobs" in eg.keys()]

                    if self.conf_mode in ["sampling", "eval_all", "semantic_entropy"] or "_shared_sampling" in self.conf_mode:
                        self.regenerate = True
                        model_name = sampler.model + ("-think" if hasattr(sampler, "think") and sampler.think else "")
                        progress_temp_path = shared_sampling_path(f"tmp_mmlu_pro", model_name, self.conf_mode, self.num_examples, None)
                    else:
                        progress_temp_path = ind_sampling_path(f"tmp_mmlu_pro", model_name, self.conf_mode, self.num_examples, None)
                        
                    with pickle_write_lock:
                        with open(progress_temp_path, 'wb') as f:
                            pickle.dump(current_progress, f)
                        print("Progress saved")
                    return 
                

                case "semantic_entropy":
                    if not self.cache_found:
                        print("Please sample responses before evaluating.")
                        sys.exit()
                    response = row["response"] 
                    prompt_messages = row["prompt_messages"] 
                    logprobs = row["logprobs"]
                    top_logprobs = row["top_logprobs"]
                    multi_responses = row["rep_responses"]
                    response_text = normalize_response(response)
                    semantic_entropy_answer, semantic_entropy = mcq_semantic_entropy(multi_responses)

                case _:
                    raise Exception(f"Unrecognized confidence type: {self.conf_mode}")

            # print(f"extracted_answer: {extracted_answer}, confidence: {confidence}")
            score = 1.0 if extracted_answer == row["answer"] else 0.0

            category = row["category"]
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["answer"],
                extracted_answer=extracted_answer,
                extracted_answer_confidence="Verbal Numerical=" + str(verbal_numerical_confidence) + ", Verbal Linguistic=" + str(verbal_linguistic_confidence) + ", Logit Perplexity=" + str(logit_perplexity_confidence) + ", Semantic Entropy=" + str(semantic_entropy),
                subject=category,
                logprobs = logprobs,
                conf_mode = self.conf_mode,
                linguistic_judge_response = judge_response
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html, score=score, metrics={category: score}, convo=convo, confidence=(confidence), 
                category=category, correct_answer=row["answer"], extracted_answer=extracted_answer,
                verbal_numerical_confidence=verbal_numerical_confidence, 
                logit_perplexity_confidence=logit_perplexity_confidence, 
                verbal_linguistic_confidence=verbal_linguistic_confidence,
                semantic_entropy_answer=semantic_entropy_answer,
                semantic_entropy=semantic_entropy,
                token_length=token_length
            )
        
        # ---------------------------------- Local transformers for sampling -----------------------------------
        if sampler.base_url == "" and self.conf_mode == "sampling-hf":
            self.conf_mode = "sampling"
            enable_thinking = hasattr(sampler, "think") and sampler.think
            tokenizer = sampler.tokenizer
            hf_model = sampler.get_hf_model()
            hf_model.config.use_flash_attention = False
            tokenizer.pad_token = tokenizer.eos_token
            inference_batch = []
            batch_size = 1
            indexed_inference_batch = []
            for i in tqdm(range(len(self.examples)), desc="Prepare prompt batch"):
                prompt = [
                    sampler._pack_message("system", sampler.system_message),
                    sampler._pack_message(
                        content=format_multichoice_question(self.examples[i], conf_mode="sampling", choices=0),
                        role="user"
                    )
                ]
                prompt_str = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking
                )
                indexed_inference_batch.append((i, prompt_str))
            for start_idx in tqdm(range(0, len(indexed_inference_batch), batch_size), desc="Inference batches"):
                end_idx = start_idx + batch_size
                batch = indexed_inference_batch[start_idx:end_idx]
                batch_indices, batch_prompts = zip(*batch)
                batch_examples = [self.examples[i] for i in batch_indices]

                tokenized_inputs = tokenizer(
                    list(batch_prompts),
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                tokenized_inputs = {k: v.to(hf_model.device) for k, v in tokenized_inputs.items()}

                input_ids = tokenized_inputs["input_ids"]
                prompt_lengths = (input_ids != tokenizer.pad_token_id).sum(dim=1)

                with torch.no_grad():
                    outputs = hf_model.generate(
                        **tokenized_inputs,
                        max_new_tokens=2048,
                        do_sample=True,
                        return_dict_in_generate=True,
                        output_logits=True,
                        output_scores=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                torch.cuda.empty_cache()

                generated_sequences = [
                    seq[prompt_len:] for seq, prompt_len in zip(outputs.sequences, prompt_lengths)
                ]
                decoded_outputs = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)

                for idx, response in zip(batch_indices, decoded_outputs):
                    example = self.examples[idx]
                    if "gemma-3" in sampler.model.lower():
                        generated_text = "Explanation:\n" + response.split("model\nExplanation:")[-1].strip()
                    else:
                        generated_text = response.split("Assistant: ")[-1].strip()
                    example["prompt_messages"] = [
                        sampler._pack_message(
                            content=format_multichoice_question(example, conf_mode="sampling", choices=0),
                            role="user"
                        )
                    ]
                    example["response"] = generated_text
                    print(f"[{idx}] {generated_text}")

                try:
                    log_probs_lst = hf_model.compute_transition_scores(
                        outputs.sequences,
                        outputs.scores,
                        normalize_logits=True
                    ).cpu().numpy(force=True)

                    for i, logprobs in enumerate(log_probs_lst):
                        self.examples[batch_indices[i]]["logprobs"] = logprobs
                        self.examples[batch_indices[i]]["top_logprobs"] = None
                except Exception as e:
                    for i in batch_indices:
                        self.examples[i]["logprobs"] = None
                        self.examples[i]["top_logprobs"] = None


            model_name = sampler.model + ("-think" if hasattr(sampler, "think") and sampler.think else "")
            regen_stored_path = shared_sampling_path("mmlu_pro", model_name, self.conf_mode, self.num_examples, None)
            with open(regen_stored_path, 'wb') as f:
                pickle.dump(self.examples, f)
                print(f"Shared sampling with vLLM completed and saved to: {regen_stored_path}")
            sys.exit()
        # ----------------------------------------------------------------------------------------------
        
        # ---------------------------------- Local vLLM for sampling -----------------------------------
        if sampler.base_url == "" and self.conf_mode == "sampling":
            from vllm import LLM, SamplingParams
            print("vllm")
            # set up vLLM mode 
            enable_thinking = hasattr(sampler, "think") and sampler.think
            is_qwen = "qwen" in sampler.model.lower()
            tokenizer = sampler.tokenizer
            visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if visible_gpus:
                num_gpus = len(visible_gpus.split(','))
            else:
                num_gpus = torch.cuda.device_count()  # fallback

            if "gguf" in sampler.model.lower():
                repo_id, filename = sampler.model.split("@")[0], sampler.model.split("@")[1]
                path = hf_hub_download(repo_id=repo_id, filename=filename)
                llm = LLM(
                    model=path,
                    max_model_len=None,
                    trust_remote_code=True,
                    tokenizer=repo_id.replace("-GGUF", ""),
                    tensor_parallel_size=num_gpus
                )
            else:
                llm = LLM(
                    model=sampler.model,
                    max_model_len=None,
                    trust_remote_code=True,
                    tokenizer_mode="auto",
                    tensor_parallel_size=num_gpus
                )
            sampling_params = SamplingParams(temperature=1, max_tokens=(10240 if enable_thinking else 2048), logprobs=5, seed=42, stop=[tokenizer.eos_token])
            batch_replication = 10
            # prepare batch
            if is_qwen:
                try:
                    inference_batch = []
                    for i in tqdm(range(len(self.examples)), desc="Prepare prompt batch"):
                        prompt = [sampler._pack_message("system", sampler.system_message), 
                                sampler._pack_message(content=format_multichoice_question(self.examples[i], conf_mode="sampling", choices=0), role="user")]
                        inference_batch.append(tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking))
                    inference_batch = inference_batch * batch_replication
                    outputs = llm.generate(inference_batch, sampling_params, use_tqdm=True)
                except jinja2.exceptions.TemplateError:
                    inference_batch = []
                    for i in tqdm(range(len(self.examples)), desc="Prepare prompt batch"):
                        prompt = [sampler._pack_message(content=format_multichoice_question(self.examples[i], conf_mode="sampling", choices=0), role="user")]
                        inference_batch.append(tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking))
                    inference_batch = inference_batch * batch_replication
                    outputs = llm.generate(inference_batch, sampling_params, use_tqdm=True)
            else:
                try:
                    inference_batch = []
                    for i in tqdm(range(len(self.examples)), desc="Prepare prompt batch"):
                        prompt = [sampler._pack_message("system", sampler.system_message), 
                                sampler._pack_message(content=format_multichoice_question(self.examples[i], conf_mode="sampling", choices=0), role="user")]
                        inference_batch.append(tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True))
                    inference_batch = inference_batch * batch_replication
                    outputs = llm.generate(inference_batch, sampling_params, use_tqdm=True)
                except jinja2.exceptions.TemplateError:
                    inference_batch = []
                    for i in tqdm(range(len(self.examples)), desc="Prepare prompt batch"):
                        prompt = [sampler._pack_message(content=format_multichoice_question(self.examples[i], conf_mode="sampling", choices=0), role="user")]
                        inference_batch.append(tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True))
                    inference_batch = inference_batch * batch_replication
                    outputs = llm.generate(inference_batch, sampling_params, use_tqdm=True)
                except:
                    inference_batch = []
                    for i in tqdm(range(len(self.examples)), desc="Prepare prompt batch"):
                        prompt = format_multichoice_question(self.examples[i], conf_mode="sampling", choices=0)
                        inference_batch.append(prompt)
                    inference_batch = inference_batch * batch_replication
                    outputs = llm.generate(inference_batch, sampling_params, use_tqdm=True)
            # outputs = llm.generate(inference_batch, sampling_params, use_tqdm=True)

            # for i, output in enumerate(outputs):
            #     generated_text = output.outputs[0].text
            #     # print(generated_text)
            #     self.examples[i]["prompt_messages"] = [sampler._pack_message(content=format_multichoice_question(self.examples[i], conf_mode="sampling", choices=0), role="user")] # such formatting to fit later analysis pipeline
            #     self.examples[i]["response"] = generated_text
            #     prob_dict_list = output.outputs[0].logprobs
            #     top_logprob_lst = []
            #     logprobs = []
            #     for d in prob_dict_list:
            #         top_logprob_lst.append({x.decoded_token: x.logprob for x in d.values()})
            #         logprobs += [x.logprob for x in d.values() if x.rank == 1]
            #     self.examples[i]["logprobs"] = (logprobs)
            #     self.examples[i]["top_logprobs"] = (top_logprob_lst)

            num_outputs_per_example = batch_replication
            for i in range(0, len(outputs), num_outputs_per_example):
                example_idx = i // num_outputs_per_example

                # Prepare prompt
                self.examples[example_idx]["prompt_messages"] = [
                    sampler._pack_message(
                        content=format_multichoice_question(self.examples[example_idx], conf_mode="sampling", choices=0),
                        role="user"
                    )
                ]

                # Initialize lists to store repeated outputs
                self.examples[example_idx]["rep_responses"] = []
                self.examples[example_idx]["rep_logprobs"] = []
                self.examples[example_idx]["rep_top_logprobs"] = []

                for j in range(num_outputs_per_example):
                    output = outputs[i + j]
                    generated_text = output.outputs[0].text
                    prob_dict_list = output.outputs[0].logprobs

                    # Extract logprobs
                    top_logprob_lst = []
                    logprobs = []

                    for d in prob_dict_list:
                        top_logprob_lst.append({x.decoded_token: x.logprob for x in d.values()})
                        logprobs += [x.logprob for x in d.values() if x.rank == 1]

                    # Save all 10 to rep_* keys
                    self.examples[example_idx]["rep_responses"].append(generated_text)
                    self.examples[example_idx]["rep_logprobs"].append(logprobs)
                    self.examples[example_idx]["rep_top_logprobs"].append(top_logprob_lst)

                    # Save the first one separately
                    if j == 0:
                        self.examples[example_idx]["response"] = generated_text
                        self.examples[example_idx]["logprobs"] = logprobs
                        self.examples[example_idx]["top_logprobs"] = top_logprob_lst


            model_name = sampler.model + ("-think" if hasattr(sampler, "think") and sampler.think else "")
            regen_stored_path = shared_sampling_path("mmlu_pro", model_name, self.conf_mode, self.num_examples, None)
            with open(regen_stored_path, 'wb') as f:
                pickle.dump(self.examples, f)
                print(f"Shared sampling with vLLM completed and saved to: {regen_stored_path}")
            sys.exit()
        # ----------------------------------------------------------------------------------------------

        # Batch eval all
        # ----------------------------------------------------------------------------------------------
        elif self.conf_mode == "batch_eval_all":
            def truncate_by_words(text, max_words=10240):
                words = text.split()
                if len(words) < max_words:
                    print("truncated")
                    return ' '.join(words[:max_words])
                else:
                    return text
            model_name = sampler.model + ("-think" if hasattr(sampler, "think") and sampler.think else "")
            regen_stored_path = shared_sampling_path("mmlu_pro", model_name, "sampling", self.num_examples, None)
            print(regen_stored_path)
            if os.path.exists(regen_stored_path):
                print("Fetching from cache")
                self.cache_found = True
                with open(regen_stored_path, 'rb') as f:
                    self.examples = pickle.load(f)
            else:
                # exit with error
                print("No cache found. Please sample first.")
                sys.exit()
            print("vllm batch eval")
            # set up vLLM mode 
            tokenizer = self.decisiveness_grader.tokenizer
            visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if visible_gpus:
                num_gpus = len(visible_gpus.split(','))
            else:
                num_gpus = torch.cuda.device_count()  # fallback

            llm = LLM(
                model=self.decisiveness_grader.model,
                max_model_len=None,
                trust_remote_code=True,
                tokenizer_mode="auto",
                tensor_parallel_size=num_gpus
            )
            sampling_params = SamplingParams(temperature=0, max_tokens=2048, logprobs=5, seed=42, stop=[tokenizer.eos_token])
            inference_batch = []
            for i in tqdm(range(len(self.examples)), desc="Prepare verbal linguistic judge prompt batch"):
                response = self.examples[i]["response"]
                response_text = remove_verbal_confidence(normalize_response(response)) # remove verbal confidence to avoid judgement biases
                response_text = truncate_by_words(response_text)
                question = format_multichoice_question(self.examples[i], conf_mode="decisiveness_grading", choices=0)
                
                prompt = [sampler._pack_message("system", sampler.system_message), 
                          sampler._pack_message(content=LINGUISTIC_CONFIDENCE_GRADER_PROMPT.format(Question=question, Response=response_text), role="user")]
                inference_batch.append(tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=True))

            outputs = llm.generate(inference_batch, sampling_params, use_tqdm=True)
            # conf_list = []
            for i, output in enumerate(outputs):
                judge_generated_text = output.outputs[0].text
                if extract_answer(self.examples[i]["response"]) == None or extract_verbal_numerical_confidence(self.examples[i]["response"]) == None:
                    self.examples[i]["linguistic_confidence"] = None 
                    self.examples[i]["linguistic_judge_response"] = judge_generated_text
                else:
                    score_patterns = [
                        r"[Cc]onfidence [Ss]core:\s*([0-1]*\.?[0-9]+)",
                        r"confidence score is\s*([0-1]*\.?[0-9]+)",
                        r"answer is\s*([0-1]*\.?[0-9]+)",
                    ]
                    for score_pattern in score_patterns:
                        scores = re.findall(score_pattern, judge_generated_text)
                        if scores:
                            print(f"deceiveness score: {np.mean([float(score) for score in scores])}")
                            # conf_list.append(np.mean([float(score) for score in scores]))
                            self.examples[i]["linguistic_confidence"] = np.mean([float(score) for score in scores])
                            break
                        else:
                            print("No score")
                            self.examples[i]["linguistic_confidence"] = None 
                    self.examples[i]["linguistic_judge_response"] = judge_generated_text
                
                
            # self.conf_mode = "eval_all"
            results = common.map_with_progress(fn, self.examples)
            results_df = pd.DataFrame()
            for result in results:
                new_row = pd.DataFrame({
                    "category":[result.category],
                    "correct_answer": [result.correct_answer],
                    "extracted_answer": [result.extracted_answer],
                    "verbal_numerical_confidence": [result.verbal_numerical_confidence],
                    "logit_perplexity_confidence": [result.logit_perplexity_confidence],
                    "verbal_linguistic_confidence": [result.verbal_linguistic_confidence]
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)
            file_stem = f"""mmlu_pro_{model_name.split("/")[-1]}_{self.conf_mode}_{self.num_examples}"""
            csv_filename = f"""LLM-Calibration-Study/results/{file_stem}.csv"""
            results_df.to_csv(csv_filename)
            return common.aggregate_results(results)
        # ----------------------------------------------------------------------------------------------

        model_name = sampler.model + ("-think" if hasattr(sampler, "think") and sampler.think else "")
        if self.conf_mode in ["sampling", "eval_all", "semantic_entropy"] or "_shared_sampling" in self.conf_mode:
            self.regenerate = True
            regen_stored_path = shared_sampling_path("mmlu_pro", model_name, self.conf_mode, self.num_examples, None)
        else:
            regen_stored_path = ind_sampling_path("mmlu_pro", model_name, self.conf_mode, self.num_examples, None)

        if self.regenerate:
            if "tmp" in self.conf_mode:
                regen_stored_path = shared_sampling_path(f"tmp_mmlu_pro", model_name, self.conf_mode, self.num_examples, None)
                if os.path.exists(regen_stored_path):
                    self.cache_found = True
                    print("Fetching from cache (partial dataset)")
                    with open(regen_stored_path, 'rb') as f:
                        self.examples = pickle.load(f)
                    results = common.map_with_progress(fn, self.examples)
                    with open(regen_stored_path, 'wb') as f:
                        pickle.dump(self.examples, f)
                else:
                    sys.exit()
            if os.path.exists(regen_stored_path):
                print("Fetching from cache")
                with open(regen_stored_path, 'rb') as f:
                    self.examples = pickle.load(f)
                    # self.examples = random.Random(0).sample(self.examples, 100)
                self.cache_found = True
                results = common.map_with_progress(fn, self.examples)
            else:
                print("No cache")
                results = common.map_with_progress(fn, self.examples)
                with open(regen_stored_path, 'wb') as f:
                    pickle.dump(self.examples, f)
            
        else:
            results = common.map_with_progress(fn, self.examples)

        if self.conf_mode == "sampling":
            with open(regen_stored_path, 'wb') as f:
                pickle.dump(self.examples, f)
            print(f"Shared sampling completed and saved to: {regen_stored_path}")
            sys.exit()
        
        if self.conf_mode == "eval_all":
            results_df = pd.DataFrame()
            for result in results:
                new_row = pd.DataFrame({
                    "category":[result.category],
                    "correct_answer": [result.correct_answer],
                    "extracted_answer": [result.extracted_answer],
                    "verbal_numerical_confidence": [result.verbal_numerical_confidence],
                    "logit_perplexity_confidence": [result.logit_perplexity_confidence],
                    "verbal_linguistic_confidence": [result.verbal_linguistic_confidence],
                    "token_length": [result.token_length],
                })
                results_df = pd.concat([results_df, new_row], ignore_index=True)
            file_stem = f"""mmlu_pro_{model_name.split("/")[-1]}_{self.conf_mode}_{self.num_examples}"""
            csv_filename = f"""LLM-Calibration-Study/results/{file_stem}.csv"""
            results_df.to_csv(csv_filename)
        
        elif self.conf_mode == "semantic_entropy":
            semantic_confidences = []
            semantic_answers = []
            for result in results:
                semantic_confidences.append(result.semantic_entropy)
                semantic_answers.append(result.semantic_entropy_answer)
            file_stem = f"""mmlu_pro_{model_name.split("/")[-1]}_{"eval_all"}_{self.num_examples}"""
            csv_filename = f"""LLM-Calibration-Study/results/{file_stem}.csv"""
            print(semantic_answers)
            if os.path.exists(csv_filename):
                results_df = pd.read_csv(csv_filename)
                results_df["semantic_entrory_answer"] = semantic_answers
                results_df["semantic_confidences"] = semantic_confidences
                results_df.to_csv(csv_filename)


        return common.aggregate_results(results)