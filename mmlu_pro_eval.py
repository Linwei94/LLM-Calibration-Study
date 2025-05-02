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
import asyncio
import openai
from openai import AsyncOpenAI

pickle_write_lock = threading.Lock()


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
            confidence = 0
            judge_response = ""

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
                    extracted_answer, confidence = consolidated_answer_extraction(benchmark="mmlu_pro", response_text=response_text, row=row, with_verbal_confidence=True)

                
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
                    extracted_answer = consolidated_answer_extraction(benchmark="mmlu_pro", response_text=response_text, row=row, with_verbal_confidence=False)
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
                    extracted_answer = consolidated_answer_extraction(benchmark="mmlu_pro", response_text=response_text, row=row, with_verbal_confidence=False)
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
                    extracted_answer = consolidated_answer_extraction(benchmark="mmlu_pro", response_text=response_text, row=row, with_verbal_confidence=False)
                    logprobs = row["logprobs"]
                    confidence = token_sar_confidence(top_logprobs)

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


                    # save current progress
                    current_progress = self.examples.copy()
                    current_progress = [eg for eg in self.examples if "logprobs" in eg.keys()]

                    if self.conf_mode in ["sampling", "eval_all"] or "_shared_sampling" in self.conf_mode:
                        self.regenerate = True
                        progress_temp_path = shared_sampling_path(f"tmp_mmlu_pro", sampler.model, self.conf_mode, self.num_examples, None)
                    else:
                        progress_temp_path = ind_sampling_path(f"tmp_mmlu_pro", sampler.model, self.conf_mode, self.num_examples, None)
                        
                    with pickle_write_lock:
                        with open(progress_temp_path, 'wb') as f:
                            pickle.dump(current_progress, f)
                        print("Progress saved")
                    return 

                case _:
                    raise Exception(f"Unrecognized confidence type: {self.conf_mode}")

            print(f"extracted_answer: {extracted_answer}, confidence: {confidence}")
            score = 1.0 if extracted_answer == row["answer"] else 0.0

            category = row["category"]
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["answer"],
                extracted_answer=extracted_answer,
                extracted_answer_confidence=confidence,
                subject=category,
                logprobs = logprobs,
                conf_mode = self.conf_mode + "\n\n Judge Response:\n" + judge_response if "linguistic" in self.conf_mode else self.conf_mode
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html, score=score, metrics={category: score}, convo=convo, confidence=float(confidence)
            )
        
        
        # ---------------------------------- Local vLLM for sampling -----------------------------------
        if sampler.base_url == "" and self.conf_mode == "sampling":
            pass

        
        # ----------------------------------------------------------------------------------------------


        # ---------------------------------- AsyncOpenAI for Sampling ----------------------------------
        if sampler.base_url and "local" in sampler.base_url and self.conf_mode == "sampling":
            self.regenerate = True
            progress_temp_path = shared_sampling_path(f"tmp_mmlu_pro", sampler.model, self.conf_mode, self.num_examples, None)
            _inproc_lock = asyncio.Lock()
            async def stream_prompt(sampler, client: AsyncOpenAI, row: str):
                
                prompt_messages = [
                    sampler._pack_message(
                        content=format_multichoice_question(row, conf_mode="sampling", choices=0), role="user"
                    )
                ]
                # resume sampling if cache exists
                if os.path.exists(progress_temp_path):
                    with open(progress_temp_path, 'rb') as f:
                        saved_progress: list[dict] = pickle.load(f)
                        answered_questions = [eg["prompt_messages"] for eg in saved_progress]
                        if prompt_messages in answered_questions:
                            question_idx = answered_questions.index(prompt_messages)
                            row = saved_progress[question_idx]
                            print("Question found in cache, skiping to the next one")
                            return 
                if sampler.system_message:
                    message_list = [sampler._pack_message("system", sampler.system_message)] + prompt_messages 
                else:
                    message_list = prompt_messages

                print("Sampling with vLLM:", sampler.model)
                response = await client.chat.completions.create(
                    model=sampler.model,
                    messages=message_list,
                    temperature=sampler.temperature,
                    logprobs=True,
                    top_logprobs=10,
                    seed=42,
                )
                response_text = response.choices[0].message.content
                top_logprob_lst = []
                for top_list in [t.top_logprobs for t in response.choices[0].logprobs.content]: 
                    top_logprob_lst.append({t.token: t. logprob for t in top_list})
                logprobs = [t.logprob for t in response.choices[0].logprobs.content]

                row["prompt_messages"] = prompt_messages
                row["response"] = response_text
                row["logprobs"] = logprobs
                row["top_logprobs"] = top_logprob_lst
                
                current_progress = [eg for eg in self.examples.copy() if "logprobs" in eg.keys()]
                
                    # ensure directory exists
                os.makedirs(os.path.dirname(progress_temp_path), exist_ok=True)

                # filelock on progress_temp_path+".lock"
                lock = FileLock(progress_temp_path + ".lock", timeout=30)

                # first grab the in-proc lock, then the file-lock
                async with _inproc_lock:
                    with lock:  
                        # write to a temp file and atomically rename
                        tmp = progress_temp_path + ".tmp"
                        with open(tmp, "wb") as f:
                            pickle.dump(current_progress, f)
                        os.replace(tmp, progress_temp_path)
                print("Progress saved")
            async def run_all_with_progress(sampler: SamplerBase, examples: list[dict], max_concurrent: int = 20):
                client = AsyncOpenAI(
                    base_url=sampler.base_url,  # Replace with your actual vLLM endpoint
                    api_key="EMPTY"
                )
                sem = asyncio.Semaphore(max_concurrent)
                async def limited(row):
                    retry = 0
                    backoff = 1
                    while True:
                        try:
                            async with sem:
                                return await stream_prompt(sampler, client, row)
                        except openai.APITimeoutError as e:
                            retry += 1
                            backoff += 1
                            backoff = min(backoff, 60)
                            time.sleep(backoff)
                            print("Retry", retry, e)
                            continue
                        except Exception as e:
                            print("Retry", retry, e)
                            retry += 1
                            continue
                # create all tasks
                tasks = [asyncio.create_task(limited(row)) for row in examples]
                results = []
                # wrap tasks in as_completed so we can update tqdm as each one finishes
                for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="mmlu_pro"):
                    res = await coro
                    results.append(res)
                return results
            asyncio.run(run_all_with_progress(sampler, self.examples))
            regen_stored_path = shared_sampling_path("mmlu_pro", sampler.model, self.conf_mode, self.num_examples, None)
            with open(regen_stored_path, 'wb') as f:
                pickle.dump(self.examples, f)
                print(f"Shared sampling with vLLM completed and saved to: {regen_stored_path}")
            sys.exit()
        # ----------------------------------------------------------------------------------------------


        if self.conf_mode in ["sampling", "eval_all"] or "_shared_sampling" in self.conf_mode:
            self.regenerate = True
            regen_stored_path = shared_sampling_path("mmlu_pro", sampler.model, self.conf_mode, self.num_examples, None)
        else:
            regen_stored_path = ind_sampling_path("mmlu_pro", sampler.model, self.conf_mode, self.num_examples, None)

        if self.regenerate:
            if "tmp" in self.conf_mode:
                regen_stored_path = shared_sampling_path(f"tmp_mmlu_pro", sampler.model, self.conf_mode, self.num_examples, None)
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

        return common.aggregate_results(results)
