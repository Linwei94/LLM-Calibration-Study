"""
Measuring Massive Multitask Language Understanding
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300
"""

import random
import pickle
import os
import pandas
import sys
from datasets import load_dataset

from .sampler.chat_completion_sampler import ChatCompletionSampler

from . import common
from .common import (
    HTML_JINJA,
    format_multichoice_question,
    normalize_response,
    extract_answer_and_confidence,
)
from .custom_types import Eval, EvalResult, SamplerBase, SingleEvalResult

from .utils.report_post_processing import *
from .utils.confidence_post_processing import *
from .utils.pre_processing import *

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
    def __init__(self, decisiveness_grader: ChatCompletionSampler, num_examples: int | None = None, conf_mode: str = "verbal_vanilla", regenerate=True):
        self.examples = preprocess(load_dataset("TIGER-Lab/MMLU-Pro")["test"])
        if num_examples:
            self.examples = random.Random(0).sample(self.examples, num_examples)
        self.conf_mode = conf_mode
        self.regenerate = regenerate
        self.num_examples = num_examples
        self.cache_found = False
        self.decisiveness_grader = decisiveness_grader

    def __call__(self, sampler: ChatCompletionSampler) -> EvalResult:
        def fn(row: dict):

            sampling = 5
            extracted_answer = ""
            confidence = 0

            match self.conf_mode:
                
                case "verbal_numerical" | "verbal_numerical_shared_sampling":
                    if self.cache_found:
                        response = row["response"] 
                        top_logprobs = row["top_logprobs"] 
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
                        row["logit_perplexity"] = sampler.logit_perplexity

                    logprobs = row["logprobs"]
                    response_text = normalize_response(response)

                    extracted_answer, confidence = extract_answer_and_confidence("\n".join(response_text.splitlines()[-2:]), options=dict(zip([chr(ord('A') + i) for i in range(len(row["options"]))], row["options"])))
                    if extracted_answer is None or extracted_answer not in "ABCDEFGHIJ" or float(confidence) < 10:
                        extracted_answer, confidence = extract_answer_and_confidence(response_text, options=dict(zip([chr(ord('A') + i) for i in range(len(row["options"]))], row["options"])))
                    confidence /= 100
                    print(f"extracted_answer: {extracted_answer}, confidence: {confidence}")


                case "logit_perplexity" | "logit_perplexity_shared_sampling":
                    sampler.logprobs = True
                    if self.cache_found:
                        response = row["response"] 
                        top_logprobs = row["top_logprobs"] 
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
                        row["logit_perplexity"] = sampler.logit_perplexity

                    response_text = normalize_response(response)
                    confidence = row["logit_perplexity"]
                    logprobs = row["logprobs"]

                    extracted_answer, _ = extract_answer_and_confidence("\n".join(response_text.splitlines()[-2:]), options=dict(zip([chr(ord('A') + i) for i in range(len(row["options"]))], row["options"])))
                    if extracted_answer is None or extracted_answer not in "ABCDEFGHIJ":
                        extracted_answer, _ = extract_answer_and_confidence(response_text, options=dict(zip([chr(ord('A') + i) for i in range(len(row["options"]))], row["options"])))
                    if extracted_answer is None or extracted_answer not in "ABCDEFGHIJ":
                        extracted_answer = extract_mcq_answer("\n".join(response_text.splitlines()[-3:]), "mmlu_pro")

                    print(f"extracted_answer: {extracted_answer}, confidence: {confidence}")


                case "verbal_linguistic" | "verbal_linguistic_shared_sampling":
                    if self.cache_found:
                        response = row["response"] 
                        top_logprobs = row["top_logprobs"] 
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
                        row["logit_perplexity"] = sampler.logit_perplexity

                    logprobs = row["logprobs"]
                    response_text = normalize_response(response)

                    extracted_answer, _ = extract_answer_and_confidence("\n".join(response_text.splitlines()[-2:]), options=dict(zip([chr(ord('A') + i) for i in range(len(row["options"]))], row["options"])))
                    if extracted_answer is None or extracted_answer not in "ABCDEFGHIJ":
                        extracted_answer, _ = extract_answer_and_confidence(response_text, options=dict(zip([chr(ord('A') + i) for i in range(len(row["options"]))], row["options"])))
                    if extracted_answer is None or extracted_answer not in "ABCDEFGHIJ":
                        extracted_answer = extract_mcq_answer("\n".join(response_text.splitlines()[-3:]), "mmlu_pro")

                    confidence = decisiveness_score(self.decisiveness_grader, format_multichoice_question(row, conf_mode="verbal_linguistic", choices=0), response_text)

                case "sampling":
                    sampler.logprobs = True
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
                    row["logit_perplexity"] = sampler.logit_perplexity
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
                conf_mode = self.conf_mode
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html, score=score, metrics={category: score}, convo=convo, confidence=float(confidence)
            )

        # Run evaluation and collect results
        if self.conf_mode in ["sampling", "eval_all"] or "_shared_sampling" in self.conf_mode:
            self.regenerate = True
            regen_stored_path = shared_sampling_path("mmlu_pro", sampler.model, self.conf_mode, self.num_examples, None)
        else:
            regen_stored_path = ind_sampling_path("mmlu_pro", sampler.model, self.conf_mode, self.num_examples, None)
            

        if self.regenerate:
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
            print(f"Shared sampling complete and saved to: {regen_stored_path}")
            sys.exit()

        return common.aggregate_results(results)
