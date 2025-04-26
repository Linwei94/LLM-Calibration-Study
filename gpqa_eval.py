"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

import random
import os
import pandas
import pickle
import sys
from .sampler.chat_completion_sampler import ChatCompletionSampler
from . import common
from .common import HTML_JINJA, format_multichoice_question
from .custom_types import Eval, EvalResult, SamplerBase, SingleEvalResult
from .utils.post_processing_report import *
from .utils.post_processing_confidence import *
from .utils.post_processing_answer import *
from .utils.pre_processing import *


class GPQAEval(Eval):
    def __init__(
        self, 
        decisiveness_grader: SamplerBase,
        n_repeats: int = 4,
        variant: str = "diamond",
        num_examples: int | None = None,  # restrict to a subset of the data for debugging
        conf_mode: str = "verbal_vanilla",
        regenerate=True,
        get_logprobs=True
    ):
        df = pandas.read_csv(
            f"https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{variant}.csv"
        )
        examples = [row.to_dict() for _, row in df.iterrows()]
        rng = random.Random(0)
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            examples = rng.sample(examples, num_examples)
        examples = examples * n_repeats
        examples = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]
        self.examples = examples
        self.n_repeats = n_repeats
        self.conf_mode = conf_mode
        self.regenerate = regenerate
        self.cache_found = False
        self.num_examples = num_examples
        self.decisiveness_grader = decisiveness_grader
        self.get_logprobs = get_logprobs

    def __call__(self, sampler: ChatCompletionSampler) -> EvalResult:
        def fn(row: dict):
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            choices = [choices[i] for i in row["permutation"]]
            correct_index = choices.index(row["Correct Answer"])
            correct_answer = "ABCD"[correct_index]
            choices_dict = dict(
                A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=row["Question"]
            )

            extracted_answer = ""
            confidence = 0

            match self.conf_mode:

                case "verbal_numerical" | "verbal_numerical_shared_sampling":
                    if self.cache_found:
                        response = row["response"] 
                        prompt_messages = row["prompt_messages"] 
                    else:
                        prompt_messages = [
                            sampler._pack_message(
                                content=format_multichoice_question(choices_dict, conf_mode="verbal_numerical"), role="user"
                            )
                        ]
                        response = sampler(prompt_messages)
                        row["prompt_messages"] = prompt_messages
                        row["response"] = response
                        row["logprobs"] = sampler.logprobs
                        row["top_logprobs"] = sampler.top_logprobs
                    logprobs = row["logprobs"]
                    response_text = normalize_response(response)
                    extracted_answer, confidence = consolidated_answer_extraction(benchmark="gpqa", response_text=response_text, choices_dict=choices_dict, with_verbal_confidence=True)
                

                case "verbal_linguistic" | "verbal_linguistic_shared_sampling":
                    if self.cache_found:
                        response = row["response"] 
                        prompt_messages = row["prompt_messages"] 
                    else:
                        prompt_messages = [
                            sampler._pack_message(
                                content=format_multichoice_question(choices_dict, conf_mode="verbal_linguistic"), role="user"
                            )
                        ]
                        response = sampler(prompt_messages)
                        row["prompt_messages"] = prompt_messages
                        row["response"] = response
                        row["logprobs"] = sampler.logprobs
                        row["top_logprobs"] = sampler.top_logprobs

                    response_text = remove_verbal_confidence(normalize_response(response)) # remove verbal confidence to avoid judgement biases
                    extracted_answer = consolidated_answer_extraction(benchmark="gpqa", response_text=response_text, choices_dict=choices_dict, with_verbal_confidence=False)
                    logprobs = row["logprobs"]
                    confidence = decisiveness_score(self.decisiveness_grader, format_multichoice_question(choices_dict, "verbal_linguistic"), response_text)


                case "logit_perplexity" | "logit_perplexity_shared_sampling":
                    if sampler.get_logprobs == False:
                        raise NotImplementedError("The selected model does not support logprobs. Force switching on by setting the model's get_logprobs to True in utils/models.py only after checking with the provider.")
                    if self.cache_found:
                        response = row["response"] 
                        prompt_messages = row["prompt_messages"] 
                    else:
                        prompt_messages = [
                            sampler._pack_message(
                                content=format_multichoice_question(choices_dict, conf_mode="logit_perplexity"), role="user"
                            )
                        ]
                        response = sampler(prompt_messages)
                        row["prompt_messages"] = prompt_messages
                        row["response"] = response
                        row["logprobs"] = sampler.logprobs
                        row["top_logprobs"] = sampler.top_logprobs
                    response_text = normalize_response(response)
                    extracted_answer = consolidated_answer_extraction(benchmark="gpqa", response_text=response_text, choices_dict=choices_dict, with_verbal_confidence=False)
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
                    extracted_answer = consolidated_answer_extraction(benchmark="gpqa", response_text=response_text, choices_dict=choices_dict, with_verbal_confidence=False)
                    logprobs = row["logprobs"]
                    confidence = token_sar_confidence(top_logprobs)
                
                    
                case "sampling":
                    if sampler.get_logprobs == False:
                        print("Sampling without logprobs.")
                    else:
                        print("Sampling with logprobs.")
                    prompt_messages = [
                        sampler._pack_message(
                            content=format_multichoice_question(choices_dict, conf_mode=self.conf_mode), role="user"
                        )
                    ]
                    response = sampler(prompt_messages)
                    row["prompt_messages"] = prompt_messages
                    row["response"] = response
                    row["logprobs"] = sampler.logprobs
                    row["top_logprobs"] = sampler.top_logprobs
                    return
                

                case _:
                    raise Exception(f"Unrecognized confidence type: {self.conf_mode}")

            score = 1.0 if extracted_answer == correct_answer else 0.0
            print(f"extracted_answer: {extracted_answer}, confidence: {confidence}")

            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
                extracted_answer_confidence=confidence,
                subject=row["Record ID"],
                logprobs = logprobs,
                conf_mode = self.conf_mode
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html, 
                score=score, 
                convo=convo, 
                metrics={"chars": len(response_text)}, 
                confidence=float(confidence)
            )


        # Run evaluation and collect results
        if self.conf_mode in ["sampling", "eval_all"] or "_shared_sampling" in self.conf_mode:
            self.regenerate = True
            regen_stored_path = shared_sampling_path("gpqa", sampler.model, self.conf_mode, self.num_examples, self.n_repeats)
        else:
            regen_stored_path = ind_sampling_path("gpqa", sampler.model, self.conf_mode, self.num_examples, self.n_repeats)
            

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
