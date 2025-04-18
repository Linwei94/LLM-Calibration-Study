"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

import random
import re
import os
import pandas
import pickle

from . import common
from .common import extract_answer_and_confidence, HTML_JINJA, format_multichoice_question
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

from .utils.post_processing import *
from .utils.pre_processing import *


class GPQAEval(Eval):
    def __init__(
        self,
        n_repeats: int = 4,
        variant: str = "diamond",
        num_examples: int | None = None,  # restrict to a subset of the data for debugging
        conf_mode: str = "verbal_vanilla",
        regenerate=True
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

    def __call__(self, sampler: SamplerBase) -> EvalResult:
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

            sampling = 5
            extracted_answer = ""
            confidence = 0

            match self.conf_mode:

                case "verbal_cot":
                    if self.cache_found:
                        response_tuple = row["sampler_responses"]
                        prompt_messages = row["prompt_messages"]
                    else:
                        prompt_messages = [
                            sampler._pack_message(
                                content=format_multichoice_question(choices_dict, conf_mode=self.conf_mode), role="user"
                            )
                        ]
                        response_tuple = sampler(prompt_messages)
                        row["sampler_responses"] = response_tuple
                        row["prompt_messages"] = prompt_messages
                    
                    response_text = normalize_response(response_tuple[0])
                    logprobs = response_tuple[2]
                    # Extract the answer from the response text
                    extracted_answer, confidence = extract_answer_and_confidence(response_text, options={k: choices_dict[k] for k in ['A', 'B', 'C', 'D']})
                    confidence /= 100
                
                case "verbal_vanilla":
                    if self.cache_found:
                        response_tuple = row["sampler_responses"]
                        prompt_messages = row["prompt_messages"]
                    else:
                        prompt_messages = [
                            sampler._pack_message(
                                content=format_multichoice_question(choices_dict, conf_mode=self.conf_mode), role="user"
                            )
                        ]
                        response_tuple = sampler(prompt_messages)
                        row["sampler_responses"] = response_tuple
                        row["prompt_messages"] = prompt_messages

                    response_text = normalize_response(response_tuple[0])
                    logprobs = response_tuple[2]
                    # Extract the answer from the response text
                    extracted_answer, confidence = extract_answer_and_confidence(response_text, options={k: choices_dict[k] for k in ['A', 'B', 'C', 'D']})
                    confidence /= 100
                
                case "single_generation":
                    if self.cache_found:
                        response_with_conf = row["sampler_responses"]
                        prompt_messages = row["prompt_messages"]
                    else:
                        prompt_messages = [
                            sampler._pack_message(
                                content=format_multichoice_question(choices_dict, conf_mode=self.conf_mode), role="user"
                            )
                        ]
                        response_with_conf = sampler(prompt_messages)
                        row["sampler_responses"] = response_with_conf
                        row["prompt_messages"] = prompt_messages

                    response_text, confidence, logprobs = response_with_conf 
                    extracted_answer = gpqa_regex_extract_response(response_text)


                case "empirical_semantic":
                    if self.cache_found:
                        response_with_conf = row["sampler_responses"]
                        prompt_messages = row["prompt_messages"]
                    else:
                        prompt_messages = [
                            sampler._pack_message(
                                content=format_multichoice_question(choices_dict, conf_mode=self.conf_mode), role="user"
                            )
                        ]
                        response_with_conf = [sampler(prompt_messages) for _ in range(sampling)]
                        row["sampler_responses"] = response_with_conf
                        row["prompt_messages"] = prompt_messages

                    logprobs = [(r[2]) for r in response_with_conf]
                    # extracted_answers = [gpqa_regex_extract_response(text[0]) for text in response_with_conf]
                    response_texts, lnll_lst, labels = get_mcq_clusters(response_with_conf, "gpqa")
                    response_text, confidence, index = empirical_semantic_confidence(lnll_lst, response_texts, labels)
                    extracted_answer = gpqa_regex_extract_response(response_text)
                    logprobs = response_with_conf[index][2] 

                case "faithfulness":
                    if self.cache_found:
                        response_with_conf = row["sampler_responses"]
                        prompt_messages = row["prompt_messages"]
                        candidate_sample = row["candidate_sample"]
                    else:
                        prompt_messages = [
                            sampler._pack_message(
                                content=format_multichoice_question(choices_dict, conf_mode=self.conf_mode), role="user"
                            )
                        ]
                        row["sampler_responses"] = (response_with_conf := sampler(prompt_messages))
                        row["prompt_messages"] = prompt_messages
                        row["candidate_sample"] = (candidate_sample := [sampler(prompt_messages)[0] for _ in range(sampling)])

                    response_text, _, logprobs = response_with_conf
                    extracted_answer = gpqa_regex_extract_response(response_text)
                    score = 1.0 if extracted_answer == correct_answer else 0.0
                    confM = confidence_by_contradiction(sampler, response_text, candidate_sample)
                    dec = decisiveness_score(sampler, format_multichoice_question(choices_dict), response_text)
                    confidence = float(1 - np.abs(dec - confM))


                case _:
                    raise Exception(f"Unrecognized confidence type: {self.conf_mode}")


            print(f"extracted_answer: {extracted_answer}, confidence: {confidence}")
            score = 1.0 if extracted_answer == correct_answer else 0.0
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
                verbal_confidence=float(confidence)
            )

        regen_stored_path = f"LLM-Calibration-Study/cache/gpqa_{sampler.model.split("/")[-1]}_{self.conf_mode}_{self.num_examples}_{self.n_repeats}"

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

        return common.aggregate_results(results)
