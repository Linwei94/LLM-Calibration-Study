"""
SimpleQA: Measuring short-form factuality in large language models 
Authors: Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman, William Fedus
https://cdn.openai.com/papers/simpleqa.pdf
""" 

import random 
import re
import pandas
import pickle
import os
from . import common
from .custom_types import Eval, EvalResult, SamplerBase, SingleEvalResult
from .utils.post_processing_report import *
from .utils.post_processing_confidence import *
from .utils.pre_processing import *
from .utils.query_templates import *


class SimpleQAEval(Eval):
    def __init__(self, grader_model: SamplerBase, decisiveness_grader: SamplerBase, num_examples: int | None = None, n_repeats: int = 1, conf_mode: str = "verbal_vanilla", regenerate=True, get_logprobs=True):
        df = pandas.read_csv(
            f"https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
        )
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported when max_examples = None"
            rng = random.Random(0)
            examples = rng.sample(examples, num_examples)
        self.examples = examples * n_repeats
        self.grader_model = grader_model
        self.conf_mode = conf_mode
        self.regenerate = regenerate
        self.n_repeats = n_repeats
        self.num_examples = num_examples
        self.cache_found = False
        self.decisiveness_grader = decisiveness_grader
        self.get_logprobs = get_logprobs

    def grade_sample(self, question: str, target: str, predicted_answer: str) -> str:
        grader_prompt = GRADER_TEMPLATE.format(
            question=question,
            target=target,
            predicted_answer=predicted_answer,
        )
        
        prompt_messages = [
            self.grader_model._pack_message(content=grader_prompt, role="user")
        ]
        grading_response = self.grader_model(prompt_messages)[0]
        
        match = re.search(r"(A|B|C)", grading_response)
        return match.group(0) if match else "C"  # Default to "NOT_ATTEMPTED" if no match

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):

            confidence = 0

            match self.conf_mode:

                case "verbal_numerical" | "verbal_numerical_shared_sampling":
                    if self.cache_found:
                        response_text = row["response_text"] 
                        prompt_messages = row["prompt_messages"] 
                    else:
                        prompt_messages = [
                            sampler._pack_message(content=format_open_ended_question(row=row, conf_mode="verbal_numerical"), role="user")
                        ]
                        response_text = sampler(prompt_messages)
                        row["prompt_messages"] = prompt_messages
                        row["response_text"] = response_text
                        row["logprobs"] = sampler.logprobs
                        row["top_logprobs"] = sampler.top_logprobs
                    response_text = row["response_text"]
                    logprobs = row["logprobs"]
                    confidence = extract_confidence_from_response(response_text)
                    if confidence == None: 
                        confidence = 0.0
                    else:
                        confidence = float(confidence) / 100


                case "verbal_linguistic" | "verbal_linguistic_shared_sampling":
                    if self.cache_found:
                        response_text = row["response_text"] 
                        prompt_messages = row["prompt_messages"] 
                    else:
                        prompt_messages = [
                            sampler._pack_message(content=format_open_ended_question(row=row, conf_mode="verbal_linguistic"), role="user")
                        ]
                        response_text = sampler(prompt_messages)
                        row["prompt_messages"] = prompt_messages
                        row["response_text"] = response_text
                        row["logprobs"] = sampler.logprobs
                        row["top_logprobs"] = sampler.top_logprobs
                    logprobs = row["logprobs"]
                    response_text = remove_verbal_confidence(response_text)
                    confidence = linguistic_confidence_score(self.decisiveness_grader, format_open_ended_question(row=row, conf_mode="verbal_linguistic"), response_text)


                case "logit_perplexity" | "logit_perplexity_shared_sampling":
                    if sampler.get_logprobs == False:
                        raise NotImplementedError("The selected model does not support logprobs. Force switching on by setting the model's get_logprobs to True in utils/models.py only after checking with the provider.")
                    if self.cache_found:
                        response_text = row["response_text"] 
                        prompt_messages = row["prompt_messages"] 
                    else:
                        prompt_messages = [
                            sampler._pack_message(content=format_open_ended_question(row=row, conf_mode="logit_perplexity"), role="user")
                        ]
                        response_text = sampler(prompt_messages)
                        row["prompt_messages"] = prompt_messages
                        row["response_text"] = response_text
                        row["logprobs"] = sampler.logprobs
                        row["top_logprobs"] = sampler.top_logprobs
                    logprobs = row["logprobs"]
                    confidence = calculate_logit_perplexity(logprobs)


                case "token_sar" | "token_sar_shared_sampling":
                    if sampler.get_logprobs == False:
                        raise NotImplementedError("The selected model does not support logprobs. Force switching on by setting the model's get_logprobs to True in utils/models.py only after checking with the provider.")
                    if self.cache_found:
                        response_text = row["response_text"] 
                        prompt_messages = row["prompt_messages"] 
                        top_logprobs = row["top_logprobs"]
                    else:
                        prompt_messages = [
                            sampler._pack_message(content=format_open_ended_question(row=row, conf_mode="token_sar"), role="user")
                        ]
                        response_text = sampler(prompt_messages)
                        row["prompt_messages"] = prompt_messages
                        row["response_text"] = response_text
                        row["logprobs"] = sampler.logprobs
                        row["top_logprobs"] = sampler.top_logprobs
                    logprobs = row["logprobs"]
                    confidence = token_sar_confidence(top_logprobs)

                    
                case "sampling":
                    if sampler.get_logprobs == False:
                        print("Sampling without logprobs.")
                    else:
                        print("Sampling with logprobs.")
                    prompt_messages = [
                        sampler._pack_message(content=format_open_ended_question(row=row, conf_mode="sampling"), role="user")
                    ]
                    response = sampler(prompt_messages)
                    row["prompt_messages"] = prompt_messages
                    row["response_text"] = response
                    row["logprobs"] = sampler.logprobs
                    row["top_logprobs"] = sampler.top_logprobs
                    return


                case _:
                    raise Exception(f"Unrecognized confidence type: {self.conf_mode}")


            grade_letter = self.grade_sample(row.get("problem", ""), row.get("answer", ""), response_text)
            # Metrics based on grading response
            is_correct = grade_letter == "A"
            is_incorrect = grade_letter == "B"
            is_not_attempted = grade_letter == "C"
            score = is_correct

            # Create HTML for each sample result
            html = common.jinja_env.from_string(common.HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["answer"],
                extracted_answer=response_text,
                extracted_answer_confidence=confidence,
                subject = row["metadata"],
                logprobs = logprobs,
                conf_mode = self.conf_mode
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(html=html, score=score, convo=convo, metrics={
                "is_correct": is_correct,
                "is_incorrect": is_incorrect,
                "is_not_attempted": is_not_attempted,
            }, confidence=float(confidence))


        # Run evaluation and collect results
        if self.conf_mode in ["sampling", "eval_all"] or "_shared_sampling" in self.conf_mode:
            self.regenerate = True
            regen_stored_path = shared_sampling_path("simpleqa", sampler.model, self.conf_mode, self.num_examples, self.n_repeats)
        else:
            regen_stored_path = ind_sampling_path("simpleqa", sampler.model, self.conf_mode, self.num_examples, self.n_repeats)

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

        # Aggregate metrics
        aggregate_metrics = {
            "is_correct": sum(result.metrics["is_correct"] for result in results) / len(results),
            "is_incorrect": sum(result.metrics["is_incorrect"] for result in results) / len(results),
            "is_not_attempted": sum(result.metrics["is_not_attempted"] for result in results) / len(results)
        }
        aggregate_metrics["is_given_attempted"] = aggregate_metrics["is_correct"] + aggregate_metrics["is_incorrect"]
        # Calculate accuracy_given_attempted
        aggregate_metrics["accuracy_given_attempted"] = (
            aggregate_metrics["is_correct"]
            / aggregate_metrics["is_given_attempted"]
            if aggregate_metrics["is_given_attempted"] > 0
            else 0
        )
        print("AGGREGATE METRICS") 
        print(aggregate_metrics) 
        print("##################")

        output_d = {
            "accuracy_given_attempted": aggregate_metrics["accuracy_given_attempted"],
            "f1": (
                2 * aggregate_metrics["accuracy_given_attempted"] * aggregate_metrics["is_correct"]
                / (aggregate_metrics["accuracy_given_attempted"] + aggregate_metrics["is_correct"])
                if (aggregate_metrics["accuracy_given_attempted"] + aggregate_metrics["is_correct"]) > 0
                else 0
            )
        }
        
        print(f"Accuracy Given Attempted: {output_d['accuracy_given_attempted']:.3f}")
        print(f"F1 Score: {output_d['f1']:.3f}")
        
        return common.aggregate_results(results)
    

