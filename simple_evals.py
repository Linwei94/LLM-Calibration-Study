import json
import argparse
import pandas as pd
import os
from . import common
from .drop_eval import DropEval
from .gpqa_eval import GPQAEval
from .humaneval_eval import HumanEval
from .math_eval import MathEval
from .mgsm_eval import MGSMEval
from .mmlu_eval import MMLUEval
from .mmlu_pro_eval import MMLUProEval
from .simpleqa_eval import SimpleQAEval
from .sampler.chat_completion_sampler import ChatCompletionSampler


def main():
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument("--model", type=str, help="Select a model by name")
    parser.add_argument("--conf_mode", type=str, help="Select a mode to extract confidence from verbal, verbal_cot", default="verbal")
    parser.add_argument("--benchmark", type=str, help="if None, use all benchmarks, otherwise use the benchmark name",
                        default="simpleqa,mmlu,mmlu_pro,math,gpqa,mgsm,drop,humaneval")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--regenerate", action="store_true", help="Run with previously generated responses")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )

    args = parser.parse_args()

    all_models = common.get_model_dict(args.model)

    if args.list_models:
        print("Available models:")
        for model_name in all_models.keys():
            print(f" - {model_name}")
        return

    if args.model:
        if args.model not in all_models:
            print(f"Error: Model '{args.model}' not found.")
            return
        models = {args.model: all_models[args.model]}

    # grading_sampler = ChatCompletionSampler(model="gpt-4o")
    # grading_sampler = ChatCompletionSampler(model="gpt-4o-mini")

    # grading_sampler = all_models["google-llama-3.1-405b-instruct-maas"]
    # equality_checker = ChatCompletionSampler(model="gpt-4-turbo-preview")

    simpleqa_grader = all_models["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"]
    equality_checker = all_models["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"]
    decisiveness_grader = all_models["meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"]
    decisiveness_grader.get_logprobs = False


    # ^^^ used for fuzzy matching, just for math
    
    def get_evals(eval_name, debug_mode, conf_mode, regenerate):
        num_examples = (
            args.examples if args.examples is not None or args.examples != 0 else (5 if debug_mode else None)
        )

        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "mmlu":
                return MMLUEval(num_examples=1 if debug_mode else num_examples, decisiveness_grader=decisiveness_grader, conf_mode=conf_mode, regenerate=regenerate)
            case "mmlu_pro":
                return MMLUProEval(num_examples=1 if debug_mode else num_examples, decisiveness_grader=decisiveness_grader, conf_mode=conf_mode, regenerate=regenerate)
            case "math":
                return MathEval(
                    equality_checker=equality_checker,
                    num_examples=num_examples,
                    n_repeats=1 if debug_mode else 10,
                )
            case "gpqa":
                return GPQAEval(
                    decisiveness_grader=decisiveness_grader,
                    n_repeats=1 if debug_mode else 1, num_examples=num_examples,
                    conf_mode=conf_mode,
                    regenerate=regenerate
                )
            case "mgsm":
                return MGSMEval(num_examples_per_lang=10 if debug_mode else 250)
            case "drop":
                return DropEval(
                    num_examples=10 if debug_mode else num_examples,
                    train_samples_per_prompt=3,
                )
            case "humaneval":
                return HumanEval(num_examples=10 if debug_mode else num_examples)
            case "simpleqa":
                return SimpleQAEval(
                    grader_model=simpleqa_grader, 
                    decisiveness_grader=decisiveness_grader,
                    num_examples=10 if debug_mode else num_examples,
                    conf_mode=conf_mode,
                    regenerate=regenerate
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    evals = {
        eval_name: get_evals(eval_name, args.debug, args.conf_mode, args.regenerate) for eval_name in args.benchmark.split(",")
    }
    os.makedirs("LLM-Calibration-Study/cache", exist_ok=True)
    os.makedirs("LLM-Calibration-Study/results", exist_ok=True)
    print(evals)
    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}
    for model_name, sampler in models.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            if "linguistic" in args.conf_mode:
                file_stem = f"""linguistic-judges/{eval_name}_{model_name.split("/")[-1]}_{args.conf_mode}_{args.examples}_{decisiveness_grader.model.split("/")[-1]}_dec_judge"""
            else:
                file_stem = f"""{eval_name}_{model_name.split("/")[-1]}_{args.conf_mode}_{args.examples}"""
            report_filename = f"""LLM-Calibration-Study/results/{file_stem}{debug_suffix}.html"""
            if not os.path.exists("LLM-Calibration-Study/results"):
                os.makedirs("LLM-Calibration-Study/results")
            print(f"""Writing report to {report_filename}""")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result, model_name, args.conf_mode, eval_name))
            metrics = result.metrics | {"score": result.score}
            print(metrics)
            result_filename = f"""LLM-Calibration-Study/results/{file_stem}{debug_suffix}.json"""
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"""Writing results to {result_filename}""")
            mergekey2resultpath[f"""{file_stem}"""] = result_filename
    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()