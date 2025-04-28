import numpy as np

import regex as re
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering

import torch
from sentence_transformers import CrossEncoder
from huggingface_hub import login
import os
from .post_processing_answer import *
from  .query_templates import *

# Semantic-based Confidence
# ------------------------------------------------------------------------------------------------------
def get_semantic_clusters(multi_response):
    lnll_lst = [(x)[1] for x in multi_response]
    response_list = [x[0] for x in multi_response]
    distance_threshold = 0.3
    model_name="all-MiniLM-L6-v2"
    embeddings = SentenceTransformer(model_name).encode(response_list)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric="cosine", linkage="average")
    labels = clustering.fit_predict(embeddings)
    return response_list, lnll_lst, labels


def get_mcq_clusters(multi_response, test = "mmlu"):
    lnll_lst = [(x)[1] for x in multi_response]
    response_list = [x[0] for x in multi_response]
    choice_map = dict()
    choice_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'a': 1, 'b': 2, 'c': 3, 'd': 4}
    labels = [choice_map.get(mcq_regex_extractors[test](c), 0) for c in response_list]
    return response_list, lnll_lst, labels


def empirical_semantic_confidence(lnll_lst, response_list, labels):
    counts = Counter(labels)
    opt_cluster, opt_conf = max([(int(cluster_id), count/sum(counts.values())) for cluster_id, count in counts.items()], key=lambda x: x[1])
    optimal_response, index = max([(response_list[i], i) for i, label in enumerate(labels) if label == opt_cluster], key=lambda x: x[1])
    return optimal_response, opt_conf, index
# ------------------------------------------------------------------------------------------------------



# faithful response uncertainty
# ------------------------------------------------------------------------------------------------------
def remove_verbal_confidence(text):
    cleaned_text = re.sub(r'[Cc]onfidence:? (\d+)?%?\r?\n?', '', text, flags=re.MULTILINE)
    return cleaned_text


def linguistic_confidence_score(sampler, question, response) -> tuple[str, float]:
    # response = remove_verbal_confidence(response) # remove verbal confidence from response
    msg = [sampler._pack_message("user", LINGUISTIC_CONFIDENCE_GRADER_PROMPT.format(Question=question, Response=response))]
    print(msg)
    score_pattern = r"[Cc]onfidence [Ss]core:\s*([0-9]*\.?[0-9]+)"
    verdict = sampler(msg)
    scores = re.findall(score_pattern, verdict)
    # assertions = re.findall(assertion_pattern, verdict)
    verdict = verdict.split("\n")
    decisiveness_scores = [float(score) for score in scores]
    if len(decisiveness_scores) > 0:
        return np.mean(decisiveness_scores)
    return 1
# ------------------------------------------------------------------------------------------------------


# Logits-based confidence
# ------------------------------------------------------------------------------------------------------
def calculate_logit_perplexity(logprobs):
    return float(np.exp(np.array([p for p in logprobs if p is not None])).mean()) 


login(os.environ["HF_TOKEN"])
relevance_model = CrossEncoder("sentence-transformers/all-MiniLM-L6-v2")
def token_sar_confidence(top_logprobs: list[dict[str, float]]) -> float:
    """
    Compute SAR-based confidence from only top_logprobs (one per token).
    Assumes the first token in each dict is the generated token.
    Based on https://arxiv.org/abs/2307.01379 
    
    Args:
        top_logprobs (List[Dict[str, float]]): List of token: logprob dictionaries.
        relevance_model (CrossEncoder): Semantic similarity model.
    
    Returns:
        float: SAR-style confidence score in [0, 1].
    """
    def rescale(scores):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scores = np.array(scores).reshape(-1, 1)  # sklearn expects 2D array
        normed_scores = scaler.fit_transform(scores)
        return normed_scores.flatten().tolist()
    
    # Extract selected tokens
    tokens = [list(logprob_dict.keys())[0] for logprob_dict in top_logprobs]

    # Compute entropy for each token
    entropies = []
    for logp_dict in top_logprobs:
        logprobs = torch.tensor(list(logp_dict.values()))
        probs = torch.exp(logprobs)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        entropies.append(entropy)
    
    # Compute semantic relevance by measuring meaning drop
    sentence = " ".join(tokens)
    pairs = []
    for i in range(len(tokens)):
        reduced = " ".join(tokens[:i] + tokens[i+1:])
        pairs.append((sentence, reduced))

    # Batch prediction
    sims = relevance_model.predict(pairs, show_progress_bar=True)

    # Some cross encoders do not scale from 0-1; uncomment the next line if it's the case 
    # sims = rescale(sims) 

    # Convert to relevances
    relevances = [max(0.0, 1.0 - sim) for sim in sims]
    
    # Normalise relevance
    total_rel = sum(relevances) + 1e-8
    weights = [r / total_rel for r in relevances]

    # Compute token-level SAR
    token_SAR = sum(e * w for e, w in zip(entropies, weights))

    # Convert to confidence score
    confidence = 1 / (1 + token_SAR)
    return confidence
# ------------------------------------------------------------------------------------------------------
