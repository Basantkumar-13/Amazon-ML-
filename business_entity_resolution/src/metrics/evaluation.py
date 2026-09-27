import numpy as np


def f_beta_score(precision, recall, beta=0.5):
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)


def compute_macro_f05(ground_truth_dict, predictions_dict):
  
    all_s1_ids = set(ground_truth_dict.keys()) | set(predictions_dict.keys())
    
    scores = []
    for s1_id in all_s1_ids:
        true_matches = ground_truth_dict.get(s1_id, set())
        pred_matches = predictions_dict.get(s1_id, set())
        
        if len(pred_matches) == 0 and len(true_matches) == 0:
            # both empty = singleton correctly identified
            scores.append(1.0)
            continue
        
        if len(pred_matches) == 0:
            scores.append(0.0)
            continue
        
        if len(true_matches) == 0:
            scores.append(0.0)
            continue
        
        tp = len(pred_matches & true_matches)
        precision = tp / len(pred_matches)
        recall = tp / len(true_matches)
        scores.append(f_beta_score(precision, recall, beta=0.5))
    
    return np.mean(scores)
