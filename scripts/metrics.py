import csv
import os
from collections import defaultdict

import numpy as np
from lifelines.utils import concordance_index
from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn.metrics import average_precision_score
from sklearn.metrics import ndcg_score
from sklearn.metrics import roc_auc_score


def cal_kendall(preds, responses):
    tau, _ = kendalltau(preds, responses)
    return tau


def cal_spearman(preds, responses):
    rho, _ = spearmanr(preds, responses)
    return rho


def cal_pearson(preds, responses):
    r, _ = pearsonr(preds, responses)
    return r


def cal_ci(preds, responses):
    ci = concordance_index(responses, preds)
    return ci


def cal_ndcg_at_k(preds, labels, k=10):
    labels = np.array(labels)
    true_rel = labels.astype(float)
    return ndcg_score(np.array([true_rel]), np.array([preds]), k=k)


def cal_precision_at_k(preds, labels, k=10):
    preds = np.array(preds)
    labels = np.array(labels)

    top_k_indices = np.argsort(-preds)[:k]

    top_k_labels = labels[top_k_indices]
    precision = np.sum(top_k_labels) / k

    return precision


def cal_mrr(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)

    order = np.argsort(-preds)
    sorted_labels = labels[order]

    relevant_positions = np.where(sorted_labels == 1)[0]

    if len(relevant_positions) == 0:
        return 0.0

    first_rank = relevant_positions[0] + 1
    mrr_score = 1.0 / first_rank
    return mrr_score


def cal_auroc(preds, labels):
    if len(np.unique(labels)) < 2:
        return np.nan
    return roc_auc_score(labels, preds)


def cal_auprc(preds, labels):
    if len(np.unique(labels)) < 2:
        return np.nan
    return average_precision_score(labels, preds)


def compute_metrics(cell_id, drug_id, pred, response, label, id2name, config, mode=None):
    cell_data = defaultdict(lambda: {'drug_ids': [], 'preds': [], 'responses': [], 'labels': []})
    pred_pc = defaultdict(dict)
    dict_metrics = {}
    for c, d, p, r, l in zip(cell_id, drug_id, pred, response, label):
        cell_data[c]['drug_ids'].append(d)
        cell_data[c]['preds'].append(p)
        cell_data[c]['responses'].append(r)
        cell_data[c]['labels'].append(l)
        pred_pc[c][d] = p

    metrics = {
        'Kendall': [], 'Spearman': [], 'pearson': [], 'CI': [],
        'NDCG@k': [], 'Precision@k': [],
        'AUROC': [], 'AUPRC': [],

    }
    for cell_id, data in cell_data.items():
        responses, preds, labels = data['responses'], data['preds'], data['labels']
        if len(responses) < 2:
            continue
        dict_metrics[cell_id] = {}
        dict_metrics[cell_id]['Kendall'] = cal_kendall(preds, responses)
        dict_metrics[cell_id]['Spearman'] = cal_spearman(preds, responses)
        dict_metrics[cell_id]['pearson'] = cal_pearson(preds, responses)
        dict_metrics[cell_id]['CI'] = cal_ci(preds, responses)  # CI

        dict_metrics[cell_id]['NDCG@k'] = cal_ndcg_at_k(preds, labels, config.k)
        dict_metrics[cell_id]['Precision@k'] = cal_precision_at_k(preds, labels, config.k)

        dict_metrics[cell_id]['AUROC'] = cal_auroc(preds, labels)
        dict_metrics[cell_id]['AUPRC'] = cal_auprc(preds, labels)

        for key in metrics:
            metrics[key].append(dict_metrics[cell_id][key])

    output_path = os.path.join(config.fold_dir, f'{mode.lower()}_preds_per_samples.csv')
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Samples_id', 'Drug_id', 'Drug_name', 'Score'])

        for cell_id, drug_dict in pred_pc.items():
            for drug_id, pred_score in drug_dict.items():
                drug_name = id2name.get(drug_id, 'Unknown')
                writer.writerow([cell_id, drug_id, drug_name, f'{pred_score:.4f}'])

    avg_metrics = {key: np.nanmean(val) for key, val in metrics.items()}
    return avg_metrics
