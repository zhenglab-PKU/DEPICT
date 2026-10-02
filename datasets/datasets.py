import random
from collections import defaultdict
from itertools import chain

import numpy as np
from torch.utils.data import Dataset


def get_thresholds(data, sensitivity_cutoff=80):
    response_list = defaultdict(list)

    for s in data:
        response_list[s.cell_id].append(s.response)

    cut_off = {
        cell_id: np.nanpercentile(responses, sensitivity_cutoff, method='nearest')
        for cell_id, responses in response_list.items()
    }

    return cut_off


def get_waterfall_thresholds(data, max_frac=0.2):
    thresholds = {}

    response_list = defaultdict(list)
    for s in data:
        response_list[s.cell_id].append(s.response)

    for cell_id, responses in response_list.items():
        responses_sorted = np.sort(responses)
        n_drugs = len(responses_sorted)

        max_idx = max(1, int(n_drugs * max_frac))

        x = np.arange(1, n_drugs + 1)
        y = responses_sorted
        line = y[0] + (y[-1] - y[0]) / (n_drugs - 1) * (x - 1)

        distances = np.abs(y - line)

        threshold_idx = np.argmax(distances)
        threshold = responses_sorted[threshold_idx]

        if threshold_idx + 1 > max_idx:
            threshold = responses_sorted[max_idx - 1]

        thresholds[cell_id] = threshold

    return thresholds


class DataPoint:

    def __init__(self, cell_id, drug_id, response=None, label=None, target=None, **kwargs):
        self.cell_id = cell_id
        self.drug_id = drug_id
        self.response = np.float32(response)
        self.label = label
        self.target = target
        for k, v in kwargs.items():
            setattr(self, k, v)


class TrainDatasets(Dataset):
    def __init__(self, data, config):

        self.data = data
        self.threshold = get_thresholds(data, config.sensitivity_cutoff)
        # self.threshold = get_waterfall_thresholds(data, max_frac=0.2)

        for s in self.data:
            s.label = int(s.response >= self.threshold[s.cell_id])

        self.cell_ids = list(self.threshold.keys())
        self.sens = defaultdict(list)
        self.insens = defaultdict(list)

        for s in self.data:
            self.sens[s.cell_id].append(s) if s.label == 1 else self.insens[s.cell_id].append(s)

        self.sampling_neg = config.sampling_neg

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        pos_samples = self.sens[cell_id]
        neg_samples = self.insens[cell_id]

        pos_chosen = len(pos_samples)
        neg_chosen = len(pos_samples)

        if len(neg_samples) <= pos_chosen:
            neg_chosen = len(neg_samples)
        if self.sampling_neg:
            samples = random.sample(pos_samples, pos_chosen) + random.sample(neg_samples, neg_chosen)
        else:
            samples = pos_samples + neg_samples

        return samples

    @staticmethod
    def collate_fn(batch_data):
        return list(chain(*batch_data))


class TestDatasets(Dataset):
    def __init__(self, data, config):
        self.data = data
        self.threshold = get_thresholds(data, config.sensitivity_cutoff)

        for s in self.data:
            s.label = int(s.response >= self.threshold[s.cell_id])

        self.cell_ids = list(self.threshold.keys())
        self.sens = defaultdict(list)
        self.insens = defaultdict(list)

        for s in self.data:
            self.sens[s.cell_id].append(s) if s.label == 1 else self.insens[s.cell_id].append(s)

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        cell_id = self.cell_ids[idx]
        pos_samples = self.sens[cell_id]
        neg_samples = self.insens[cell_id]

        samples = pos_samples + neg_samples

        return samples

    @staticmethod
    def collate_fn(batch_data):
        return list(chain(*batch_data))
