import torch
import torch.nn.functional as F
from torch import nn


class ListwiseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_label, y_true_sens):
        p_pred = F.softmax(y_pred, dim=0) + 1e-9
        p_label = F.softmax(y_true_sens, dim=0) + 1e-9

        loss = -torch.sum(p_label * torch.log(p_pred))
        return {'total_loss': loss}


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.MSEloss = nn.MSELoss()

    def forward(self, y_pred, y_label, y_true_sens):
        loss = self.MSEloss(y_pred, y_true_sens)
        return {'total_loss': loss}


class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.BCELoss = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_label, y_true_sens):
        y_label = y_label.to(device=y_pred.device, dtype=y_pred.dtype)

        loss = self.BCELoss(y_pred, y_label)
        return {'total_loss': loss}


class PairwiseLoss(nn.Module):
    def __init__(self, margin=0.5, max_pairs=300):
        super().__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.max_pairs = max_pairs

    def forward(self, y_pred, y_label=None, y_true_sens=None):

        device = y_pred.device
        dtype = y_pred.dtype

        n = y_pred.size(0)
        loss_rank = torch.tensor(0.0, device=device, dtype=dtype)

        if n < 2:
            return {'total_loss': loss_rank}

        all_pairs = torch.combinations(torch.arange(n, device=device), r=2)
        num_total = all_pairs.size(0)
        num_sample = min(self.max_pairs, num_total)

        if num_sample == 0:
            return {'total_loss': loss_rank}

        sel_idx = torch.randperm(num_total, device=device)[:num_sample]
        sampled_pairs = all_pairs[sel_idx]

        i1, i2 = sampled_pairs[:, 0], sampled_pairs[:, 1]
        p1, p2 = y_pred[i1], y_pred[i2]
        r1, r2 = y_true_sens[i1], y_true_sens[i2]

        target = torch.sign(r1 - r2)
        valid_mask = (target != 0)

        if valid_mask.any():
            loss_rank += self.ranking_loss(p1[valid_mask], p2[valid_mask], target[valid_mask])

        return {'total_loss': loss_rank}


class HierarchicalLoss(nn.Module):
    def __init__(self, alpha=0.5, margin=0.5, max_pairs=150):
        super().__init__()
        self.alpha = alpha
        self.classification_loss = nn.BCEWithLogitsLoss()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.max_pairs = max_pairs

    def forward(self, y_pred, y_label, y_true_sens):
        device = y_pred.device
        dtype = y_pred.dtype

        loss_cls = self.classification_loss(y_pred, y_label.float())

        loss_rank = torch.tensor(0.0, device=device, dtype=dtype)

        pos_mask = (y_label == 1)
        neg_mask = (y_label == 0)

        pos_idx = torch.where(pos_mask)[0]
        neg_idx = torch.where(neg_mask)[0]

        if len(pos_idx) >= 2:
            sampled_pos = pos_idx[torch.randperm(len(pos_idx), device=device)[:self.max_pairs]]
            loss_rank += self._compute_pairwise_rank(y_pred[sampled_pos], y_true_sens[sampled_pos])

        if len(neg_idx) >= 2:
            sampled_neg = neg_idx[torch.randperm(len(neg_idx), device=device)[:self.max_pairs]]
            loss_rank += self._compute_pairwise_rank(y_pred[sampled_neg], y_true_sens[sampled_neg])

        pn_sample = min(len(pos_idx), len(neg_idx), self.max_pairs)
        if pn_sample >= 1:
            sampled_pos = pos_idx[torch.randperm(len(pos_idx), device=device)[:pn_sample]]
            sampled_neg = neg_idx[torch.randperm(len(neg_idx), device=device)[:pn_sample]]

            preds_pos = y_pred[sampled_pos]
            preds_neg = y_pred[sampled_neg]
            target = torch.ones(pn_sample, device=device, dtype=dtype)
            loss_rank += self.ranking_loss(preds_pos, preds_neg, target)

        combine_loss = (1 - self.alpha) * loss_cls + self.alpha * loss_rank

        return {
            'total_loss': combine_loss,
            'cls_loss': loss_cls.detach(),
            'rank_loss': loss_rank.detach()
        }

    def _compute_pairwise_rank(self, preds, reals):
        device = preds.device
        dtype = preds.dtype
        n = preds.size(0)

        if n < 2:
            return torch.tensor(0.0, device=device, dtype=dtype)

        all_pairs = torch.combinations(torch.arange(n, device=device), r=2)
        num_total = all_pairs.size(0)
        num_sample = min(self.max_pairs, num_total)
        if num_sample == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)

        sel_idx = torch.randperm(num_total, device=device)[:num_sample]
        sampled_pairs = all_pairs[sel_idx]

        i1, i2 = sampled_pairs[:, 0], sampled_pairs[:, 1]
        p1, p2 = preds[i1], preds[i2]
        r1, r2 = reals[i1], reals[i2]

        target = torch.sign(r1 - r2)
        mask = (target != 0)

        if mask.any():
            return self.ranking_loss(p1[mask], p2[mask], target[mask])
        else:
            return torch.tensor(0.0, device=device, dtype=dtype)
