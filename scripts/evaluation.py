import logging
from itertools import groupby

import torch

from datasets.utils import get_feats
from scripts.metrics import compute_metrics

logger = logging.getLogger(__name__)


def log_metrics(metric, mode, epoch=None, fold=None):
    for k, v in metric.items():
        if epoch is None:
            # TEST: (epoch is None)
            if fold:
                logger.info('%s : %s for fold %d = %.4f' % (mode, k, fold, v))
            else:
                logger.info('%s : Avg %s = %.4f' % (mode, k, v))
        else:
            # VAL/TRAIN: (epoch)
            if fold:
                logger.info('%s (Epoch %d) : %s for fold %d = %.4f' % (mode, epoch, k, fold, v))
            else:
                logger.info('%s (Epoch %d) : Avg %s = %.4f' % (mode, epoch, k, v))


def evaluate(args, config, fold, model, dataloader, id2name, criterion=None, epoch=None, expression=None, mutation=None,
             mode=None):
    all_cell_ids = []
    all_drug_ids = []
    all_preds = []
    all_response = []
    all_labels = []

    total_loss, cls_loss, rank_loss = 0, 0, 0

    for batch in dataloader:

        # data extraction for training
        cell_ids = [s.cell_id for s in
                    batch]  # list cell_id: ["SIDM00631",...] [B=batch_size*(all_neg+all_pos)≈32*(40+150)]

        drug_ids = torch.tensor([s.drug_id for s in batch], dtype=torch.long,
                                device=config.device)  # tensor: torch.Size([B])
        targets = torch.tensor([s.target for s in batch], device=config.device)  # tensor: torch.Size([B])

        responses = torch.tensor([s.response for s in batch], device=config.device)  # tensor: torch.Size([B])
        labels = torch.tensor([s.label for s in batch], device=config.device)  # tensor: torch.Size([B])

        mut_feats, exp_feats = None, None
        if args.input in ['all', 'mut']:
            mut_feats = get_feats(cell_ids, mutation).to(config.device)  # [B * mut_dim]
        if args.input in ['all', 'exp']:
            exp_feats = get_feats(cell_ids, expression).to(config.device)  # [B * exp_dim]
        if mut_feats is None and exp_feats is None:
            raise ValueError(f"Unknown input type: {args.input}")

        preds = model(
            drug_id=drug_ids,
            mut_feat=mut_feats,
            exp_feat=exp_feats,
            target=targets
        )  # tensor: torch.Size([B])

        # calculate the loss
        cell_counts = [len(list(group)) for _, group in groupby(cell_ids)]
        pred_groups = torch.split(preds, cell_counts)
        resp_groups = torch.split(responses, cell_counts)
        label_groups = torch.split(labels, cell_counts)

        if mode == 'VAL':
            loss_dicts_list = [criterion(p, l, a) for p, l, a in zip(pred_groups, label_groups, resp_groups)]
            combine_loss_list = torch.stack([d['total_loss'] for d in loss_dicts_list])
            weights = torch.tensor(cell_counts, dtype=torch.float32, device=config.device)
            weights = weights / weights.sum()
            b_total_loss = torch.sum(combine_loss_list * weights)
            total_loss += b_total_loss.item()

            if args.loss == 'Hierarchical':
                cls_loss_list = torch.stack([d['cls_loss'] for d in loss_dicts_list])
                rank_loss_list = torch.stack([d['rank_loss'] for d in loss_dicts_list])
                b_cls_loss = torch.sum(cls_loss_list * weights)
                b_rank_loss = torch.sum(rank_loss_list * weights)
                cls_loss += b_cls_loss.item()
                rank_loss += b_rank_loss.item()

        all_cell_ids.extend(cell_ids)
        all_drug_ids.extend(drug_ids.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_response.extend(responses.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_total_loss = total_loss / len(dataloader)
    avg_cls_loss = cls_loss / len(dataloader) if args.loss == 'Hierarchical' else 0.0
    avg_rank_loss = rank_loss / len(dataloader) if args.loss == 'Hierarchical' else 0.0

    # Compute metrics after evaluation
    metrics = compute_metrics(
        all_cell_ids,
        all_drug_ids,
        all_preds,
        all_response,
        all_labels,
        id2name,
        config,
        mode
    )
    log_metrics(metrics, mode, epoch, fold)

    return {'avg_total_loss': avg_total_loss, 'avg_cls_loss': avg_cls_loss, 'avg_rank_loss': avg_rank_loss}, metrics
