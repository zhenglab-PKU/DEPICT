from itertools import groupby

import torch

from datasets.utils import get_feats


def train_step(args, config, batch, model, criterion, expression=None, mutation=None):
    # data extraction for training
    cell_ids = [s.cell_id for s in batch]  # list cell_id: ["SIDM00631",...] [B=batch_size*(neg+pos)¡Ö32*(40+40)]

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

    loss_dicts_list = [criterion(p, l, r) for p, l, r in zip(pred_groups, label_groups, resp_groups)]

    combine_loss_list = torch.stack([d['total_loss'] for d in loss_dicts_list])
    weights = torch.tensor(cell_counts, dtype=torch.float32, device=config.device)
    weights = weights / weights.sum()
    b_total_loss = torch.sum(combine_loss_list * weights)

    if args.loss == 'Hierarchical':
        cls_loss_list = torch.stack([d['cls_loss'] for d in loss_dicts_list])
        rank_loss_list = torch.stack([d['rank_loss'] for d in loss_dicts_list])
        b_cls_loss = torch.sum(cls_loss_list * weights)
        b_rank_loss = torch.sum(rank_loss_list * weights)
        return b_total_loss, b_cls_loss, b_rank_loss
    else:
        return b_total_loss, torch.tensor(0.0), torch.tensor(0.0)
