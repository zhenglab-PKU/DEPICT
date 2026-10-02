import csv
import logging
import os

from torch.utils.data import DataLoader

from datasets.datasets import *
from datasets.utils import get_response_data, get_drug_info, get_exp_data, get_mut_data, get_split_data, \
    split_response_data, create_datapoints, pad_samples
from models.ListRank import ListRank
from models.loss import *
from scripts.evaluation import evaluate
from scripts.train_step import train_step
from utils.utils import compute_gnorm, compute_pnorm

logger = logging.getLogger(__name__)


def train_fold(args, config, fold):
    logger.info(f'Start constructing datasets for fold_{fold}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.update('device', device)

    response = get_response_data(
        config.root_dir,
        config.drug_response_file,
        config.drug_select_file
    )

    id_to_name, name_to_id, target_dict, resistant_dict = get_drug_info(
        config.root_dir,
        config.drug_info_file
    )

    config.update('num_drugs', len(id_to_name))

    # loading omic data
    expression = {}
    mutation = {}
    target = None
    config.update('exp_dim', None)
    config.update('mut_dim', None)

    if args.input in ['all', 'exp']:
        expression, exp_missing_genes, num_exp_gene = get_exp_data(
            config.root_dir,
            config.exp_file,
            config.exp_select_file
        )

        config.update('exp_dim', num_exp_gene)

        if exp_missing_genes:
            logger.info(f"{len(exp_missing_genes)} genes not found in expression data: {', '.join(exp_missing_genes)}")

    if args.input in ['all', 'mut']:
        mutation, mut_missing_genes, num_mut_genes, target = get_mut_data(
            config.root_dir,
            config.mut_file,
            config.mut_select_file,
            target_dict,
            resistant_dict
        )
        config.update('mut_dim', num_mut_genes)

        if mut_missing_genes:
            logger.info(f"{len(mut_missing_genes)} genes not found in mutation data: {', '.join(mut_missing_genes)}")

        if args.input == "all":
            mutation, target = pad_samples(mutation, target, list(expression.keys()))

    # loading drug response data and split

    logger.info(f'num_expression: {len(expression)} | '
                f'num_mutation: {len(mutation)} | '
                f'num_drugs: {len(id_to_name)}')

    split_ids = get_split_data(config, fold)

    train_resp, val_resp, test_resp = split_response_data(
        response,
        split_ids['train'],
        split_ids['val'],
        split_ids['test']
    )

    logger.info(f'Number of train responses: {len(train_resp)} | '
                f'Number of val responses: {len(val_resp)} | '
                f'Number of test responses: {len(test_resp)}')

    train_points = create_datapoints(train_resp, target)
    val_points = create_datapoints(val_resp, target)
    test_points = create_datapoints(test_resp, target)

    train_dataset = TrainDatasets(train_points, config)
    train_eval_dataset = TestDatasets(train_points, config)
    val_dataset = TestDatasets(val_points, config)
    test_dataset = TestDatasets(test_points, config)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.batch_size,
        collate_fn=train_dataset.collate_fn
    )
    train_eval_dataloader = DataLoader(
        train_eval_dataset,
        shuffle=False,
        batch_size=config.batch_size,
        collate_fn=train_eval_dataset.collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=config.batch_size,
        collate_fn=val_dataset.collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.batch_size,
        collate_fn=test_dataset.collate_fn
    )

    # init model
    model = ListRank(config).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    if args.loss == 'Hierarchical':
        criterion = HierarchicalLoss(alpha=config.alpha, margin=config.margin)
    elif args.loss == 'MSE':
        criterion = MSELoss()
    elif args.loss == 'BCE':
        criterion = BCEWithLogitsLoss()
    elif args.loss == 'Listwise':
        criterion = ListwiseLoss()
    elif args.loss == 'Pairwise':
        criterion = PairwiseLoss()
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    logger.info(f'Start training fold_{fold}')

    loss_list = []
    val_metrics_list = []
    train_metrics_list = []
    gnorm=0
    pnorm=0
    # early stop initialize
    best_val_metric = float('inf')
    patience = config.early_stop_patience
    patience_counter = 0
    best_epoch = 0
    save_path = os.path.join(config.fold_dir, f'best_model.pt')

    # train loop
    for epoch in range(1, config.num_epochs + 1):
        # train
        total_loss, cls_loss, rank_loss = 0, 0, 0
        model.train()
        for i, batch in enumerate(train_dataloader, 1):
            optimizer.zero_grad()
            b_total_loss, b_cls_loss, b_rank_loss = train_step(
                args=args,
                config=config,
                batch=batch,
                model=model,
                criterion=criterion,
                expression=expression,
                mutation=mutation
            )

            b_total_loss.backward()

            total_loss += b_total_loss.item()
            cls_loss += b_cls_loss.item()
            rank_loss += b_rank_loss.item()

            if i == len(train_dataloader):
                gnorm = compute_gnorm(model)
                pnorm = compute_pnorm(model)

                logger.info(
                    f'[Epoch {epoch} FinalBatch {i}] | '
                    f'GNorm: {gnorm:.4f} | PNorm: {pnorm:.4f}'
                )

            optimizer.step()

        train_avg_total_loss = total_loss / len(train_dataloader)
        train_avg_cls_loss = cls_loss / len(train_dataloader)
        train_avg_rank_loss = rank_loss / len(train_dataloader)

        # eval
        model.eval()
        with torch.no_grad():
            _, train_metrics = evaluate(
                args=args,
                config=config,
                fold=fold,
                epoch=epoch,
                model=model,
                criterion=criterion,
                dataloader=train_eval_dataloader,
                id2name=id_to_name,
                expression=expression,
                mutation=mutation,
                mode='TRAIN'
            )
            val_loss_dict, val_metrics = evaluate(
                args=args,
                config=config,
                fold=fold,
                epoch=epoch,
                model=model,
                criterion=criterion,
                dataloader=val_dataloader,
                id2name=id_to_name,
                expression=expression,
                mutation=mutation,
                mode='VAL'
            )

        loss_list.append({
            'epoch': epoch,
            'train_total_loss': train_avg_total_loss,
            'train_cls_loss': train_avg_cls_loss,
            'train_rank_loss': train_avg_rank_loss,
            'val_total_loss': val_loss_dict['avg_total_loss'],
            'val_cls_loss': val_loss_dict['avg_cls_loss'],
            'val_rank_loss': val_loss_dict['avg_rank_loss'],
            'gnorm': gnorm,
            'pnorm': pnorm
        })

        val_metrics_list.append(val_metrics)
        train_metrics_list.append(train_metrics)

        logger.info(
            f"Epoch {epoch}: Train Loss: {train_avg_total_loss:.4f} | Val Loss: {val_loss_dict['avg_total_loss']:.4f}")

        # early stop
        current_val_metric = val_loss_dict['avg_total_loss']

        scheduler.step(current_val_metric)

        # Avoid early stop too early; only enable after 30 epochs
        if epoch >= config.save_epoch_strat:
            if current_val_metric < best_val_metric - 0.001:
                best_val_metric = current_val_metric
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
                logger.info(f"New best model saved at epoch {epoch}.")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
                break

    # record loss
    loss_path = os.path.join(config.fold_dir, 'train_val_loss.csv')
    with open(loss_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=loss_list[0].keys())
        writer.writeheader()
        writer.writerows(loss_list)

    # record val metrics
    val_metrics_path = os.path.join(config.fold_dir, 'val_metrics.csv')
    for i, d in enumerate(val_metrics_list):
        d['Epoch'] = i + 1
    fieldnames = ['Epoch'] + [k for k in val_metrics_list[0].keys() if k != 'Epoch']
    with open(val_metrics_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(val_metrics_list)

    # record train metrics
    train_metrics_path = os.path.join(config.fold_dir, 'train_metrics.csv')
    for i, d in enumerate(train_metrics_list):
        d['Epoch'] = i + 1
    fieldnames = ['Epoch'] + [k for k in train_metrics_list[0].keys() if k != 'Epoch']
    with open(train_metrics_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_metrics_list)

    # test
    if args.fold != 6:
        logger.info(f'load best model for test')
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            _, test_metrics = evaluate(
                args=args,
                config=config,
                fold=fold,
                model=model,
                criterion=criterion,
                dataloader=test_dataloader,
                id2name=id_to_name,
                expression=expression,
                mutation=mutation,
                mode='TEST'
            )

            test_metrics_path = os.path.join(config.model_save_dir, 'test_metrics.csv')
            with open(test_metrics_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Average'])
                for metric_name, avg_value in test_metrics.items():
                    writer.writerow([metric_name, avg_value])
