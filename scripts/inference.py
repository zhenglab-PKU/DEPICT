import logging
import os

import pandas as pd
from tqdm import tqdm

from datasets.datasets import *
from datasets.utils import get_exp_data, get_mut_data, get_drug_info,pad_samples
from models.ListRank import ListRank
from models.loss import *

logger = logging.getLogger(__name__)


def normalize_scores_by_drug(df):
    df = df.copy()
    df['Score_norm'] = np.nan
    for drug_name, sub_df in df.groupby('Drug_name'):
        scores = sub_df['Score'].values
        min_val, max_val = np.min(scores), np.max(scores)
        if max_val == min_val:
            norm_scores = np.full_like(scores, 0.5, dtype=float)
        else:
            norm_scores = (scores - min_val) / (max_val - min_val)
        df.loc[sub_df.index, 'Score_norm'] = norm_scores
    return df


def inference(args, config, timestamp):
    logger.info(f'Start constructing datasets for inference')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.update('device', device)

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
    sample_ids = []
    if args.input in ['all', 'exp']:
        expression, exp_missing_genes, num_exp_gene = get_exp_data(
            config.root_dir,
            config.exp_file,
            config.exp_select_file
        )
        config.update('exp_dim', num_exp_gene)
        sample_ids = list(expression.keys())
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
        sample_ids = list(expression.keys())
        if args.input == "all":
            mutation, target = pad_samples(mutation, target, list(expression.keys()))

        if mut_missing_genes:
            logger.info(f"{len(mut_missing_genes)} genes not found in mutation data: {', '.join(mut_missing_genes)}")

    drug_path = os.path.join(config.root_dir, 'data', config.inference_drug_file)
    inference_drugs = pd.read_csv(str(drug_path), index_col=0, header=0).index.tolist()

    drug_ids = torch.tensor([name_to_id[name] for name in inference_drugs], dtype=torch.long).to(device)
    drug_ids_list = drug_ids.cpu().tolist()

    model = ListRank(config).to(device)
    model.load_state_dict(torch.load(config.model_path, weights_only=True))
    model.eval()

    results = []
    cell_embeddings = []
    drug_embeddings = {}

    logger.info(f'Start inference on {len(sample_ids)} samples × {len(drug_ids_list)} drugs')
    with torch.no_grad():
        for sample_id in tqdm(sample_ids, desc='Inference'):
            exp_feats = None
            mut_feats = None
            if args.input in ['all', 'mut']:
                mut_feats = mutation[sample_id].to(device)
                mut_feats = mut_feats.expand(len(drug_ids_list), -1)
            if args.input in ['all', 'exp']:
                exp_feats = expression[sample_id].to(device)
                exp_feats = exp_feats.expand(len(drug_ids_list), -1)

            if target is not None and sample_id in target.index:
                targets = []
                for drug_id in drug_ids_list:
                    if target is not None:
                        if drug_id in target.columns:
                            val = target.loc[sample_id, drug_id]
                        else:
                            val = 0.0
                    else:
                        val = 0.0
                    targets.append(val)
                targets = torch.tensor(targets, dtype=torch.float32, device=device)
            else:
                targets = torch.zeros(len(drug_ids_list), dtype=torch.float32, device=device)

            scores, cell_emb, drug_emb = model(
                drug_id=drug_ids,
                mut_feat=mut_feats,
                exp_feat=exp_feats,
                target=targets,
                return_emb=True
            )
            scores_list = scores.cpu().numpy().tolist()
            cell_emb_np = cell_emb.cpu().numpy()
            drug_emb_np = drug_emb.cpu().numpy()

            cell_embeddings.append({
                "sample_id": sample_id,
                "embedding": cell_emb_np[0]
            })

            for drug_id, drug_vec in enumerate(drug_emb_np):
                global_drug_id = drug_ids_list[drug_id]
                drug_name = id_to_name[global_drug_id]
                if drug_name not in drug_embeddings:
                    drug_embeddings[drug_name] = drug_vec

            for drug_id, score in enumerate(scores_list):
                global_drug_id = drug_ids_list[drug_id]
                drug_name = id_to_name[global_drug_id]
                results.append({
                    "Samples_id": sample_id,
                    "Drug_id": global_drug_id,
                    "Drug_name": drug_name,
                    "Score": score
                })

    results_path = os.path.join(config.root_dir, 'results', f'{timestamp}_inference_results.csv')
    results_df = pd.DataFrame(results)
    if args.normalize:
        results_df = normalize_scores_by_drug(results_df)
    results_df.to_csv(results_path, index=False)
    logger.info(f'Saved inference results → {results_path}')

    sample_ids = [x['sample_id'] for x in cell_embeddings]
    cell_matrix = np.stack([x['embedding'] for x in cell_embeddings])
    cell_emb_cols = [f'cell_emb_{i}' for i in range(cell_matrix.shape[1])]
    cell_df = pd.DataFrame(cell_matrix, columns=cell_emb_cols)
    cell_df.insert(0, 'sample_id', sample_ids)
    cell_df.to_csv(os.path.join(config.root_dir, 'results', f'{timestamp}_embeddings.csv'), index=False)

    drug_names = list(drug_embeddings.keys())
    drug_matrix = np.stack(list(drug_embeddings.values()))
    drug_emb_cols = [f'drug_emb_{i}' for i in range(drug_matrix.shape[1])]
    drug_df = pd.DataFrame(drug_matrix, columns=drug_emb_cols)
    drug_df.insert(0, 'drug_name', drug_names)
    drug_df.to_csv(os.path.join(config.root_dir, 'results', f'{timestamp}_drug_embeddings.csv'), index=False)
