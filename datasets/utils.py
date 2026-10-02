import os

import numpy as np
import pandas as pd
import torch

from datasets.datasets import DataPoint


def get_response_data(root_dir, drug_response_file, drug_list_file):
    response_path = os.path.join(root_dir, 'data', drug_response_file)
    drug_response_df = pd.read_csv(str(response_path), header=0)

    drug_path = os.path.join(root_dir, 'data', drug_list_file)
    selected_drugs = pd.read_csv(str(drug_path), index_col=0, header=0).index.tolist()

    drug_response_df = drug_response_df[drug_response_df['DRUG_NAME'].isin(selected_drugs)]
    # auc convert 1-auc
    drug_response = {(row['SANGER_MODEL_ID'], row['EMB_ID']): 1 - row['AUC'] for _, row in drug_response_df.iterrows()}

    return drug_response


def get_drug_info(root_dir, drug_info_file):
    drug_info_path = os.path.join(root_dir, 'data', drug_info_file)
    drug_info_df = pd.read_csv(str(drug_info_path), header=0)
    # mapping dict
    id_to_name = dict(zip(drug_info_df['EMB_ID'], drug_info_df['DRUG_name']))
    name_to_id = dict(zip(drug_info_df['DRUG_name'], drug_info_df['EMB_ID']))
    target_dict = {
        row.EMB_ID: [t.strip() for t in row.TARGET.split(',')]
        for _, row in drug_info_df.iterrows()
        if pd.notna(row.TARGET)
    }

    # resistant_dict
    resistant_dict = {
        row.EMB_ID: [r.strip() for r in row.RESISTANT.split(',')]
        for _, row in drug_info_df.iterrows()
        if pd.notna(row.RESISTANT)
    }

    return id_to_name, name_to_id, target_dict, resistant_dict


def get_exp_data(root_dir, exp_file, gene_list_file):
    exp_path = os.path.join(root_dir, 'data', exp_file)
    exp_df = pd.read_csv(str(exp_path), index_col=0, header=0)

    gene_path = os.path.join(root_dir, 'data', gene_list_file)
    selected_genes = pd.read_csv(str(gene_path), index_col=0, header=0).index.tolist()

    missing_genes = list(set(selected_genes) - set(exp_df.index))

    exp_df = exp_df.reindex(selected_genes).fillna(0)

    exp_dict = {
        sample: torch.as_tensor(exp_df[sample].to_numpy(dtype=np.float32))
        for sample in exp_df.columns
    }

    return exp_dict, missing_genes, len(selected_genes)


def get_mut_data(root_dir, mut_file, gene_list_file, target_dict=None, resistant_dict=None):
    mut_long_path = os.path.join(root_dir, 'data', mut_file)
    mut_long_df = pd.read_csv(str(mut_long_path),header=0)
    mut_long_df.loc[:, 'value'] = 1
    mut_df = pd.crosstab(index=mut_long_df['gene_symbol'],
                          columns=mut_long_df['Sample'],
                          values=mut_long_df['value'],
                          aggfunc='mean',
                          dropna=False).fillna(0).astype(int)


    gene_path = os.path.join(root_dir, 'data', gene_list_file)
    selected_genes = pd.read_csv(str(gene_path), index_col=0, header=0).index.tolist()

    missing_genes = list(set(selected_genes) - set(mut_df.index.unique()))

    mut_df = mut_df.reindex(selected_genes).fillna(0)

    mut_dict = {
        sample: torch.as_tensor(mut_df[sample].to_numpy(dtype=np.float32))
        for sample in mut_df.columns
    }

    mut_long_df["mutation"] = mut_long_df["gene_symbol"] + "_" + mut_long_df["protein_change"]
    mut2sample = mut_long_df.groupby('mutation')['Sample'].apply(set)

    samples = mut_long_df["Sample"].unique()
    drugs = list(target_dict.keys())

    target_df = pd.DataFrame(0, index=samples, columns=drugs, dtype=int)
    for drug in drugs:
        hit_samples = set()

        muts = target_dict.get(drug, [])
        for mut in muts:
            hit_samples |= mut2sample.get(mut, set())

        if resistant_dict is not None:
            res_muts = resistant_dict.get(drug, [])
            for rmut in res_muts:
                hit_samples -= mut2sample.get(rmut, set())

        if hit_samples:
            target_df.loc[list(hit_samples), drug] = 1
    return mut_dict, missing_genes, len(selected_genes), target_df


def pad_samples(mut_dict, target_df, exp_keys):

    mut_dict_padded = {}
    gene_len = len(next(iter(mut_dict.values())))

    for sample in exp_keys:
        if sample in mut_dict:
            mut_dict_padded[sample] = mut_dict[sample]
        else:
            mut_dict_padded[sample] = torch.zeros(gene_len, dtype=torch.float32)

    target_df_padded = target_df.reindex(index=exp_keys, fill_value=0)

    return mut_dict_padded, target_df_padded


def get_split_data(config, fold):
    split_ids = {}
    for split in ['train', 'val', 'test']:

        cell_file = os.path.join(config.root_dir, 'data/splits', f"fold_{fold}", f"{split}_id.csv")
        cells = pd.read_csv(cell_file)['model_id'].tolist()
        split_ids[split] = cells

    return split_ids


def split_response_data(drug_response, train_idx, val_idx, test_idx):
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)

    # train_set = set(random.sample(list(train_set), int(0.9 * len(train_set))))

    def extract(cell_set):
        data = []
        for (cell, drug), response in drug_response.items():
            if cell in cell_set:
                data.append((cell, drug, response))
        return data

    return extract(train_set), extract(val_set), extract(test_set)


def create_datapoints(resp_list, target):
    points = []
    for s in resp_list:  # s: (cell_id, drug_id, response)
        dp = DataPoint(*s)

        if target is not None:
            if dp.cell_id in target.index and dp.drug_id in target.columns:
                dp.target = target.loc[dp.cell_id, dp.drug_id]
            else:
                dp.target = 0.0
        else:
            dp.target = 0.0
        points.append(dp)
    return points


def get_feats(cell_ids, feats_dict):
    feats = [feats_dict[cell_id] for cell_id in cell_ids]
    return torch.stack(feats, dim=0)
