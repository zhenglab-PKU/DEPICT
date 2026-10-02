import torch
import torch.nn as nn


def get_activation(activation_string):
    # activation function
    act = activation_string.lower()
    if act == "relu":
        return nn.ReLU()
    elif act == "leakyrelu":
        return nn.LeakyReLU()
    elif act == "prelu":
        return nn.PReLU()
    elif act == "gelu":
        return nn.GELU()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation: {activation_string}")


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim_list, output_dim=None, dropout=0.5, act_fn='relu'):
        super().__init__()
        # basic MLP for feature encoder

        layers = []
        # input layer
        layers.append(nn.Linear(input_dim, hidden_dim_list[0]))
        layers.append(nn.LayerNorm(hidden_dim_list[0]))
        if act_fn is not None:
            activation = get_activation(act_fn)
            layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # hidden layer
        for i in range(1, len(hidden_dim_list)):
            layers.append(nn.Linear(hidden_dim_list[i - 1], hidden_dim_list[i]))
            layers.append(nn.LayerNorm(hidden_dim_list[i]))
            if act_fn is not None:
                activation = get_activation(act_fn)
                layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # optional output layer
        if output_dim is not None:
            layers.append(nn.Linear(hidden_dim_list[-1], output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FeatEncoder(nn.Module):
    def __init__(self, dim_m=None, dim_e=None, hidden_dim_list=None, output_dim=128,
                 dropout=0.5, act_fn='relu'):
        super().__init__()
        # feature encoder for mutation or expression or both

        self.has_m = dim_m is not None
        self.has_e = dim_e is not None

        if self.has_m:
            self.m_encoder = MLP(
                input_dim=dim_m,
                hidden_dim_list=hidden_dim_list,
                output_dim=output_dim,
                dropout=dropout,
                act_fn=act_fn
            )

        if self.has_e:
            self.e_encoder = MLP(
                input_dim=dim_e,
                hidden_dim_list=hidden_dim_list,
                output_dim=output_dim,
                dropout=dropout,
                act_fn=act_fn
            )

        if self.has_m and self.has_e:
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * output_dim, output_dim),
                get_activation(act_fn),
                nn.Dropout(dropout),
                nn.Linear(output_dim, output_dim),
                nn.Sigmoid()
            )

    def forward(self, m=None, e=None):

        # only mutation
        if m is not None and e is None:
            return self.m_encoder(m)

        # only expression
        elif e is not None and m is None:
            return self.e_encoder(e)

        # mutation expression gate
        elif m is not None and e is not None:
            m_proj = self.m_encoder(m)
            e_proj = self.e_encoder(e)
            residual = m_proj + e_proj
            gate = self.gate_mlp(torch.cat([m_proj, e_proj], dim=1))
            fusion = gate * m_proj + (1 - gate) * e_proj + 0.1 * residual
            return fusion

        else:
            raise ValueError("eatEncoder.forward() requires at least one input: either mutation or expression.")


class ListRank(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ListRank model

        self.drug_encoder = nn.Embedding(
            config.num_drugs,
            config.output_dim
        )

        self.cell_encoder = FeatEncoder(
            dim_m=config.mut_dim,
            dim_e=config.exp_dim,
            hidden_dim_list=config.hidden_dim_list,
            output_dim=config.output_dim,
            dropout=config.dropout,
            act_fn=config.act_fn
        )
        self.drug_bias = nn.Parameter(torch.full((config.num_drugs,), 0.5))

        self.basic_score = MLP(
            input_dim=2 * config.output_dim + config.interaction_dim,
            hidden_dim_list=[128, 64],
            output_dim=1,
            dropout=config.dropout,
            act_fn=config.act_fn
        )

        self.bias_score = MLP(
            input_dim=2 * config.output_dim + config.interaction_dim,
            hidden_dim_list=[128, 64],
            output_dim=1,
            dropout=config.dropout,
            act_fn=config.act_fn
        )

        self.cell_proj = nn.Linear(
            config.output_dim,
            config.interaction_dim,
            bias=False
        )

        self.drug_proj = nn.Linear(
            config.output_dim,
            config.interaction_dim,
            bias=False
        )

    def adjust_bias(self, score, bias_score, target, drug_id, w_bias=1.0):

        pos_mask = score >= 0
        neg_mask = score < 0

        adjusted = torch.zeros_like(bias_score)
        bias = self.drug_bias[drug_id]

        adjusted[pos_mask] = bias_score[pos_mask]

        adjusted[neg_mask] = (torch.sigmoid(score[neg_mask]) + bias[neg_mask])

        adjusted = adjusted * target * w_bias

        return adjusted

    def forward(self, drug_id=None, mut_feat=None, exp_feat=None, target=None, return_emb=None):

        drug_emb = self.drug_encoder(drug_id)

        cell_emb = self.cell_encoder(m=mut_feat, e=exp_feat)

        interaction = self.cell_proj(cell_emb) * self.drug_proj(drug_emb)

        combined = torch.cat([cell_emb, drug_emb, interaction], dim=1)

        score = self.basic_score(combined).squeeze(-1)

        if target is not None:
            target_mask = target == 1
            cell_target = cell_emb[target_mask]
            drug_target = drug_emb[target_mask]
            interaction_target = interaction[target_mask]

            bias_combined = torch.cat([cell_target, drug_target, interaction_target], dim=1)
            bias_score = self.bias_score(bias_combined).squeeze(-1)
            #adjusted_bias = self.adjust_bias(score[target_mask], bias_score, target[target_mask], drug_id[target_mask])

            score[target_mask] = bias_score

        if return_emb:
            return score, cell_emb, drug_emb
        else:
            return score
