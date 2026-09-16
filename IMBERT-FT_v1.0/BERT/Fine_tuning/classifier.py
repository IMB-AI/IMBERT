"""Classification heads placed on top of the fine-tuned IMBERT encoder.

Every head takes the last encoder layer and reduces it to a single logit; the
`output_type` entry of the configuration file selects which one is used:

    geneformer           -> IMBERTBinaryClassifier_GF       (mean over positions)
    encoder              -> IMBERTBinaryClassifier_CLS      (first position)
    encoderWithgeneformer-> IMBERTBinaryClassifier_CLSWithGF (both, concatenated)
    geneformerMLP        -> IMBERTBinaryClassifier_GFMLP    (mean + MLP)
    encoderMLP           -> IMBERTBinaryClassifier_CLSMLP   (first position + MLP)

The logits are raw scores: the training loop applies `BCEWithLogitsLoss`, so
call `torch.sigmoid` to turn them into probabilities.
"""

import torch
import torch.nn as nn


def _build_mlp(hidden_size, n_dense_layer, dense_act, dropout):
    """Stack of `Linear -> activation -> dropout` blocks; returns (layers, out_dim).

    `n_dense_layer` is the `n_layers` configuration entry: a list of hidden
    sizes, e.g. `n_layers = 128, 64`.  `n_layers = 0` (or an empty value) means
    no MLP, so the classifier sits directly on the pooled encoder output.
    """
    layers = []
    prev_dim = hidden_size
    for h_dense in n_dense_layer:
        if int(h_dense) <= 0:
            continue
        layers.append(nn.Linear(prev_dim, int(h_dense), bias=False))
        layers.append(nn.GELU() if dense_act == "GELU" else nn.ReLU())
        layers.append(nn.Dropout(dropout))
        prev_dim = int(h_dense)

    return layers, prev_dim


class IMBERTBinaryClassifier_GF(nn.Module):
    def __init__(self, config, imbert, logger):
        super().__init__()

        hidden_size = config["Model"]["hyperparams"].as_int("hidden_size")

        self.imbert = imbert
        self.clf = nn.Linear(in_features=hidden_size, out_features=1, bias=False)

    def forward(self, x, segment_label, attention_mask):
        embeddings_output, encoded_layers, _, _, _ = self.imbert(x, segment_label, attention_mask)
        last_encoded_layer = encoded_layers[-1]

        cls_token = last_encoded_layer.mean(dim=1)

        pred_scores = self.clf(cls_token)

        return last_encoded_layer, pred_scores


class IMBERTBinaryClassifier_CLS(nn.Module):
    def __init__(self, config, imbert, logger):
        super().__init__()

        hidden_size = config["Model"]["hyperparams"].as_int("hidden_size")

        self.imbert = imbert
        self.clf = nn.Linear(in_features=hidden_size, out_features=1, bias=False)

    def forward(self, x, segment_label, attention_mask):
        embeddings_output, encoded_layers, _, _, _ = self.imbert(x, segment_label, attention_mask)
        last_encoded_layer = encoded_layers[-1]
        
        cls_token = last_encoded_layer[:, 0, :]

        pred_scores = self.clf(cls_token)

        return last_encoded_layer, pred_scores

    

class IMBERTBinaryClassifier_CLSWithGF(nn.Module):
    def __init__(self, config, imbert, logger):
        super().__init__()

        hidden_size = config["Model"]["hyperparams"].as_int("hidden_size")

        self.imbert = imbert
        self.clf = nn.Linear(in_features=hidden_size*2, out_features=1, bias=False)

    def forward(self, x, segment_label, attention_mask):
        embeddings_output, encoded_layers, _, _, _ = self.imbert(x, segment_label, attention_mask)
        last_encoded_layer = encoded_layers[-1]

        cls_token_ = last_encoded_layer[:, 0, :]
        gf_token = last_encoded_layer[:, 1:, :].mean(dim=1)
        
        cls_token = torch.cat((cls_token_, gf_token), dim=1)
        pred_scores = self.clf(cls_token)

        return last_encoded_layer, pred_scores


class IMBERTBinaryClassifier_GFMLP(nn.Module):
    def __init__(self, config, imbert, logger):
        super().__init__()

        hidden_size = config["Model"]["hyperparams"].as_int("hidden_size")
        n_dense_layer = config["FineTuning"]['hyperparams'].as_list('n_layers')
        dropout = config["FineTuning"]["hyperparams"].as_float("cls_dropout_prob")
        dense_act = config["FineTuning"]["hyperparams"]["dense_act"]

        layers, prev_dim = _build_mlp(hidden_size, n_dense_layer, dense_act, dropout)

        self.imbert = imbert
        self.mlp = nn.Sequential(*layers)
        self.clf = nn.Linear(in_features=prev_dim, out_features=1, bias=False)

    def forward(self, x, segment_label, attention_mask):
        embeddings_output, encoded_layers, _, _, _ = self.imbert(x, segment_label, attention_mask)
        last_encoded_layer = encoded_layers[-1]

        sample_embeddings = last_encoded_layer.mean(dim=1)
        cls_token = self.mlp(sample_embeddings)

        pred_scores = self.clf(cls_token)

        return last_encoded_layer, pred_scores


class IMBERTBinaryClassifier_CLSMLP(nn.Module):
    def __init__(self, config, imbert, logger):
        super().__init__()

        hidden_size = config["Model"]["hyperparams"].as_int("hidden_size")
        n_dense_layer = config["FineTuning"]['hyperparams'].as_list('n_layers')
        dropout = config["FineTuning"]["hyperparams"].as_float("cls_dropout_prob")
        dense_act = config["FineTuning"]["hyperparams"]["dense_act"]

        layers, prev_dim = _build_mlp(hidden_size, n_dense_layer, dense_act, dropout)

        self.imbert = imbert
        self.mlp = nn.Sequential(*layers)
        self.clf = nn.Linear(in_features=prev_dim, out_features=1, bias=False)

    def forward(self, x, segment_label, attention_mask):
        embeddings_output, encoded_layers, _, _, _ = self.imbert(x, segment_label, attention_mask)
        last_encoded_layer = encoded_layers[-1]

        sample_embeddings = last_encoded_layer[:, 0, :]
        cls_token = self.mlp(sample_embeddings)

        pred_scores = self.clf(cls_token)

        return last_encoded_layer, pred_scores

