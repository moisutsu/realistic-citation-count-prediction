from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertAttention


class PoolerForMultipleOutput(nn.Module):
    def __init__(
        self,
        pooling_method_for_multiple_output: str,
        pooling_method_for_single_output: str,
        max_embeddings_count: Optional[int] = None,
        hidden_size: Optional[int] = None,
        weighting_per_embedding: bool = False,
        bert: Optional[BertModel] = None,
        use_layer_numbers: Optional[list[int]] = None,
        without_special_tokens: bool = True,
    ):
        super().__init__()

        # validate
        assert pooling_method_for_multiple_output in [
            "mean",
            "self_attention_w_position_embeddings-first",
            "self_attention_w_position_embeddings-mean",
            "transformer",
            "concat",
            "attention",
            "bert",
        ]

        self.pooling_method_for_multiple_output = pooling_method_for_multiple_output
        self.pooling_method_for_single_output = pooling_method_for_single_output
        self.max_embeddings_count = max_embeddings_count
        self.hidden_size = hidden_size
        self.weighting_per_embedding = weighting_per_embedding
        self.without_special_tokens = without_special_tokens

        self.pooler_for_single_output = Pooler(
            pooling_method_for_single_output,
            without_special_tokens=self.without_special_tokens,
        )

        if (
            "self_attention_w_position_embeddings"
            in self.pooling_method_for_multiple_output
        ):
            config = BertConfig(hidden_size=self.hidden_size)

            self.position_embeddings = nn.Embedding(
                self.max_embeddings_count, self.hidden_size
            )
            self.bert_attention = BertAttention(config=config)

            self.register_buffer(
                "position_ids", torch.arange(self.max_embeddings_count)
            )

        elif self.pooling_method_for_multiple_output == "attention":
            self.attention_vector = nn.Linear(self.hidden_size, 1)

        elif self.pooling_method_for_multiple_output == "bert":
            if use_layer_numbers is None:
                use_layer_numbers = list(range(bert.config.num_hidden_layers))

            bert_layers = nn.ModuleList(
                [bert.encoder.layer[layer_number] for layer_number in use_layer_numbers]
            )

            bert_encoder_config = bert.config
            bert_encoder_config.num_hidden_layers = len(use_layer_numbers)
            self.bert_encoder = type(bert.encoder)(bert_encoder_config)
            self.bert_encoder.layer = bert_layers

        if self.weighting_per_embedding:
            self.weight_per_embedding = WeightPerEmbedding(self.max_embeddings_count)

    # TODO: Support mask
    def forward(self, xs: list[Tensor]) -> Tensor:
        pooled_xs = []
        for x in xs:
            pooled_xs.append(self.pooler_for_single_output(x))

        pooled_xs = torch.cat(pooled_xs)

        if self.weighting_per_embedding:
            pooled_xs = (pooled_xs.T * self.weight_per_embedding(pooled_xs.shape[0])).T

        if self.pooling_method_for_multiple_output == "mean":
            pooled_output = pooled_xs.mean(dim=0)

        elif (
            "self_attention_w_position_embeddings"
            in self.pooling_method_for_multiple_output
        ):
            position_ids = self.position_ids[: pooled_xs.shape[0]]
            position_embeddings = self.position_embeddings(position_ids)

            pooled_xs += position_embeddings

            # use embedding of first page for pooled output
            if (
                self.pooling_method_for_multiple_output
                == "self_attention_w_position_embeddings-first"
            ):
                pooled_output = self.bert_attention(pooled_xs.unsqueeze(dim=0))[0][0][0]

            elif (
                self.pooling_method_for_multiple_output
                == "self_attention_w_position_embeddings-mean"
            ):
                pooled_output = (
                    self.bert_attention(pooled_xs.unsqueeze(dim=0))[0]
                    .mean(dim=1)
                    .squeeze()
                )

            else:
                raise ValueError(
                    f"No such self-attention pooling: {self.pooling_method_for_multiple_output}"
                )

        elif self.pooling_method_for_multiple_output == "transformer":
            raise NotImplementedError

        elif self.pooling_method_for_multiple_output == "concat":
            pooled_xs = F.pad(
                pooled_xs, (0, 0, 0, self.max_embeddings_count - pooled_xs.shape[0])
            )
            pooled_output = torch.flatten(pooled_xs)

        elif self.pooling_method_for_multiple_output == "attention":
            attention_score = F.softmax(self.attention_vector(pooled_xs), dim=0)
            pooled_output = (pooled_xs * attention_score).sum(dim=0)

        elif self.pooling_method_for_multiple_output == "bert":
            bert_output = self.bert_encoder(pooled_xs.unsqueeze(0)).last_hidden_state
            pooled_output = bert_output.mean(dim=1).squeeze()

        return pooled_output


class Pooler(nn.Module):
    def __init__(self, pooling_method: str, without_special_tokens: bool = True):
        super().__init__()

        # validate
        assert pooling_method in ["cls", "mean", "max", "none"]

        self.pooling_method = pooling_method
        self.without_special_tokens = without_special_tokens

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor = None,
    ) -> Tensor:

        pooled_output: Tensor

        if self.pooling_method == "none":
            return x

        if self.pooling_method == "cls":
            pooled_output = x[:, 0]
            return pooled_output

        if self.without_special_tokens:
            x = x[:, 1:-1]

            if not attention_mask is None:
                attention_mask = attention_mask[:, 1:-1]

        if self.pooling_method == "mean":
            if attention_mask is None:
                pooled_output = x.mean(dim=1)

            else:
                x[attention_mask.long() == 0, :] = 0
                acctual_length = attention_mask.sum(dim=1, keepdim=True)
                pooled_output = x.sum(dim=1) / acctual_length

        elif self.pooling_method == "max":
            x[attention_mask.long() == 0, :] = -1e9
            pooled_output = x.max(dim=1).values

        else:
            raise ValueError(f"No such pooling method: {self.pooling_method}")

        return pooled_output


class WeightPerEmbedding(nn.Module):
    def __init__(self, max_embeddings_count: int):
        super().__init__()

        self.max_embeddings_count = max_embeddings_count

        self.softmax = nn.Softmax(dim=0)
        self.weight = nn.Parameter(
            torch.nn.init.normal_(torch.empty(self.max_embeddings_count), mean=0, std=1)
        )

    def forward(self, embeddings_count: int) -> Tensor:
        return self.softmax(self.weight[:embeddings_count])
