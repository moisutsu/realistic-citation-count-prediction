from typing import Optional

import torch
from torch import Tensor, nn
from transformers import AutoModel, BertModel

from .pooler import Pooler


class BertForPaperScoreWithRegression(nn.Module):
    def __init__(
        self,
        pooling_method: str = "cls",
        pooling_without_special_tokens: bool = True,
        bert_model: str = "bert-base-uncased",
        classifier_dropout: float = 0.1,
        use_published_date: bool = False,
    ):
        super().__init__()

        self.pooling_method = pooling_method
        self.pooling_without_special_tokens = pooling_without_special_tokens
        self.bert_model = bert_model
        self.classifier_dropout = classifier_dropout
        self.use_published_date = use_published_date

        self.bert: BertModel = AutoModel.from_pretrained(self.bert_model)
        self.pooler = Pooler(self.pooling_method, self.pooling_without_special_tokens)
        self.dropout = nn.Dropout(classifier_dropout)
        if use_published_date:
            self.pooled_output2scalars = nn.Linear(self.bert.config.hidden_size + 1, 1)
        else:
            self.pooled_output2scalars = nn.Linear(self.bert.config.hidden_size, 1)
        self.criterion = nn.MSELoss()

    def fit(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        target: Tensor,
        published_dates: Optional[Tensor] = None,
    ) -> Tensor:
        scalars = self(
            input_ids, attention_mask, token_type_ids, published_dates
        ).flatten()
        loss = self.criterion(scalars, target)
        return loss

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        published_dates: Optional[Tensor] = None,
    ) -> Tensor:
        # Processing for models without token_type_ids such as Longformer
        if token_type_ids[0][0].item() == -1:
            token_type_ids = None

        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state
        pooled_output = self.pooler(output, attention_mask)
        pooled_output = self.dropout(pooled_output)

        if self.use_published_date:
            published_dates = published_dates.view(-1, 1)
            pooled_output = torch.cat((pooled_output, published_dates), axis=1)

        scalars = self.pooled_output2scalars(pooled_output)

        return scalars

    def resize_token_embeddings(self, new_num_tokens: int):
        self.bert.resize_token_embeddings(new_num_tokens)
