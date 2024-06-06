import torch
import torch.nn as nn
from transformers import (
    BertModel, BertPreTrainedModel,
    RobertaModel, RobertaPreTrainedModel
)


def extract_entity(sequence_output, e_mask):
    extended_e_mask = e_mask.unsqueeze(-1)
    extended_e_mask = extended_e_mask.float() * sequence_output
    extended_e_mask, _ = extended_e_mask.max(dim=-2)
    return extended_e_mask.float()


class REBertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.relation_emb_dim = config.hidden_size
        self.bert = BertModel(config)
        self.fclayer = nn.Linear(self.relation_emb_dim * 2, self.relation_emb_dim)
        self.classifier = nn.Sequential(nn.Linear(self.relation_emb_dim, self.relation_emb_dim),
                                        nn.Tanh(),
                                        nn.Linear(self.relation_emb_dim, self.num_labels)
                                        )
        self.ce = nn.CrossEntropyLoss()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            e1_mask=None,
            e2_mask=None,
            target=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs[0]  # Sequence of hidden-states of the last layer
        e1_mark = extract_entity(sequence_output, e1_mask)
        e2_mark = extract_entity(sequence_output, e2_mask)
        pooled_output = torch.cat([e1_mark, e2_mark], dim=-1)
        sentence_embeddings = self.fclayer(pooled_output)
        # sentence_embeddings = outputs[1]
        sentence_embeddings = torch.tanh(sentence_embeddings)
        pred = self.classifier(sentence_embeddings)
        if target is not None:
            loss = self.ce(pred, target)
        else:
            loss = pred
        return loss
