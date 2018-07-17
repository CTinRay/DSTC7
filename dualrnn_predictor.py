import math
import torch
from base_predictor import BasePredictor
from modules import DualRNN


class DualRNNPredictor(BasePredictor):
    """

    Args:
        dim_embed (int): Number of dimensions of word embeddings.
        dim_hidden (int): Number of dimensions of intermediate
            information embeddings.
    """

    def __init__(self, embeddings, dim_hidden,
                 dropout_rate=0.2, **kwargs):
        super(DualRNNPredictor, self).__init__(**kwargs)
        self.dim_hidden = dim_hidden
        self.model = DualRNN(embeddings.size(1), dim_hidden)
        self.embeddings = torch.nn.Embedding(embeddings.size(0),
                                             embeddings.size(1))
        self.embeddings.weight = torch.nn.Parameter(embeddings)

        # use cuda
        self.model = self.model.to(self.device)
        self.embeddings = self.embeddings.to(self.device)

        # make optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate)

    def _run_iter(self, batch, training):
        context = self.embeddings(batch['context'].to(self.device))
        options = self.embeddings(batch['options'].to(self.device))
        logits = self.model.forward(
            context.to(self.device),
            batch['context_lens'],
            options.to(self.device),
            batch['option_lens'])
        predicts = logits.max(-1)[1]
        probs = torch.nn.functional.softmax(logits, -1)
        loss = (-torch.log(probs) *
                batch['labels'].float().to(self.device)).sum()
        return predicts, loss

    def _predict_batch(self, batch, max_len=30):
        logits = self.model.forward(
            batch['context'].to(self.device),
            batch['context_lens'],
            batch['option'].to(self.device),
            batch['option_lens'])
        predicts = logits.max(-1)[1]
        return predicts
