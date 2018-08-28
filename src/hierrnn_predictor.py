import torch
from base_predictor import BasePredictor
from modules import HierRNN, NLLLoss, RankLoss


class HierRNNPredictor(BasePredictor):
    """

    Args:
        dim_embed (int): Number of dimensions of word embeddings.
        dim_hidden (int): Number of dimensions of intermediate
            information embeddings.
    """

    def __init__(self, embeddings, dim_hidden,
                 dropout_rate=0.2, loss='NLLLoss', margin=0, threshold=None,
                 similarity='inner_product', **kwargs):
        super(HierRNNPredictor, self).__init__(**kwargs)
        self.dim_hidden = dim_hidden
        self.model = HierRNN(embeddings.size(1), dim_hidden,
                             similarity=similarity)
        self.embeddings = torch.nn.Embedding(embeddings.size(0),
                                             embeddings.size(1))
        self.embeddings.weight = torch.nn.Parameter(embeddings)

        # use cuda
        self.model = self.model.to(self.device)
        self.embeddings = self.embeddings.to(self.device)

        # make optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate)

        self.loss = {
            'NLLLoss': NLLLoss(),
            'RankLoss': RankLoss(margin, threshold)
        }[loss]

    def _run_iter(self, batch, training):
        with torch.no_grad():
            context = self.embeddings(batch['context'].to(self.device))
            options = self.embeddings(batch['options'].to(self.device))
        logits = self.model.forward(
            context.to(self.device),
            batch['utterance_ends'],
            options.to(self.device),
            batch['option_lens'])
        loss = self.loss(logits, batch['labels'].to(self.device))
        return logits, loss

    def _predict_batch(self, batch):
        context = self.embeddings(batch['context'].to(self.device))
        options = self.embeddings(batch['options'].to(self.device))
        logits = self.model.forward(
            context.to(self.device),
            batch['utterance_ends'],
            options.to(self.device),
            batch['option_lens'])
        # predicts = logits.max(-1)[1]
        return logits

    def _encode_option_batch(self, batch):
        options = self.embeddings(batch['options'].to(self.device))
        encoded = self.model.option_encoder(options, batch['option_lens'])
        return encoded

    def _predict_batch_option_encoded(self, batch):
        context = self.embeddings(batch['context'].to(self.device))
        options_hidden = batch['options_hidden'].to(self.device)
        size = options_hidden.size()
        options_hidden = options_hidden.expand(len(context), size[0], size[1])
        logits = self.model.half_forward(
            context.to(self.device),
            batch['utterance_ends'],
            options_hidden)
        return logits
