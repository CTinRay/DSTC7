import torch
from base_predictor import BasePredictor
from modules import HierRNN, UttHierRNN, UttBinHierRNN, RoleHierRNN, RoleEncHierRNN, NLLLoss, RankLoss


class HierRNNPredictor(BasePredictor):
    """

    Args:
        dim_embed (int): Number of dimensions of word embeddings.
        dim_hidden (int): Number of dimensions of intermediate
            information embeddings.
    """

    def __init__(self, embeddings, dim_hidden,
                 dropout_rate=0.2, loss='NLLLoss', margin=0, threshold=None,
                 similarity='inner_product', fine_tune_emb=False,
                 has_info=False, **kwargs):
        super(HierRNNPredictor, self).__init__(**kwargs)
        self.dim_hidden = dim_hidden
        self.has_info = has_info
        if self.has_info:
            dim_embed = embeddings.size(1) + 2
        else:
            dim_embed = embeddings.size(1)

        self.model = HierRNN(dim_embed, dim_hidden,
                             similarity=similarity, has_emb=fine_tune_emb,
                             vol_size=embeddings.size(0))

        if fine_tune_emb:
            self.embeddings = self.model.embeddings
        else:
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
            if self.has_info:
                context = torch.cat([context,
                                     batch['prior'].unsqueeze(-1)
                                                   .to(self.device),
                                     batch['suggested'].unsqueeze(-1)
                                                       .to(self.device)
                                     ], -1)
                options = torch.cat([options,
                                     batch['option_prior'].unsqueeze(-1)
                                                          .to(self.device),
                                     batch['option_suggested'].unsqueeze(-1)
                                                              .to(self.device)
                                     ], -1)
                # options = [torch.cat([opt,
                #                       opt_prior.to(self.device),
                #                       opt_suggest.to(self.device)],
                #                      -1)
                #            for opt, opt_prior, opt_suggest in zip(
                #     options,
                #     batch['option_prior'],
                #     batch['option_suggested'])
                # ]
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


class UttHierRNNPredictor(BasePredictor):
    """

    Args:
        dim_embed (int): Number of dimensions of word embeddings.
        dim_hidden (int): Number of dimensions of intermediate
            information embeddings.
    """

    def __init__(self, embeddings, dim_hidden,
                 dropout_rate=0.2, loss='NLLLoss', margin=0, threshold=None,
                 similarity='inner_product', fine_tune_emb=False,
                 has_info=False, model_type='UttHierRNN', **kwargs):
        super(UttHierRNNPredictor, self).__init__(**kwargs)
        self.dim_hidden = dim_hidden
        self.has_info = has_info
        if self.has_info:
            dim_embed = embeddings.size(1) + 2
        else:
            dim_embed = embeddings.size(1)

        if model_type == 'UttHierRNN':
            self.model = UttHierRNN(dim_embed, dim_hidden,
                                    similarity=similarity, has_emb=fine_tune_emb,
                                    vol_size=embeddings.size(0))
        elif model_type == 'UttBinHierRNN':
            self.model = UttBinHierRNN(dim_embed, dim_hidden,
                                       similarity=similarity, has_emb=fine_tune_emb,
                                       vol_size=embeddings.size(0))
        else:
            print('Model type {} not supported!!!!!'.format(model_type))
            return None

        if fine_tune_emb:
            self.embeddings = self.model.embeddings
        else:
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
            if self.has_info:
                context = torch.cat([context,
                                     batch['prior'].unsqueeze(-1)
                                                   .to(self.device),
                                     batch['suggested'].unsqueeze(-1)
                                                       .to(self.device)
                                     ], -1)
                options = torch.cat([options,
                                     batch['option_prior'].unsqueeze(-1)
                                                          .to(self.device),
                                     batch['option_suggested'].unsqueeze(-1)
                                                              .to(self.device)
                                     ], -1)
                # options = [torch.cat([opt,
                #                       opt_prior.to(self.device),
                #                       opt_suggest.to(self.device)],
                #                      -1)
                #            for opt, opt_prior, opt_suggest in zip(
                #     options,
                #     batch['option_prior'],
                #     batch['option_suggested'])
                # ]
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


class RoleHierRNNPredictor(BasePredictor):
    def __init__(self, embeddings, dim_hidden,
                 dropout_rate=0.2, loss='NLLLoss', margin=0, threshold=None,
                 similarity='inner_product', fine_tune_emb=False,
                 has_info=False, **kwargs):
        super(RoleHierRNNPredictor, self).__init__(**kwargs)
        self.dim_hidden = dim_hidden
        self.has_info = has_info
        if self.has_info:
            dim_embed = embeddings.size(1) + 2
        else:
            dim_embed = embeddings.size(1)

        self.model = RoleHierRNN(dim_embed, dim_hidden,
                                similarity=similarity, has_emb=fine_tune_emb,
                                vol_size=embeddings.size(0))

        if fine_tune_emb:
            self.embeddings = self.model.embeddings
        else:
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
            context1 = self.embeddings(batch['context1'].to(self.device))
            context2 = self.embeddings(batch['context2'].to(self.device))
            options = self.embeddings(batch['options'].to(self.device))
            """
            if self.has_info:
                context = torch.cat([context,
                                     batch['prior'].unsqueeze(-1)
                                                   .to(self.device),
                                     batch['suggested'].unsqueeze(-1)
                                                       .to(self.device)
                                     ], -1)
                options = torch.cat([options,
                                     batch['option_prior'].unsqueeze(-1)
                                                          .to(self.device),
                                     batch['option_suggested'].unsqueeze(-1)
                                                              .to(self.device)
                                     ], -1)
            """
        logits = self.model.forward(
            context1.to(self.device),
            batch['utterance_ends1'],
            context2.to(self.device),
            batch['utterance_ends2'],
            options.to(self.device),
            batch['option_lens'])
        loss = self.loss(logits, batch['labels'].to(self.device))
        return logits, loss

    def _predict_batch(self, batch):
        context1 = self.embeddings(batch['context1'].to(self.device))
        context2 = self.embeddings(batch['context2'].to(self.device))
        options = self.embeddings(batch['options'].to(self.device))
        logits = self.model.forward(
            context1.to(self.device),
            batch['utterance_ends1'],
            context2.to(self.device),
            batch['utterance_ends2'],
            options.to(self.device),
            batch['option_lens'])
        # predicts = logits.max(-1)[1]
        return logits


class RoleEncHierRNNPredictor(BasePredictor):
    def __init__(self, embeddings, dim_hidden,
                 dropout_rate=0.2, loss='NLLLoss', margin=0, threshold=None,
                 similarity='inner_product', fine_tune_emb=False,
                 has_info=False, **kwargs):
        super(RoleEncHierRNNPredictor, self).__init__(**kwargs)
        self.dim_hidden = dim_hidden
        self.has_info = has_info
        if self.has_info:
            dim_embed = embeddings.size(1) + 2
        else:
            dim_embed = embeddings.size(1)

        self.model = RoleEncHierRNN(dim_embed, dim_hidden,
                                    similarity=similarity, has_emb=fine_tune_emb,
                                    vol_size=embeddings.size(0))

        if fine_tune_emb:
            self.embeddings = self.model.embeddings
        else:
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
            context1 = self.embeddings(batch['context1_masked'].to(self.device))
            context2 = self.embeddings(batch['context2_masked'].to(self.device))
            options = self.embeddings(batch['options'].to(self.device))
            """
            if self.has_info:
                context = torch.cat([context,
                                     batch['prior'].unsqueeze(-1)
                                                   .to(self.device),
                                     batch['suggested'].unsqueeze(-1)
                                                       .to(self.device)
                                     ], -1)
                options = torch.cat([options,
                                     batch['option_prior'].unsqueeze(-1)
                                                          .to(self.device),
                                     batch['option_suggested'].unsqueeze(-1)
                                                              .to(self.device)
                                     ], -1)
            """
        logits = self.model.forward(
            context1.to(self.device),
            context2.to(self.device),
            batch['utterance_ends1_masked'],
            options.to(self.device),
            batch['option_lens'])
        loss = self.loss(logits, batch['labels'].to(self.device))
        return logits, loss

    def _predict_batch(self, batch):
        context1 = self.embeddings(batch['context1_masked'].to(self.device))
        context2 = self.embeddings(batch['context2_masked'].to(self.device))
        options = self.embeddings(batch['options'].to(self.device))
        logits = self.model.forward(
            context1.to(self.device),
            context2.to(self.device),
            batch['utterance_ends1_masked'],
            options.to(self.device),
            batch['option_lens'])
        # predicts = logits.max(-1)[1]
        return logits


