import torch


class DualRNN(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, dim_hidden):
        super(DualRNN, self).__init__()
        self.context_encoder = LSTMEncoder(dim_embeddings, dim_hidden)
        self.option_encoder = LSTMEncoder(dim_embeddings, dim_hidden)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * dim_hidden, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

    def forward(self, context, context_lens, options, option_lens):
        context_hidden = self.context_encoder(context, context_lens)
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            # option_hidden.size() == (batch, dim_hidden)
            option_hidden = self.option_encoder(option,
                                                [ol[i] for ol in option_lens])
            # logit.size() == (batch, 1)
            logit = self.mlp(torch.cat([context_hidden, option_hidden], -1))
            logits.append(logit)
        logits = torch.cat(logits, 1)
        return logits


class LSTMEncoder(torch.nn.Module):
    """ 

    Args:

    """
    def __init__(self, dim_embeddings, dim_hidden):
        super(LSTMEncoder, self).__init__()
        self.rnn = torch.nn.LSTM(dim_embeddings,
                                 dim_hidden,
                                 1,
                                 bidirectional=True,
                                 batch_first=True)

    def forward(self, seqs, seq_lens):
        _, hidden = self.rnn(seqs)
        _, c = hidden
        return c.transpose(1, 0).contiguous().view(c.size(1), -1)
