import math
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
        self.transform = torch.nn.Linear(2 * dim_hidden, 2 * dim_hidden)
        self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, context, context_lens, options, option_lens):
        context_hidden = self.context_encoder(context, context_lens)
        predict_option = self.transform(context_hidden)
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            # option_hidden.size() == (batch, dim_hidden)
            option_hidden = self.option_encoder(option,
                                                [ol[i] for ol in option_lens])
            # logit.size() == (batch,)
            # logit = (predict_option * option_hidden).sum(-1)
            logit = self.cos(predict_option, option_hidden)
            logits.append(logit)
        logits = torch.stack(logits, 1)
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


class RankLoss(torch.nn.Module):
    """ 
    Args:
        
    """
    def __init__(self, margin=0):
        super(RankLoss, self).__init__()
        self.margin_ranking_loss = torch.nn.MarginRankingLoss(margin)

    def forward(self, logits, labels):
        positive_mask = (1 - labels).byte()
        positive_min = torch.min(logits.masked_fill(positive_mask, math.inf), -1)[0]

        negative_mask = labels.byte()
        negative_max = torch.max(logits.masked_fill(negative_mask, -math.inf), -1)[0]

        loss = self.margin_ranking_loss(positive_min, negative_max,
                                        torch.ones_like(negative_max))
        return loss


class NLLLoss(torch.nn.Module):
    """ 
    Args:
        
    """
    def __init__(self, epsilon=1e-6):
        super(NLLLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, labels):
        batch_size = logits.size(0)
        probs = torch.nn.functional.softmax(logits, -1)
        loss = (-torch.log(probs + self.epsilon) * labels.float()).sum() / batch_size
        return loss