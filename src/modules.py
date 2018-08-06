import math
import torch
import pdb


class DualRNN(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, dim_hidden,
                 similarity='inner_product'):
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
        self.similarity = {
            'cos': torch.nn.CosineSimilarity(dim=-1, eps=1e-6),
            'inner_product': BatchInnerProduct()
        }[similarity]

    def forward(self, context, context_lens, options, option_lens):
        context_hidden = self.context_encoder(context, context_lens)
        predict_option = self.transform(context_hidden)
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            # option_hidden.size() == (batch, dim_hidden)
            option_hidden = self.option_encoder(option,
                                                [ol[i] for ol in option_lens])
            # logit.size() == (batch,)
            logit = self.similarity(predict_option, option_hidden)
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits


class HierRNN(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, dim_hidden,
                 similarity='inner_product'):
        super(DualRNN, self).__init__()
        self.context_encoder = HierRNNEncoder(dim_embeddings,
                                              dim_hidden, dim_hidden)
        self.option_encoder = LSTMEncoder(dim_embeddings, dim_hidden)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * dim_hidden, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        self.transform = torch.nn.Linear(2 * dim_hidden, 2 * dim_hidden)
        self.similarity = {
            'cos': torch.nn.CosineSimilarity(dim=-1, eps=1e-6),
            'inner_product': BatchInnerProduct()
        }[similarity]

    def forward(self, context, context_ends, options, option_lens):
        context_hidden = self.context_encoder(context, context_ends)
        predict_option = self.transform(context_hidden)
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            # option_hidden.size() == (batch, dim_hidden)
            option_hidden = self.option_encoder(option,
                                                [ol[i] for ol in option_lens])
            # logit.size() == (batch,)
            logit = self.similarity(predict_option, option_hidden)
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits


class HierRNNEncoder(torch.nn.Module):
    """ 

    Args:

    """
    def __init__(self, dim_embeddings, dim_hidden1=128, dim_hidden2=128):
        super(HierRNNEncoder, self).__init__()
        self.rnn1 = torch.nn.LSTM(dim_embeddings,
                                  dim_hidden1,
                                  1,
                                  bidirectional=True,
                                  batch_first=True)
        self.rnn2 = LSTMEncoder(dim_hidden1, dim_hidden2)
        self.register_buffer('padding', torch.zeros(dim_hidden2))

    def forward(self, seq, ends):
        batch_size = seq.size(0)

        outputs, _ = self.rnn1(seq)
        end_outputs = [[outputs[b, end] for end in ends[b]]
                       for b in range(batch_size)]
        end_outputs = pad_and_cat(end_outputs, self.padding)
        encoded = self.rnn2(end_outputs)
        return encoded


class BatchInnerProduct(torch.nn.Module):
    """ 

    Args:

    """

    def __init__(self):
        super(BatchInnerProduct, self).__init__()

    def forward(self, a, b):
        return (a * b).sum(-1)


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
        self.margin = margin

    def forward(self, logits, labels):
        positive_mask = (1 - labels).byte()
        positive_min = torch.min(logits.masked_fill(
            positive_mask, math.inf), -1)[0]

        negative_mask = labels.byte()
        negative_max = torch.max(logits.masked_fill(
            negative_mask, -math.inf), -1)[0]

        loss = self.margin_ranking_loss(positive_min, negative_max,
                                        torch.ones_like(negative_max))
        loss = (negative_max - positive_min).mean()
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
        loss = (-torch.log(probs + self.epsilon) *
                labels.float()).sum() / batch_size
        return loss


def pad_and_cat(tensors, padding):
    """ Pad lists to have same number of elements, and concatenate
    those elements to a 3d tensor.

    Args:
        tensors (list of list of Tensors): Each list contains
            list of operand embeddings. Each operand embedding is of
            size (dim_element,).
        padding (Tensor):
            Element used to pad lists, with size (dim_element,).

    Return:
        n_tensors (list of int): Length of lists in tensors.
        tensors (Tensor): Concatenated tensor after padding the list.
    """
    n_tensors = [len(ts) for ts in tensors]
    pad_size = max(n_tensors)

    # pad to has same number of operands for each problem
    tensors = [ts + (pad_size - len(ts)) * [padding]
               for ts in tensors]

    # tensors.size() = (batch_size, pad_size, dim_hidden)
    tensors = torch.stack([torch.stack(t)
                           for t in tensors], dim=0)

    return n_tensors, tensors
