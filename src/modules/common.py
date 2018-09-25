import math
import torch
import pdb
# from .transformer import EncoderLayer as TransformerEncoder
from .transformer2 import Encoder as TransformerEncoder


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
                 similarity='inner_product', has_emb=False, vol_size=-1):
        super(HierRNN, self).__init__()
        self.context_encoder = HierRNNEncoder(dim_embeddings,
                                              dim_hidden, dim_hidden)
        self.option_encoder = LSTMEncoder(dim_embeddings, dim_hidden)
        """
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * dim_hidden, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )
        """
        self.transform = torch.nn.Linear(2 * dim_hidden, 2 * dim_hidden)
        self.similarity = {
            'cos': torch.nn.CosineSimilarity(dim=-1, eps=1e-6),
            'inner_product': BatchInnerProduct()
        }[similarity]
        if has_emb:
            self.embeddings = torch.nn.Embedding(vol_size,
                                                 dim_embeddings)

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


class UttHierRNN(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, dim_hidden,
                 similarity='inner_product', has_emb=False, vol_size=-1,
                 dropout_rate=0.0):
        super(UttHierRNN, self).__init__()
        self.utterance_encoder = LSTMEncoder(dim_embeddings, dim_hidden)
        self.context_encoder = HierRNNEncoder(dim_embeddings,
                                              dim_hidden, dim_hidden,
                                              self.utterance_encoder.rnn)
        self.dropout_ctx_encoder = torch.nn.Dropout(p=dropout_rate)
        self.transform = torch.nn.Linear(2 * dim_hidden, 2 * dim_hidden)
        self.similarity = {
            'cos': torch.nn.CosineSimilarity(dim=-1, eps=1e-6),
            'inner_product': BatchInnerProduct()
        }[similarity]
        if has_emb:
            self.embeddings = torch.nn.Embedding(vol_size,
                                                 dim_embeddings)

    def forward(self, context, context_ends, options, option_lens):
        context_hidden = self.context_encoder(context, context_ends)
        context_hidden = self.dropout_ctx_encoder(context_hidden)
        predict_option = self.transform(context_hidden)
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            # option_hidden.size() == (batch, dim_hidden)
            option_hidden = self.utterance_encoder(option,
                                                   [ol[i] for ol in option_lens])
            # logit.size() == (batch,)
            logit = self.similarity(predict_option, option_hidden)
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits


class UttBinHierRNN(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, dim_hidden,
                 similarity='inner_product', has_emb=False, vol_size=-1,
                 dropout_rate=0.0):
        super(UttBinHierRNN, self).__init__()
        self.utterance_encoder = LSTMEncoder(dim_embeddings, dim_hidden)
        self.context_encoder = HierRNNEncoder(dim_embeddings,
                                              dim_hidden, dim_hidden,
                                              self.utterance_encoder.rnn)
        self.transform = torch.nn.Linear(2 * dim_hidden, 2 * dim_hidden)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4 * dim_hidden, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1, bias=False)
        )
        self.similarity = {
            'cos': torch.nn.CosineSimilarity(dim=-1, eps=1e-6),
            'inner_product': BatchInnerProduct()
        }[similarity]
        if has_emb:
            self.embeddings = torch.nn.Embedding(vol_size,
                                                 dim_embeddings)

    def forward(self, context, context_ends, options, option_lens):
        context_hidden = self.context_encoder(context, context_ends)
        # predict_option = self.transform(context_hidden)
        predict_option = context_hidden
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            # option_hidden.size() == (batch, dim_hidden)
            option_hidden = self.utterance_encoder(option,
                                                   [ol[i] for ol in option_lens])
            option_concat = torch.cat([option_hidden, predict_option], -1)
            logit = self.mlp(option_concat)

            #logit = self.mlp(option_hidden + predict_option)
            
            #logit = self.similarity(predict_option, option_hidden)
            logit = torch.reshape(logit, (-1,))
            logits.append(logit)
        
        logits = torch.stack(logits, 1)
        # print(logits.data[0])
        return logits


class RoleHierRNN(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, dim_hidden,
                 similarity='inner_product', has_emb=False, vol_size=-1):
        super(RoleHierRNN, self).__init__()
        self.utterance_encoder = LSTMEncoder(dim_embeddings, dim_hidden)
        self.context1_encoder = HierRNNEncoder(dim_embeddings,
                                               dim_hidden, dim_hidden,
                                               self.utterance_encoder.rnn)
        self.context2_encoder = HierRNNEncoder(dim_embeddings,
                                               dim_hidden, dim_hidden,
                                               self.utterance_encoder.rnn)
        self.transform = torch.nn.Linear(2 * dim_hidden, 2 * dim_hidden)
        self.similarity = {
            'cos': torch.nn.CosineSimilarity(dim=-1, eps=1e-6),
            'inner_product': BatchInnerProduct()
        }[similarity]
        if has_emb:
            self.embeddings = torch.nn.Embedding(vol_size,
                                                 dim_embeddings)

    def forward(self, context1, context_ends1, 
                context2, context_ends2, 
                options, option_lens):
        context1_hidden = self.context1_encoder(context1, context_ends1)
        context2_hidden = self.context2_encoder(context2, context_ends2)
        context_hidden = torch.add(context1_hidden, context2_hidden)
        predict_option = self.transform(context_hidden)
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            # option_hidden.size() == (batch, dim_hidden)
            option_hidden = self.utterance_encoder(option,
                                                   [ol[i] for ol in option_lens])
            # logit.size() == (batch,)
            logit = self.similarity(predict_option, option_hidden)
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits


class RoleEncHierRNN(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, dim_hidden,
                 similarity='inner_product', has_emb=False, vol_size=-1):
        super(RoleEncHierRNN, self).__init__()
        self.context_encoder = RoleHierRNNEncoder(dim_embeddings,
                                                  dim_hidden, dim_hidden)
        self.option_encoder = LSTMEncoder(dim_embeddings, dim_hidden)
        self.transform = torch.nn.Linear(2 * dim_hidden, 2 * dim_hidden)
        self.similarity = {
            'cos': torch.nn.CosineSimilarity(dim=-1, eps=1e-6),
            'inner_product': BatchInnerProduct()
        }[similarity]
        if has_emb:
            self.embeddings = torch.nn.Embedding(vol_size,
                                                 dim_embeddings)

    def forward(self, context1, context2,
                context_ends, 
                options, option_lens):
        context_hidden = self.context_encoder(context1, context2, context_ends)
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
    def __init__(self, 
                 dim_embeddings, 
                 dim_hidden1=128, 
                 dim_hidden2=128,
                 rnn1=None,
                 rnn2=None):
        super(HierRNNEncoder, self).__init__()

        if rnn1 is not None:
            self.rnn1 = rnn1
        else:
            self.rnn1 = torch.nn.LSTM(dim_embeddings,
                                      dim_hidden1,
                                      1,
                                      bidirectional=True,
                                      batch_first=True)
        if rnn2 is not None:
            self.rnn2 = rnn2
        else:
            self.rnn2 = LSTMEncoder(2 * dim_hidden1, dim_hidden2)

        self.register_buffer('padding', torch.zeros(2 * dim_hidden2))

    def forward(self, seq, ends):
        batch_size = seq.size(0)

        outputs, _ = self.rnn1(seq)
        end_outputs = [[outputs[b, end] for end in ends[b]]
                       for b in range(batch_size)]
        lens, end_outputs = pad_and_cat(end_outputs, self.padding)
        encoded = self.rnn2(end_outputs, list(map(len, ends)))
        return encoded


class RoleHierRNNEncoder(torch.nn.Module):
    """ 

    Args:

    """
    def __init__(self, dim_embeddings, dim_hidden1=128, dim_hidden2=128):
        super(RoleHierRNNEncoder, self).__init__()
        self.rnn1_1 = torch.nn.LSTM(dim_embeddings,
                                    dim_hidden1,
                                    1,
                                    bidirectional=True,
                                    batch_first=True)
        self.rnn1_2 = torch.nn.LSTM(dim_embeddings,
                                    dim_hidden1,
                                    1,
                                    bidirectional=True,
                                    batch_first=True)
        self.rnn2 = LSTMEncoder(2 * dim_hidden1, dim_hidden2)
        self.register_buffer('padding', torch.zeros(2 * dim_hidden2))

    def forward(self, seq1, seq2, ends):
        batch_size = seq1.size(0)

        outputs1, _ = self.rnn1_1(seq1)
        outputs2, _ = self.rnn1_2(seq2)
        outputs = torch.add(outputs1, outputs2)
        end_outputs = [[outputs[b, end] for end in ends[b]]
                       for b in range(batch_size)]
        lens, end_outputs = pad_and_cat(end_outputs, self.padding)
        encoded = self.rnn2(end_outputs, list(map(len, ends)))
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

    def forward(self, seqs, seq_lens=None):
        _, hidden = self.rnn(seqs)
        _, c = hidden
        return c.transpose(1, 0).contiguous().view(c.size(1), -1)


class RankLoss(torch.nn.Module):
    """ 
    Args:

    """

    def __init__(self, margin=0.2, threshold=None):
        super(RankLoss, self).__init__()
        self.margin_ranking_loss = torch.nn.MarginRankingLoss(margin)
        self.margin = margin
        self.threshold = None
        self.margin = margin

    def forward(self, logits, labels):
        positive_mask = (1 - labels).byte()
        positive_logits = logits.masked_fill(positive_mask, math.inf)
        positive_min = lse_min(positive_logits)

        negative_mask = labels.byte()
        negative_logits = logits.masked_fill(negative_mask, -math.inf)
        negative_max = lse_max(negative_logits)

        ones = torch.ones_like(negative_max)
        if self.threshold is None:
            loss = self.margin_ranking_loss(positive_min,
                                            negative_max + self.margin,
                                            ones)
        else:
            loss = (self.margin_ranking_loss(positive_min,
                                             self.threshold + self.margin,
                                             ones)
                    + self.margin_ranking_loss(negative_max,
                                               self.threshold - self.margin,
                                               - ones)
                    )

        return loss.mean()


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


def pad_seqs(tensors, pad_element):
    lengths = list(map(len, tensors))
    max_length = max(lengths)
    padded = []
    for tensor, length in zip(tensors, lengths):
        if max_length > length:
            padding = torch.stack(
                [pad_element] * (max_length - length), 0
            )
            padded.append(torch.cat([tensor, padding], 0))
        else:
            padded.append(tensor)

    return lengths, torch.stack(padded, 0)


def lse_max(a, dim=-1):
    max_a = torch.max(a, dim=dim, keepdim=True)[0]
    lse = torch.log(torch.sum(torch.exp(a - max_a), dim=dim)) + max_a
    return lse


def lse_min(a, dim=-1):
    return -lse_max(-a, dim)
