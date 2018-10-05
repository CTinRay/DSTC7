import torch
import torch.nn.functional as F
import pdb

from .common import HierRNNEncoder, LSTMEncoder, LSTMPoolingEncoder, Conv1dEncoder, pad_and_cat
from .attention import CoAttentionEncoder, IntraAttention


class MCAN(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, dim_hidden,
                 similarity='inner_product', has_emb=False, vol_size=-1,
                 dropout_rate=0.0, utt_enc_type='rnn',
                 use_co_att=False, use_intra_att=False,
                 only_last_context=False):
        
        super(MCAN, self).__init__()
        self.dim_embeddings = self.dim_features = dim_embeddings

        self.use_intra_att = use_intra_att
        if use_intra_att:
            self.intra_att_context = IntraAttention(dim_embeddings)
            self.intra_att_option = IntraAttention(dim_embeddings)
            self.dim_features += 3

        self.use_co_att = use_co_att
        if use_co_att:
            self.co_att_encoder = CoAttentionEncoder(dim_embeddings)
            self.dim_features += 9

        if utt_enc_type == 'rnn':
            self.utterance_encoder = LSTMPoolingEncoder(self.dim_features,
                                                        dim_hidden,
                                                        'meanmax')
        else:
            self.utterance_encoder = Conv1dEncoder(
                dim_embeddings,
                dim_hidden // 2,
                kernel_sizes=[2, 3, 4, 5])
        
        self.only_last_context = only_last_context
        if not only_last_context:
            ctx_enc = LSTMEncoder(4 * dim_hidden, 2 * dim_hidden)
            self.context_encoder = HierRNNEncoder(
                self.dim_features,
                dim_hidden, 2 * dim_hidden,
                utt_enc=self.utterance_encoder,
                ctx_enc=ctx_enc,
                utt_enc_type='pool-lstm')

        h_dim = 8 * dim_hidden
        self.prediction_layer = torch.nn.Sequential(
            HighwayNetwork(h_dim, n_layers=2),
            torch.nn.Linear(h_dim, 1, bias=True)
        )

        if has_emb:
            self.embeddings = torch.nn.Embedding(vol_size,
                                                 dim_embeddings)

    def forward(self, context, context_ends, options, option_lens):
        context_lens = [ends[-1] for ends in context_ends]
        
        if self.only_last_context:
            context_last = [
                list(context[i, (ends[-2] if len(ends)>1 else 0):ends[-1]])
                for i, ends in enumerate(context_ends)
            ]
            padding = torch.zeros_like(context[0, 0]).cuda()
            context_lens, context = pad_and_cat(context_last, padding)

        if self.use_intra_att:
            context_intra = self.intra_att_context(context, context_lens)
        
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            option_len = [ol[i] for ol in option_lens]

            if self.use_co_att:
                context_cast, option_cast = self.co_att_encoder(
                    context, context_lens, option, option_len)
            else:
                context_cast = context
                option_cast = option

            if self.use_intra_att:
                option_intra = self.intra_att_option(option, option_len)
                context_cast = torch.cat([context_cast, context_intra], -1)
                option_cast = torch.cat([option_cast, option_intra], -1)

            # context_hidden = self.context_encoder(context_cast, context_ends)
            if self.only_last_context:
                context_hidden = self.utterance_encoder(context_cast)
            else:
                context_hidden = self.context_encoder(context_cast, context_ends)
            # option_hidden.size() == (batch, dim_hidden)
            option_hidden = self.utterance_encoder(option_cast)

            fused = torch.cat(
                [option_hidden*context_hidden, option_hidden-context_hidden],
                -1
            )
            
            logit = self.prediction_layer(fused)

            logit = torch.reshape(logit, (-1,))
            logits.append(logit)

        logits = torch.stack(logits, 1)
        return logits


class HighwayNetwork(torch.nn.Module):
    def __init__(self, dim_hidden, n_layers=1):
        super(HighwayNetwork, self).__init__()
        self.dim_hidden = dim_hidden

        self.layers = torch.nn.ModuleList(
            [
                HighwayLayer(dim_hidden)
                for i in range(n_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))

        return x


class HighwayLayer(torch.nn.Module):
    def __init__(self, dim_hidden):
        super(HighwayLayer, self).__init__()
        self.dim_hidden = dim_hidden

        self.H = torch.nn.Sequential(
            torch.nn.Linear(dim_hidden, dim_hidden),
            torch.nn.ReLU()
        )
        
        self.T = torch.nn.Sequential(
            torch.nn.Linear(dim_hidden, dim_hidden),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        t = self.T(x)
        h = self.H(x)

        y = h * t + x * (1-t)
        return y
