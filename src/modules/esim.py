import math
import torch
import torch.nn.functional as F

from torch import nn

from .common import HierRNNEncoder, LSTMEncoder, LSTMPoolingEncoder, pad_and_cat
from .mcan import HighwayNetwork
from .attention import make_mask


class ESIM(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, dim_hidden,
                 similarity='inner_product', has_emb=False,
                 vol_size=-1, dropout_rate=0.0, use_projection=False):
        super(ESIM, self).__init__()
        self.dim_embeddings = dim_embeddings
        self.dim_hidden = dim_hidden
        self.use_projection = use_projection
        
        self.bilstm1 = torch.nn.LSTM(dim_embeddings,
                                     dim_hidden,
                                     1,
                                     bidirectional=True,
                                     batch_first=True)
        self.bilstm2 = torch.nn.LSTM(dim_hidden,
                                     dim_hidden,
                                     1,
                                     bidirectional=True,
                                     batch_first=True)
       
        self.matching = ESIMCoAttention(2 * dim_hidden,
                                        dim_hidden,
                                        self.use_projection)

        self.prediction_layer = torch.nn.Sequential(
            torch.nn.Linear(4 * 2 * dim_hidden, dim_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(dim_hidden, 1, bias=True)
        )

        if has_emb:
            self.embeddings = torch.nn.Embedding(vol_size,
                                                 dim_embeddings)

    def forward(self, context, context_ends, options, option_lens):
        batch_size, n_options, len_option, dim_option = options.size()
        
        context_lens = [ends[-1] for ends in context_ends]
        context_mask = make_mask(context, context_lens)

        outputs_context, _ = self.bilstm1(context)
        outputs_options, _ = self.bilstm1(options.view(-1, len_option, dim_option))
        outputs_options = outputs_options.view(batch_size, n_options, len_option, -1)

        logits = []
        for i, option in enumerate(outputs_options.transpose(0, 1)):
            option_len = [ol[i] for ol in option_lens]
            option_mask = make_mask(option, option_len)

            enhanced_context, enhanced_option = self.matching(
                outputs_context, context_lens,
                option, option_len
            )

            composed_context, _ = self.bilstm2(enhanced_context)
            composed_option, _ = self.bilstm2(enhanced_option)

            composed_context_avg = torch.sum(composed_context * context_mask.unsqueeze(-1), dim=1)
            composed_context_avg /= torch.sum(context_mask, dim=1, keepdim=True)

            composed_option_avg = torch.sum(composed_option * option_mask.unsqueeze(-1), dim=1)
            composed_option_avg /= torch.sum(option_mask, dim=1, keepdim=True)

            composed_context_max, _ = torch.max(
                composed_context.masked_fill(context_mask.unsqueeze(-1) == 0, -1e7), dim=1)
            composed_option_max, _ = torch.max(
                composed_option.masked_fill(option_mask.unsqueeze(-1) == 0, -1e7), dim=1)

            fused = torch.cat(
                [composed_context_avg, composed_context_max,
                 composed_option_avg, composed_option_max],
                -1
            )
            
            logit = self.prediction_layer(fused)

            logit = torch.reshape(logit, (-1,))
            logits.append(logit)

        logits = torch.stack(logits, 1)
        return logits


class ESIMCoAttention(nn.Module):
    def __init__(self, dim_hidden, dim_output, use_projection=False):
        super(ESIMCoAttention, self).__init__()
        
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.use_projection = use_projection

        if use_projection:
            self.M = nn.Linear(self.dim_hidden, self.dim_hidden)
        
        self.F = nn.Sequential(nn.Linear(4 * dim_hidden, dim_output), nn.ReLU())
        
    def forward(self, query, query_lens, option, option_lens):
        if self.use_projection:
            affinity = torch.matmul(self.M(query), option.transpose(1, 2))
        else:
            affinity = torch.matmul(query, option.transpose(1, 2))
        
        # affinity.size() = (batch_size, query_len, option_len)
        mask_query = make_mask(query, query_lens)
        # mask_query.size() = (batch_size, query_len)
        mask_option = make_mask(option, option_lens)
        # mask_option.size() = (batch_size, option_len)

        masked_affinity_query = affinity.masked_fill(
            mask_query.unsqueeze(2) == 0, -math.inf)
        masked_affinity_option = affinity.masked_fill(
            mask_option.unsqueeze(1) == 0, -math.inf)

        query_weights = F.softmax(masked_affinity_option, dim=2)
        summary_query = torch.matmul(query_weights, option)
        option_weights = F.softmax(masked_affinity_query, dim=1)
        summary_option = torch.matmul(option_weights.transpose(1, 2),
                                      query)

        query_match = self.F(torch.cat([summary_query,
                                        query,
                                        summary_query-query,
                                        summary_query*query], -1))
        option_match = self.F(torch.cat([summary_option,
                                         option,
                                         summary_option-option,
                                         summary_option*option], -1))

        return query_match, option_match 
