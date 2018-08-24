



import collections
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import shared_base
import utils

def _get_dropped_weights(w_raw, dropout_p, is_training):
    dropped_w = F.dropout(w_raw, p=dropout_p, training=is_training)
    if isinstance(dropped_w, torch.nn.Parameter):
        dropped_w = dropped_w.clone()
    return dropped_w


class LockedDropout(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class EmbeddingDropout(torch.nn.Embedding):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 max_norm=None,
                 norm_type=2,
                 scale_grad_by_freq=False,
                 sparse=False,
                 dropout=0.1,
                 scale=None):

        torch.nn.Embedding.__init__(self,
                                    num_embeddings=num_embeddings,
                                    embedding_dim=embedding_dim,
                                    max_norm=max_norm,
                                    norm_type=norm_type,
                                    scale_grad_by_freq=scale_grad_by_freq,
                                    sparse=sparse)
        self.dropout = dropout
        assert (dropout >= 0.0) and (dropout < 1.0), ('Dropout must be >= 0.0 '
                                                      'and < 1.0')
        self.scale = scale

    def forward(self, inputs):
        if self.training:
            dropout = self.dropout
        else:
            dropout = 0

        if dropout:
            mask = self.weight.data.new(self.weight.size(0), 1)
            mask.bernoulli_(1 - dropout)
            mask = mask.expand_as(self.weight)
            mask = mask / (1 - dropout)
            masked_weight = self.weight * Variable(mask)
        else:
            masked_weight = self.weight
        if self.scale and self.scale != 1:
            masked_weight = masked_weight * self.scale

        return F.embedding(inputs,
                           masked_weight,
                           max_norm=self.max_norm,
                           norm_type=self.norm_type,
                           scale_grad_by_freq=self.scale_grad_by_freq,
                           sparse=self.sparse)


class RNN(shared_base.SharedModel):
    def __init__(self, args, corpus):
        shared_base.SharedModel.__init__(self)

        self.args = args
        self.corpus = corpus

        self.decoder = torch.nn.Linear(in_features=args.shared_hid, out_features=corpus.num_tokens)
        self.encoder = EmbeddingDropout(corpus.num_tokens, args.shared_embed, dropout=args.shared_dropoute)
        self.lockdrop = LockedDropout()


        ## ???
        if self.args.tie_weights:
            self.decoder.weight = self.encoder.weight

        self.w_xc = torch.nn.Linear(args.shared_embed, args.shared_hid)
        self.w_xh = torch.nn.Linear(args.shared_embed, args.shared_hid)

        # The raw weights are stored here because the hidden-to-hidden weights
        # are weight dropped on the forward pass.
        self.w_hc_raw = torch.nn.Parameter(torch.Tensor(args.shared_hid, args.shared_hid))
        self.w_hh_raw = torch.nn.Parameter(torch.Tensor(args.shared_hid, args.shared_hid))
        self.w_hc = None
        self.w_hh = None

        self.w_h = collections.defaultdict(dict)
        self.w_c = collections.defaultdict(dict)

        for idx in range(args.num_blocks):
            for jdx in range(idx + 1, args.num_blocks):
                self.w_h[idx][jdx] = torch.nn.Linear(args.shared_hid, args.shared_hid, bias=False)
                self.w_c[idx][jdx] = torch.nn.Linear(args.shared_hid, args.shared_hid, bias=False)

        self._w_h = torch.nn.ModuleList([self.w_h[idx][jdx]
                                   for idx in self.w_h
                                   for jdx in self.w_h[idx]])
        self._w_c = torch.nn.ModuleList([self.w_c[idx][jdx]
                                   for idx in self.w_c
                                   for jdx in self.w_c[idx]])

        if args.mode == 'train':
            self.batch_norm = torch.nn.BatchNorm1d(args.shared_hid)
        else:
            self.batch_norm = None

        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)

    def forward(self, inputs, dag, hidden=None, is_train=True):
        time_steps = inputs.size(0)
        batch_size = inputs.size(1)

        is_train = is_train and self.args.mode in ['train']

        self.w_hh = _get_dropped_weights(self.w_hh_raw,
                                         self.args.shared_wdrop,
                                         self.training)
        self.w_hc = _get_dropped_weights(self.w_hc_raw,
                                         self.args.shared_wdrop,
                                         self.training)

        if hidden is None:
            hidden = self.static_init_hidden[batch_size]

        embed = self.encoder(inputs)

        if self.args.shared_dropouti > 0:
            embed = self.lockdrop(embed, self.args.shared_dropouti if is_train else 0)

        clipped_num = 0
        max_clipped_norm = 0
        h1tohT = []
        logits = []
        for step in range(time_steps):
            x_t = embed[step]
            logit, hidden = self.cell(x_t, hidden, dag)

            hidden_norms = hidden.norm(dim=-1)
            max_norm = 25.0
            if hidden_norms.data.max() > max_norm:
                hidden_norms = hidden_norms.data.cpu().numpy()

                clipped_num += 1
                if hidden_norms.max() > max_clipped_norm:
                    max_clipped_norm = hidden_norms.max()

                clip_select = hidden_norms > max_norm
                clip_norms = hidden_norms[clip_select]

                mask = np.ones(hidden.size())
                normalizer = max_norm/clip_norms
                normalizer = normalizer[:, np.newaxis]

                mask[clip_select] = normalizer
                hidden *= torch.autograd.Variable(
                    torch.FloatTensor(mask).cuda(), requires_grad=False)

            logits.append(logit)
            h1tohT.append(hidden)

        h1tohT = torch.stack(h1tohT)
        output = torch.stack(logits)
        raw_output = output
        if self.args.shared_dropout > 0:
            output = self.lockdrop(output,
                                   self.args.shared_dropout if is_train else 0)

        dropped_output = output

        decoded = self.decoder(
            output.view(output.size(0)*output.size(1), output.size(2)))
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))

        extra_out = {'dropped': dropped_output,
                     'hiddens': h1tohT,
                     'raw': raw_output}
        return decoded, hidden, extra_out

    def cell(self, x, h_prev, dag):
        """Computes a single pass through the discovered RNN cell."""
        c = {}
        h = {}
        f = {}

        f[0] = self.get_f(dag[-1][0].name)
        c[0] = torch.sigmoid(self.w_xc(x) + F.linear(h_prev, self.w_hc, None))
        h[0] = (c[0]*f[0](self.w_xh(x) + F.linear(h_prev, self.w_hh, None)) +
                (1 - c[0])*h_prev)

        leaf_node_ids = []
        q = collections.deque()
        q.append(0)

        while True:
            if len(q) == 0:
                break

            node_id = q.popleft()
            nodes = dag[node_id]

            for next_node in nodes:
                next_id = next_node.id
                if next_id == self.args.num_blocks:
                    leaf_node_ids.append(node_id)
                    assert len(nodes) == 1, ('parent of leaf node should have '
                                             'only one child')
                    continue

                w_h = self.w_h[node_id][next_id]
                w_c = self.w_c[node_id][next_id]

                f[next_id] = self.get_f(next_node.name)
                c[next_id] = torch.sigmoid(w_c(h[node_id]))
                h[next_id] = (c[next_id]*f[next_id](w_h(h[node_id])) +
                              (1 - c[next_id])*h[node_id])

                q.append(next_id)


        leaf_nodes = [h[node_id] for node_id in leaf_node_ids]
        output = torch.mean(torch.stack(leaf_nodes, 2), -1)

        # stabilizing the Updates of omega
        if self.batch_norm is not None:
            output = self.batch_norm(output)

        return output, h[self.args.num_blocks - 1]

    def get_f(self, name):
        name = name.lower()
        if name == 'relu':
            f = F.relu
        elif name == 'tanh':
            f = torch.tanh
        elif name == 'identity':
            f = lambda x: x
        elif name == 'sigmoid':
            f = torch.sigmoid
        return f

    def reset_parameters(self):
        init_range = 0.025 if self.args.mode == 'train' else 0.04
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.args.shared_hid)
        return utils.get_variable(zeros, self.args.cuda, requires_grad=False)