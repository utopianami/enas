
import config
import collections
import os
import utils
import torch
import torch.nn.functional as F



Node = collections.namedtuple('Node', ['id', 'name'])

def _construct_dags(prev_nodes, activations, func_names, num_blocks):
    dags = []
    for nodes, func_ids in zip(prev_nodes, activations):
        dag = collections.defaultdict(list)

        # add first node
        dag[-1] = [Node(0, func_names[func_ids[0]])]
        dag[-2] = [Node(0, func_names[func_ids[0]])]

        # add following nodes
        for jdx, (idx, func_id) in enumerate(zip(nodes, func_ids[1:])):
            dag[utils.to_item(idx)].append(Node(jdx + 1, func_names[func_id]))

        leaf_nodes = set(range(num_blocks)) - dag.keys()

        # merge with avg
        for idx in leaf_nodes:
            dag[idx] = [Node(num_blocks, 'avg')]

        last_node = Node(num_blocks + 1, 'h[t]')
        dag[num_blocks] = [last_node]
        dags.append(dag)

    return dags


class Controller(torch.nn.Module):
    def __init__(self, args):
        super(Controller, self).__init__()
        self.args = args
        self.func_names = self.args.shared_rnn_activations

        # num_token list
        self.num_tokens = [len(args.shared_rnn_activations)]  # "['tanh', 'ReLU', 'identity', 'sigmoid']"
        for idx in range(self.args.num_blocks):  # self.args.num_blocks = 12
            self.num_tokens += [idx + 1, len(args.shared_rnn_activations)]

        self.num_total_tokens = len(args.shared_rnn_activations) + self.args.num_blocks + 1 # +1: first_input
        self.encoder = torch.nn.Embedding(num_embeddings=self.num_total_tokens, embedding_dim=args.controller_hid)
        self.lstm = torch.nn.LSTMCell(input_size=args.controller_hid, hidden_size=args.controller_hid)

        # edge weight
        self.decoders = []
        for idx, size in enumerate(self.num_tokens):
            decoder = torch.nn.Linear(in_features=args.controller_hid, out_features=size)
            self.decoders.append(decoder)

        self._decoders = torch.nn.ModuleList(self.decoders)
        self.reset_parameters()

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def forward(self, inputs, hidden, block_idx):
        embed = self.encoder(inputs)

        # hidden = (hx,cx)
        hx, cx = self.lstm(input=embed, hx=hidden)
        logits = self.decoders[block_idx](hx)

        logits /= self.args.softmax_temperature  # softmax_temperature5.0

        # exploration: 이중에서 선택 vs. exploitation: 그 중 큰 것을 선택
        # train에서 다양한 model을 선택하기 위해
        # exploration: tanh의 효과, tanh_c -1, 1이 아닌 더 큰 세팅을 원해서
        # tanh_c 2.5 를 봤다더라 ....ㅋㅋㅋ
        if self.args.mode == 'train':
            logits = (self.args.tanh_c * F.tanh(logits))

        return logits, (hx, cx)

    def sample(self, batch_size=1, with_details=False, save_dir=None):
        assert batch_size >= 1

        # [B, L, H]
        tmp = torch.Tensor([self.num_total_tokens-1]).long()
        inputs = utils.get_variable(tmp, requires_grad=False)
        hidden = None

        activations = []
        entropies = []
        log_probs = []
        prev_nodes = []

        for block_idx in range(2 * (self.args.num_blocks - 1) + 1):
            logits, hidden = self.forward(inputs, hidden, block_idx)

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -(log_prob * probs).sum(1, keepdim=False)  # ????????

            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(1, utils.get_variable(action, requires_grad=False))

            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])

            # 0: function, 1: previous node
            mode = block_idx % 2
            inputs = utils.get_variable(action[:, 0] + sum(self.num_tokens[:mode]), requires_grad=False)

            if mode == 0:  # function
                activations.append(action[:, 0])
            elif mode == 1:
                prev_nodes.append(action[:, 0])

        prev_nodes = torch.stack(prev_nodes).transpose(0, 1)
        activations = torch.stack(activations).transpose(0, 1)

        dags = _construct_dags(prev_nodes, activations, self.func_names, self.args.num_blocks)

        # 나중에 사용
        if save_dir is not None:
            for idx, dag in enumerate(dags):
                utils.draw_network(dag,
                                   os.path.join(save_dir, f'graph{idx}.png'))

        # 다시 확인
        if with_details:
            return dags, torch.cat(log_probs), torch.cat(entropies)

        return dags



if __name__ == '__main__':
    args, _ = config.get_args()
    print(args.num_gpu, args.cuda)