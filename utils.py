import collections
import numpy as np
import torch
from torch.autograd import Variable
import pygraphviz as pgv
import os


def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out
class keydefaultdict(collections.defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

def to_item(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x

    if float(torch.__version__[0:3]) < 0.4:
        assert (x.dim() == 1) and (len(x) == 1)
        return x[0]

    return x.item()


def batchify(data, bsz, use_cuda):
    # code from https://github.com/pytorch/examples/blob/master/word_language_model/main.py
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if use_cuda:
        data = data.cuda()
    return data

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def draw_network(dag, path):
    makedirs(os.path.dirname(path))
    graph = pgv.AGraph(directed=True, strict=True,
                       fontname='Helvetica', arrowtype='open') # not work?

    checked_ids = [-2, -1, 0]

    if -1 in dag:
        add_node(graph, -1, 'x[t]')
    if -2 in dag:
        add_node(graph, -2, 'h[t-1]')

    add_node(graph, 0, dag[-1][0].name)

    for idx in dag:
        for node in dag[idx]:
            if node.id not in checked_ids:
                add_node(graph, node.id, node.name)
                checked_ids.append(node.id)
            graph.add_edge(idx, node.id)

    graph.layout(prog='dot')
    graph.draw(path)

def add_node(graph, node_id, label, shape='box', style='filled'):
    if label.startswith('x'):
        color = 'white'
    elif label.startswith('h'):
        color = 'skyblue'
    elif label == 'tanh':
        color = 'yellow'
    elif label == 'ReLU':
        color = 'pink'
    elif label == 'identity':
        color = 'orange'
    elif label == 'sigmoid':
        color = 'greenyellow'
    elif label == 'avg':
        color = 'seagreen3'
    else:
        color = 'white'

    if not any(label.startswith(word) for word in  ['x', 'avg', 'h']):
        label = f"{label}\n({node_id})"

    graph.add_node(
            node_id, label=label, color='black', fillcolor=color,
            shape=shape, style=style,
    )