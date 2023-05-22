import torch.nn.functional as F
import os
import psutil
import time
import utils
import dgl
from dataset import CnfData, NpzData, collate, p_collate
from network import GNNSAT, GNNSATAttention
import random
from collections import OrderedDict
import numpy as np
from sklearn.metrics import accuracy_score
import torch


def visualization(matrix):
    from sklearn.manifold import TSNE
    X = np.array(matrix)
    X_embedded = TSNE(n_components=2, random_state=123456).fit_transform(X)
    return X_embedded


def draw(X_tsne, labels, title):
    mid = int(len(labels) / 2)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # plt.title(title)
    ax.scatter(X_tsne[:mid, 0], X_tsne[:mid, 1],
               marker='o', c='red', label='Sat')
    ax.scatter(X_tsne[mid:, 0], X_tsne[mid:, 1],
               marker='s', c='green', label='Unsat')
    ax.legend()
    # ax.legend()
    plt.savefig(f'pic/{title}.svg', format='svg')
    plt.close()


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True


def test(args):
    if args.data_type == 'npz':
        dataset = NpzData(args.test_path)
        collate_fn = collate
    elif args.data_type == 'cnf':
        dataset = CnfData(args.test_path)
        collate_fn = p_collate
    else:
        raise RuntimeError('unsupport data type '+args.data_type)

    test_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=0,
        shuffle=False, collate_fn=collate_fn)

    relations = OrderedDict()
    relations[('var', 'clause')] = ('vcp', 'vcn')
    relations[('clause', 'var')] = ('cvp', 'cvn')

    net_args = (128, 2, ('var', 'clause'),
                ('cvp', 'cvn', 'vcp', 'vcn'),
                relations, args.step, False, args.use_var, True)

    if args.attention:
        model = GNNSATAttention(*net_args)
    else:
        model = GNNSAT(*net_args)

    model.load_state_dict(torch.load(args.test_model, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(test_dataloader):
            if args.data_type == 'npz':
                (bg, label, _) = data
            elif args.data_type == 'cnf':
                (bg, label) = data
            label = np.argmax(label, axis=1)
            prediction = model(bg)
            embed = visualization(prediction)
            draw(embed, label, 'MPR-150')


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    set_seed(123456)
    import ast
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--attention', type=ast.literal_eval, default=False,
                        help='use attention (default: True')
    parser.add_argument('-c', '--test_path', type=str, default='data/satlab/150/test.txt',
                        help='where you store test data (default: data/satlab/100/test.txt)')
    parser.add_argument('-b', '--batch_size', type=int, default=1000,
                        help='batch_size (default: 32)')
    parser.add_argument('-p', '--test_model', type=str,
                        default='model/sat/100/satisfied_2020-08-14_02-53-28_loss_0.9831_acc_0.7686.pth',
                        help='the model path you test')
    parser.add_argument('-t', '--data_type', type=str, default='npz',
                        help='data type (npz, cnf)')
    parser.add_argument('-z', '--step', type=int, default=9,
                        help='step (default: 9)')
    parser.add_argument('-V', '--use_var', type=ast.literal_eval, default=False,
                        help='use var to predict (default: False)')
    args = parser.parse_args()

    print(args)

    test(args)
