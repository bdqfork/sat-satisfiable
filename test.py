
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from collections import OrderedDict
import random
from network import GNNSAT, GNNSATAttention
from dataset import CnfData, NpzData, collate, p_collate
import dgl


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
                relations, args.step, False, args.use_var)

    if args.attention:
        model = GNNSATAttention(*net_args)
    else:
        model = GNNSAT(*net_args)

    model.load_state_dict(torch.load(args.test_model, map_location='cpu'))
    model.eval()
    with torch.no_grad():
        acc_sum = 0
        for iter, data in enumerate(test_dataloader):
            if args.data_type == 'npz':
                (bg, label, _) = data
            elif args.data_type == 'cnf':
                (bg, label) = data

            prediction = model(bg)

            classifier = torch.argmax(prediction, dim=1).detach().cpu().numpy()
            label = torch.argmax(label, dim=1).detach().numpy()
            acc = accuracy_score(classifier, label)
            acc_sum += acc
            print('Iter {} Test Accuracy {:.4f}'.format(iter+1, acc))
        print('Test Avg Accuracy {:.4f}'.format(
            acc_sum / len(test_dataloader)))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    set_seed(123456)
    import ast
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--attention', type=ast.literal_eval, default=True,
                        help='use attention (default: True')
    parser.add_argument('-c', '--test_path', type=str, default='data/encoded_Quasigroup/test_unsat.txt',
                        help='where you store test data (default: data/satlab/100/test.txt)')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='batch_size (default: 32)')
    parser.add_argument('-p', '--test_model', type=str,
                        default='model/pooling/100/satisfied_2021-01-19_23-53-48_loss_1.0012_acc_0.7598.pth',
                        help='the model path you test')
    parser.add_argument('-t', '--data_type', type=str, default='cnf',
                        help='data type (npz, cnf)')
    parser.add_argument('-z', '--step', type=int, default=9,
                        help='step (default: 9)')
    parser.add_argument('-V', '--use_var', type=ast.literal_eval, default=False,
                        help='use var to predict (default: False)')
    args = parser.parse_args()

    print(args)

    test(args)
