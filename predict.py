
import torch
from collections import OrderedDict
from network import GNNSAT, GNNSATAttention
import utils
import time
import psutil
import os
import torch.nn.functional as F


def getMem():
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    mem = info.uss/1024/1024
    return mem


def predict(args):
    start_time = time.time()
    relations = OrderedDict()
    relations[('var', 'clause')] = ('vcp', 'vcn')
    relations[('clause', 'var')] = ('cvp', 'cvn')

    net_args = (128, 2, ('var', 'clause'),
                ('cvp', 'cvn', 'vcp', 'vcn'),
                relations, 9, False)

    if args.attention:
        model = GNNSATAttention(*net_args)
    else:
        model = GNNSAT(*net_args)

    model.load_state_dict(torch.load(args.test_model, map_location='cpu'))
    model.eval()

    var_num, clause_num, g = utils.to_graph(args.input)
    print(f'Variables: {var_num}, Clauses: {clause_num}')
    with torch.no_grad():
        prediction = model(g)
        prediction = F.softmax(prediction[0], dim=0)

        print('SATISFIABLE with probability: {:.4f}'.format(prediction[1]))
        print('UNSATISFIABLE with probability: {:.4f}'.format(
            prediction[0]))

        print(f'CPU time: {round((time.time() - start_time),4)} s')
        print(f'Memory: {round(getMem(),4)} Mb')

    with open(args.output, 'w') as f:
        f.write(f'Variables: {var_num}, Clauses: {clause_num}\n')
        f.write(
            f"{'SATISFIABLE with probability: {: .4f}'.format(prediction[1])}\n")
        f.write(
            f"{'UNSATISFIABLE with probability: {: .4f}'.format(prediction[0])}\n")
        f.write(f'CPU time: {round((time.time() - start_time),4)} s\n')
        f.write(f'Memory: {round(getMem(),4)} Mb\n')


if __name__ == "__main__":
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='data/c_satlab/200/sat_000000.cnf',
                        help='input cnf file')
    parser.add_argument('-o', '--output', type=str, default='out.txt',
                        help='output result')
    parser.add_argument('-p', '--test_model', type=str,
                        default='model/pooling/600/satisfied_2021-01-20_00-44-57_loss_0.8347_acc_0.8652.pth',
                        help='the model path you test')
    parser.add_argument('-T', '--attention', type=ast.literal_eval, default=True,
                        help='use attention (default: True')
    args = parser.parse_args()

    predict(args)
