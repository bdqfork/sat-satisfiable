import torch
import numpy as np
import dgl
import os
import utils


class NpzData(torch.utils.data.Dataset):
    """Read the SAT question from the file and convert it into a graph"""

    def __init__(self, data_path):
        super(NpzData, self).__init__()
        self.cnfs = []
        with open(data_path, 'r') as f:
            data_path = data_path[: data_path.rfind('/')]
            for line in f.readlines():
                line = line.strip()
                line = line[line.rfind('/')+1:]
                line = os.path.join(data_path, line)
                self.cnfs.append(line)
        self.cache_dir = 'cache'

    def __getitem__(self, index):
        cnf_file = self.cnfs[index]
        cache_path = os.path.join(self.cache_dir, cnf_file)+'.pkl'
        if os.path.exists(cache_path):
            return utils.load_cache(cache_path)

        var_num, _, g, y_sat = utils.to_graph(cnf_file, True)

        # get label
        y = np.zeros(2)
        assignment = np.array(y_sat, dtype='float32')
        target = 0 if len(assignment) == 0 else 1
        y[target] = 1

        # if it can be satisfied, return the assignment of all variables
        if target == 1:
            assigns = np.tile(np.zeros(2), (var_num, 1))
            for i in range(var_num):
                assigns[i][y_sat[i]] = 1
        else:
            assigns = None
        utils.save_cache(cache_path, [g, y, assigns])
        return g, y, assigns

    def __len__(self):
        return len(self.cnfs)


class CnfData(torch.utils.data.Dataset):
    """Read the SAT question from the file and convert it into a graph"""

    def __init__(self, data_path):
        super(CnfData, self).__init__()
        self.cnfs = []
        with open(data_path, 'r') as f:
            for line in f.readlines():
                self.cnfs.append(line)
        self.cache_dir = 'cache'

    def __getitem__(self, index):
        cnf_file, y_sat = self.cnfs[index].split(',')
        cache_path = os.path.join(self.cache_dir, cnf_file)+'.pkl'
        if os.path.exists(cache_path):
            return utils.load_cache(cache_path)

        _, _, g = utils.to_graph(cnf_file)

        # get label
        y = np.zeros(2)
        y[int(y_sat)] = 1
        utils.save_cache(cache_path, [g, y])
        return g, y

    def __len__(self):
        return len(self.cnfs)


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels, assigns = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    assigns = list(map(lambda assign: assign if assign is None
                       else torch.FloatTensor(assign), assigns))
    return batched_graph, torch.FloatTensor(labels), assigns


def p_collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.FloatTensor(labels)
