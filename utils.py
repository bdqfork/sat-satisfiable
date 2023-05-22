import numpy as np
import pickle
import os


class EarlyStopping(object):
    def __init__(self, save_dir, prefixs, patience=10):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.prefixs = prefixs
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if acc >= self.best_acc:
                self.best_loss = loss
                self.best_acc = acc
                self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def _getFilename(self, prefix):
        return os.path.join(self.save_dir, prefix+'_loss_{:.4f}_acc_{:.4f}.pth'.format(self.best_loss, self.best_acc))

    def save_checkpoint(self, models):
        """Saves model when validation loss decreases."""
        import torch

        if isinstance(models, tuple):
            for i, model in enumerate(models):
                torch.save(model.state_dict(
                ), self._getFilename(self.prefixs[i]))
        else:
            torch.save(models.state_dict(), self._getFilename(self.prefixs))
        print('Save model loss {:.4f}  acc {:.4f}'.format(
            self.best_loss, self.best_acc))

    def load_checkpoint(self, models):
        """Load the latest checkpoint."""
        import torch

        if isinstance(models, tuple):
            for i, model in enumerate(models):
                model.load_state_dict(torch.load(
                    self._getFilename(self.prefixs[i])))
        else:
            models.load_state_dict(torch.load(
                self._getFilename(self.prefixs)))


def get_all_files(path, pattern):
    files = []
    lsdir = os.listdir(path)
    dirs = [i for i in lsdir if os.path.isdir(os.path.join(path, i))]
    if dirs:
        for i in dirs:
            files += get_all_files(os.path.join(path, i), pattern)
    files += [os.path.abspath(os.path.join(path, i)) for i in lsdir if os.path.isfile(
        os.path.join(path, i)) and os.path.splitext(i)[1] == pattern and os.path.getsize(os.path.join(path, i))]
    return files


def load_cache(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as cache:
            return pickle.load(cache)


def save_cache(filepath, data):
    if not os.path.exists(os.path.split(filepath)[0]):
        try:
            os.makedirs(os.path.split(filepath)[0])
        except FileExistsError:
            pass
    with open(filepath, 'wb') as cache:
        pickle.dump(data, cache)


def parse_cnf(cnf_file: str) -> tuple:
    var_num = 0
    clause_num = 0
    indices = []
    values = []
    with open(cnf_file, 'r') as cnf:
        for line in cnf.readlines():
            if line.strip() == '':
                continue
            splited_line = line.split()
            if splited_line[0] == 'c':
                continue
            if splited_line[0] == 'p':
                var_num = int(splited_line[2])
            if splited_line[0] not in ('c', 'p'):
                for var in splited_line:
                    var = int(var)
                    if var == 0:
                        continue
                    value = [0, 0]
                    if var > 0:
                        var = var - 1
                        value[0] = 1
                    if var < 0:
                        var = -var - 1
                        value[1] = 1
                    indices.append((clause_num, var))
                    values.append(value)
                clause_num += 1
    return var_num, clause_num, indices, values


def parse_npz(npz_file: str) -> tuple:
    datas = np.load(npz_file)
    indices, values, y_sat = datas['indices'], datas['values'], datas['y_sat']
    clause_num, var_num = np.max(indices, axis=0) + 1
    return var_num, clause_num, indices, values, y_sat


def to_graph(cnf_file: str, is_npz: bool = False) -> tuple:
    if is_npz:
        var_num, clause_num, indices, values, y_sat = parse_npz(cnf_file)
    else:
        var_num, clause_num, indices, values = parse_cnf(cnf_file)

    # separately count clauses and variables
    clausep, varp = [], []
    clausen, varn = [], []
    for i, (x, y) in enumerate(indices):
        if np.argmax(values[i]) == 0:
            clausep.append(x)
            varp.append(y)
        else:
            clausen.append((x))
            varn.append((y))

    clausep, varp = np.array(clausep), np.array(varp)
    clausen, varn = np.array(clausen), np.array(varn)

    # two kinds of nodes, variable nodes and clause nodes
    ntypes = ('clause', 'var')
    # since DGL has no undirected edges, it uses bidirectional edges to achieve
    etypes = ('cvp', 'cvn', 'vcp', 'vcn')

    import dgl
    import torch

    # build graph
    g = dgl.heterograph({
        (ntypes[0], etypes[0], ntypes[1]): (clausep, varp),
        (ntypes[0], etypes[1], ntypes[1]): (clausen, varn),
        (ntypes[1], etypes[2], ntypes[0]): (varp, clausep),
        (ntypes[1], etypes[3], ntypes[0]): (varn, clausen)})

    for i, ntype in enumerate(ntypes):
        g.nodes[ntype].data['x'] = torch.tensor(
            [i]*g.number_of_nodes(ntype)).long().reshape(-1, 1)

    if is_npz:
        return var_num, clause_num, g, y_sat
    else:
        return var_num, clause_num, g
