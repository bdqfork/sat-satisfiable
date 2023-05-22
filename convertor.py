from __future__ import print_function, with_statement

from utils import get_all_files
import tqdm


def convert(source, target):
    cnf = list()
    cnf.append(list())
    maxvar = 0

    with open(source, 'r') as f:
        for line in f.readlines():
            tokens = line.split()
            if len(tokens) == 0 or tokens[0] == "p" or tokens[0] == "c":
                continue
            for tok in tokens:
                lit = int(tok)
                maxvar = max(maxvar, abs(lit))
                if lit == 0:
                    cnf.append(list())
                else:
                    cnf[-1].append(lit)

    assert len(cnf[-1]) == 0
    cnf.pop()

    new_cnf = list()
    for clause in cnf:
        while len(clause) > 3:
            new_clause = list()
            for i in range(0, len(clause), 2):
                if i+1 < len(clause):
                    new_cnf.append(list())
                    new_cnf[-1].append(clause[i])
                    new_cnf[-1].append(clause[i+1])
                    maxvar += 1
                    new_cnf[-1].append(-maxvar)
                    new_clause.append(maxvar)
                else:
                    new_clause.append(clause[i])
            clause = new_clause
        new_cnf.append(clause)

    with open(target, 'w') as f:
        f.write(f'p cnf {maxvar} {len(new_cnf)}\n')
        for clause in new_cnf:
            f.write(" ".join([str(lit) for lit in clause]) + " 0\n")


def convert_batch(source_dir, target_dir):
    import os
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    sat_cnfs = get_all_files(source_dir, '.cnf')
    bar = tqdm.tqdm(total=len(sat_cnfs))
    for sat_cnf in sat_cnfs:
        target_3sat_cnf = sat_cnf.replace(source_dir, target_dir)
        convert(sat_cnf, target_3sat_cnf)
        bar.update()
    bar.close()


if __name__ == "__main__":
    convert_batch('data/sat_comp2020_10m', 'data/sat_comp2020_10m_3sat')
