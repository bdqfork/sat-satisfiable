import os
import random
from cnfgen import RandomKCNF
# Number of lits
LIT_LIMIT = (3, 3)
# Number of variables
VAR_LIMIT = (100, 799)
# clause/var
RATE_LIMIT = (1, 10)


def generate(number: int, data_path: str) -> None:
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    count = 0
    import progressbar
    with progressbar.ProgressBar(max_value=number) as bar:
        while count < number:
            filepath = os.path.join(data_path, '{}.cnf'.format(count))
            var_count = random.randint(VAR_LIMIT[0], VAR_LIMIT[1])
            rate = random.uniform(RATE_LIMIT[0], RATE_LIMIT[1])
            clause_count = int(var_count * rate)
            lit_count = random.randint(LIT_LIMIT[0], LIT_LIMIT[1])
            try:
                F = RandomKCNF(lit_count, var_count, clause_count)
                with open(filepath, 'w') as target:
                    target.write(F.dimacs())
                count += 1
                bar.update(count)
            except ValueError:
                pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number', type=int, default=1000,
                        help='how many cnfs you want to generate (default: 1000)')
    parser.add_argument('-p', '--data_path', type=str, default='random3sat',
                        help='where you want to store cnfs (default: random3sat)')
    args = parser.parse_args()
    generate(args.number, args.data_path)
