The random3sat folder saves our randomly generated non-UR 3-SAT data.

#### CNFGEN
We use `cnfgen==0.9` to generate data. The usage is as follows：

```
cnfgen randkcnf k n m
```

> Sample a random CNF with m clauses of k literals over n variables. Clauses are sampled without replacement. To get hard UNSAT random 3-CNFs the user should pick about cn clauses where c>4.5. Unfortunately this hardness is asymptotic, therefore n may need to be quite large.

#### Data generation

Since cnfgen can only generate one piece of data at a time, we wrote the `generator.py` script to generate data in batches. The usage is as follows:

```python
python generator.py -n 10000 -p random3sat
```

The `-p` parameter indicates the folder where the generated data is saved. The `-n` parameter indicates how many pieces of data are generated.

#### Data filtering

After the data is generated, use `minisat` to solve the problem. Then we screened out data sets of different scales as data for testing models. The ratio of unsat to sat in each data set is 1:1, 100 for each.