We use Python3, DGL 5.0 and Pytorch 1.6 to implement the model. You can use the command `pip install -r requirements.txt` to install all dependencies. dataset.py is used to read data and convert it into input. train.py builds and trains the Message passing relation model. test.py is used to load model for prediction. If you need to retrain, follow the procedure below to reproduce our experiment.

#### Data preparation
You first need to download data from [End2EndSAT](http://www.cs.ubc.ca/labs/beta/Projects/End2EndSAT/) and unzip it to the data folder. There are 11 datasets from 100 variables to 600 variables. In each dataset, the training set train.txt, the validation set valid.txt and the test set test.txt have been divided.

#### Training
By using the train.py script, the parameters of the training model and the data set path are passed in for training. As the 100-variable dataset, the following command can be used.

```python
python train.py -t data/100/train.txt -v data/100/valid.txt -a model/100 -g 0
```

Among them, **-t** specifies the training set, **-v** specifies the validation set, **-a** indicates the storage path of the model parameters, and **-g 0** indicates the use of **gpu 0**.

If you want to train the **attention model**, you only need to add the **-T** parameter to True. eg.

```python
python train.py -T True -t data/100/train.txt -v data/100/valid.txt -a model/100 -g 0
```

For more training parameters, you can use `python train.py -h` to view detail usage. In this way, we trained 11 models on different datasets.

#### Prediction

By using the test.py script, you can load the trained model parameters to predict and output the accuracy rate on the specified test set. As the 100-variable dataset, the following command can be used.

```python
python test.py -c data/100/test.txt -p model/100/satisfied.pth
```

The **-c** parameter specifies the test set, and **-p** specifies the trained model parameters.

By adjusting the **-p** parameter to load different models and the **-c** parameter to specify different test sets, you can test the performance of the model in terms of accuracy and generalization.

If you want to test the **attention model**, you only need to add the **-T** parameter to True. eg.

```python
python test.py -T True -c data/100/test.txt -p model/100/attetion.pth
```

For more testing parameters, you can use `python test.py -h` to view detail usage. By loading the model training on small-scale data and testing on larger-scale data, the generalization ability of the model can be evaluated.

