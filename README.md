# Dynamic Memory Networks in TensorFlow

DMN+ implementation in TensorFlow for question answering on the bAbI 10k dataset.

Structure and parameters from [Dynamic Memory Networks for Visual and Textual Question Answering](https://arxiv.org/abs/1603.01417) which is henceforth referred to as Xiong et al.

Adapted from Stanford's [cs224d](http://cs224d.stanford.edu/) assignment 2 starter code and using methods from [Dynamic Memory Networks in Theano](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano) for importing the Babi-10k dataset.

Original code is here: https://github.com/barronalex/Dynamic-Memory-Networks-in-TensorFlow

This code has been modified with GRU cells in answer module

## Repository Contents
| file | description |
| --- | --- |
| `model.py` | contains the DMN+ model |
| `train.py` | trains the model on a specified (-b) babi task|
| `test.py` | tests the model on a specified (-b) babi task |
| `preload.py` | prepares bAbI data for input into DMN |
| `attention_gru_cell.py` | contains a custom Attention GRU cell implementation |
| `fetch_babi_data.sh` | shell script to fetch bAbI tasks (from [DMNs in Theano](https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano)) |


## Set up environment
Install Tensorflow with python 2.7.x (https://www.tensorflow.org/install/)

sudo apt-get install python-pip python-dev python-virtualenv

virtualenv --system-site-packages tensorflow

source ~/tensorflow/bin/activate

pip install --upgrade tensorflow

# optional if you would like to run on gpu
pip install --upgrade tensorflow-gpu 

## Usage


Run the included shell script to fetch the data

	bash fetch_babi_data.sh

Use 'dmn_train.py' to train the DMN+ model contained in 'model.py'

	python train.py --task_id 2

Once training is finished, test the model on a specified task

	python test.py --task_id 2

The l2 regularization constant can be set with -l2-loss (-l). All other parameters were specified by [Xiong et al](https://arxiv.org/abs/1603.01417) and can be found in the 'Config' class in 'model.py'.

## Performance and accuracy are currently tested