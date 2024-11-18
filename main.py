import argparse
from dataset import dataset
from model_train import *
from attacks import *
import transformers 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score, f1_score, accuracy_score
import torch
import datasets
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Parsing input arguments and initilizing variables
parser = argparse.ArgumentParser()
parser.add_argument("-dataset", "--dataset", help = "specify a dataset on which an attack is to be performed", default = "IMDb")
parser.add_argument("-torch_num_threads", "--torch-num-threads", help = "number of CPU threads PyTorch uses for parallel operations", default = 1)
parser.add_argument("-model_name", "--model-name", help = "name of the pretrained NLP model", default = "roberta-base")
parser.add_argument("-max_sequence_length", "--max-sequence-length", help = "maximum sequence length", default = 512)
parser.add_argument("-attack_name", "--attack-name", help = "name of an attack to be performed", default = "prediction_loss_based_mia")
parser.add_argument("-num_trials", "--num-trials", help = "number of trials of an attack to be performed", default = 500)
parser.add_argument("-num_epochs", "--num-epochs", help = "number of epochs in the fine-tuning of the model", default = 1)
args = parser.parse_args()

dataset_name = args.dataset 
torch_num_threads = int(args.torch_num_threads)
torch.set_num_threads(torch_num_threads) 
model_name = args.model_name
max_sequence_length = int(args.max_sequence_length)
attack_name = args.attack_name
num_trials = int(args.num_trials)
print(num_trials)
num_epochs = int(args.num_epochs)

print("..loading dataset")
#### loading a dataset
wdataset = dataset(dataset_name)
train_data, test_data = wdataset.train_test_split()

print("..fine-mapping of pretrained NLP model")
#### fine-tuning of pretrained NLP model
# model_name = 'roberta_base'
# max_sequence_length = 512
log_dir = '/nfs/scistore17/robingrp/adepope/DataExtractionAttacks/logfiles'
output_dir = '/nfs/scistore17/robingrp/adepope/DataExtractionAttacks/output'
wmodel = pretrained_models(model_name, max_sequence_length, output_dir, log_dir)
trainer, train_data, test_data = wmodel.fine_tune(train_data, test_data, num_epochs)

print("..performing attacks")
#### performing an attack
# attack_name = 'prediction_loss_based_mia'
# num_trials = 500
wattack = attack(attack_name, num_trials)
mia_metrics = wattack.perform_attack(trainer, train_data, test_data)





