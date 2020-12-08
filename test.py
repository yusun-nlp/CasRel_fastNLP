import argparse
import os
import random
import time
from fastNLP import Trainer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config
from data_loader import load_data, get_data_iter
from model import CasRel, MyCallBack
from utils import metric

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='Model Controller')
parser.add_argument('--model_name', type=str, default='CasRel', help='name of the model')
parser.add_argument('--dataset', type=str, default='NYT', help='specify the dataset from ["NYT","WebNLG"]')
parser.add_argument('--bert_model_name', type=str, default='bert-base-cased', help='the name of pretrained bert model')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--multi_gpu', type=bool, default=False, help='if use multiple gpu')
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--test_epoch', type=int, default=1)
parser.add_argument('--max_len', type=int, default=150)
parser.add_argument('--period', type=int, default=50)
args = parser.parse_args()

con = config.Config(args)

# get the data and dataloader
print("-" * 5 + "Starting processing data" + "-" * 5)
data_bundle, rel_vocab, num_rels = load_data(con.train_path, con.dev_path, con.test_path, con.rel_dict_path)
print("Test process data:")
test_data_iter = get_data_iter(con, data_bundle.get_dataset('test'), rel_vocab, num_rels, is_test=True)
print("-" * 5 + "Data processing done" + "-" * 5)

# check the checkpoint dir
if not os.path.exists(con.save_weights_dir):
    os.mkdir(con.save_weights_dir)
# check the log dir
if not os.path.exists(con.save_logs_dir):
    os.mkdir(con.save_logs_dir)


def test():
    model = CasRel(con, num_rels)
    path = os.path.join(con.save_weights_dir, con.weights_save_name)
    model.load_state_dict(torch.load(path))
    model.cuda()
    model.eval()
    precision, recall, f1_score = metric(test_data_iter, rel_vocab, con, model, True)
    print("f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}".format(f1_score, precision, recall))
