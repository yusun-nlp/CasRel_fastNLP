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
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--test_epoch', type=int, default=5)
parser.add_argument('--max_len', type=int, default=150)
parser.add_argument('--period', type=int, default=50)
args = parser.parse_args()

con = config.Config(args)

# get the data and dataloader
print("-" * 5 + "Starting processing data" + "-" * 5)
data_bundle, rel_vocab, num_rels = load_data(con.train_path, con.dev_path, con.test_path, con.rel_dict_path)
print("Train process data:")
train_data_iter = get_data_iter(con, data_bundle.get_dataset('train'), rel_vocab, num_rels)
print("Dev process data:")
dev_data_iter = get_data_iter(con, data_bundle.get_dataset('dev'), rel_vocab, num_rels, is_test=True)
print("-" * 5 + "Data processing done" + "-" * 5)

# check the checkpoint dir
if not os.path.exists(con.save_weights_dir):
    os.mkdir(con.save_weights_dir)
# check the log dir
if not os.path.exists(con.save_logs_dir):
    os.mkdir(con.save_logs_dir)


# define the loss function
def loss(pred, gold, mask):
    pred = pred.squeeze(-1)
    los = F.binary_cross_entropy(pred, gold, reduction='none')
    if los.shape != mask.shape:
        mask = mask.unsqueeze(-1)
    los = torch.sum(los * mask) / torch.sum(mask)
    return los


def logging(s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(os.path.join(con.save_logs_dir, con.log_save_name), 'a+') as f_log:
            f_log.write(s + '\n')


def train():
    # initialize the model
    print("-" * 5 + "Initializing the model" + "-" * 5)
    model = CasRel(con, num_rels)
    # model.cuda()

    # whether use multi GPU
    if con.multi_gpu:
        model = nn.DataParallel(model)
    model.train()

    # define the optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=con.learning_rate)

    # other
    global_step = 0
    loss_sum = 0

    best_f1_score = 0.0
    best_precision = 0.0
    best_recall = 0.0

    best_epoch = 0
    init_time = time.time()
    start_time = time.time()

    # the training loop
    print("-" * 5 + "Start training" + "-" * 5)
    for epoch in range(con.max_epoch):
        for batch_x, batch_y in train_data_iter:
            if global_step == 20:
                break
            pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails = model(batch_x)
            sub_heads_loss = loss(pred_sub_heads, batch_y['sub_heads'], batch_x['mask'])
            sub_tails_loss = loss(pred_sub_tails, batch_y['sub_tails'], batch_x['mask'])
            obj_heads_loss = loss(pred_obj_heads, batch_y['obj_heads'], batch_x['mask'])
            obj_tails_loss = loss(pred_obj_tails, batch_y['obj_tails'], batch_x['mask'])
            total_loss = (sub_heads_loss + sub_tails_loss) + (obj_heads_loss + obj_tails_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            global_step += 1
            loss_sum += total_loss.item()

            if global_step % con.period == 0:
                cur_loss = loss_sum / con.period
                elapsed = time.time() - start_time
                logging("epoch: {:3d}, step: {:4d}, speed: {:5.2f}ms/b, train loss: {:5.3f}".
                        format(epoch, global_step, elapsed * 1000 / con.period, cur_loss))
                loss_sum = 0
                start_time = time.time()

        if (epoch + 1) % con.test_epoch == 0:
            eval_start_time = time.time()
            model.eval()
            # call the test function
            precision, recall, f1_score = metric(dev_data_iter, rel_vocab, con, model)
            model.train()
            logging('epoch {:3d}, eval time: {:5.2f}s, f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}'.
                    format(epoch, time.time() - eval_start_time, f1_score, precision, recall))

            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_epoch = epoch
                best_precision = precision
                best_recall = recall
                logging("Saving the model, epoch: {:3d}, best f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}".
                        format(best_epoch, best_f1_score, precision, recall))
                # save the best model
                path = os.path.join(con.save_weights_dir, con.weights_save_name)
                torch.save(model.state_dict(), path)

        # manually release the unused cache
        # torch.cuda.empty_cache()

    logging("-" * 5 + "Finish training" + "-" * 5)
    logging("best epoch: {:3d}, best f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2}, total time: {:5.2f}s".
            format(best_epoch, best_f1_score, best_precision, best_recall, time.time() - init_time))


train()

# model = CasRel(con, num_rels)
# # model.cuda()
#
# # define the optimizer
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=con.learning_rate)
#
# Trainer(train_data_iter, model, optimizer, batch_size=con.batch_size, n_epochs=con.max_epoch, print_every=con.period,
#         callbacks=[MyCallBack(dev_data_iter, rel_vocab, con)])
