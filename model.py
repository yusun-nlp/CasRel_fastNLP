import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastNLP import Callback
from transformers import *
from utils import metric


class CasRel(nn.Module):
    def __init__(self, config, num_rels):
        super().__init__()
        self.config = config
        self.bert_encoder = BertModel.from_pretrained(self.config.bert_model_name)
        self.bert_dim = 768  # the size of hidden state
        self.sub_heads_linear = nn.Linear(self.bert_dim, 1)
        self.sub_tails_linear = nn.Linear(self.bert_dim, 1)
        self.obj_heads_linear = nn.Linear(self.bert_dim, num_rels)
        self.obj_tails_linear = nn.Linear(self.bert_dim, num_rels)

    def get_encoded_text(self, token_ids, mask):
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.bert_encoder(token_ids, attention_mask=mask)[0]
        return encoded_text

    def get_subs(self, encoded_text):
        """
        Subject Taggers
        :param encoded_text: input sentenced pretrained with BERT
        :return: predicted subject head, predicted object head
        """
        # [batch_size, seq_len, 1]
        pred_sub_heads = self.sub_heads_linear(encoded_text)
        pred_sub_heads = torch.sigmoid(pred_sub_heads)
        # [batch_size, seq_len, 1]
        pred_sub_tails = self.sub_tails_linear(encoded_text)
        pred_sub_tails = torch.sigmoid(pred_sub_tails)
        return pred_sub_heads, pred_sub_tails

    def get_objs_for_specific_sub(self, sub_head_mapping, sub_tail_mapping, encoded_text):
        """
        Relation-specific Object Taggers
        :param sub_head_mapping:
        :param sub_tail_mapping:
        :param encoded_text: input sentenced pretrained with BERT
        :return: predicted object head, predicted object tail
        """
        # [batch_size, 1, bert_dim]
        sub_head = torch.matmul(sub_head_mapping, encoded_text)
        # [batch_size, 1, bert_dim]
        sub_tail = torch.matmul(sub_tail_mapping, encoded_text)
        # [batch_size, 1 bert_dim]
        sub = (sub_head + sub_tail) / 2
        # [batch_size, seq_len, bert_dim]
        encoded_text = encoded_text + sub
        # [batch_size, seq_len, rel_num]
        pred_obj_heads = self.obj_heads_linear(encoded_text)
        pred_obj_heads = torch.sigmoid(pred_obj_heads)
        pred_obj_tails = self.obj_tails_linear(encoded_text)
        pred_obj_tails = torch.sigmoid(pred_obj_tails)
        return pred_obj_heads, pred_obj_tails

    def forward(self, data):
        # [batch_size, seq_len]
        token_ids = data['token_ids']
        # [batch_size, seq_len]
        mask = data['mask']
        # [batch_size, seq_len, bert_dim(768)]
        encoded_text = self.get_encoded_text(token_ids, mask)
        # [batch_size, seq_len, 1]
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_text)
        # [batch_size, 1, seq_len]
        sub_head_mapping = data['sub_head'].unsqueeze(1)
        # [batch_size, 1, seq_len]
        sub_tail_mapping = data['sub_tail'].unsqueeze(1)
        # [batch_size, seq_len, rel_num]
        pred_obj_heads, pred_obj_tails = self.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping,
                                                                        encoded_text)
        return pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails


class MyCallBack(Callback):
    def __init__(self, data_iter, rel_vocab, config):
        super().__init__()
        self.loss_sum = 0
        self.global_step = 0

        self.data_iter = data_iter
        self.rel_vocab = rel_vocab
        self.config = config

    def logging(self, s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with open(os.path.join(self.config.save_logs_dir, self.config.log_save_name), 'a+') as f_log:
                f_log.write(s + '\n')

    # define the loss function
    def loss(self, pred, gold, mask):
        pred = pred.squeeze(-1)
        los = F.binary_cross_entropy(pred, gold, reduction='none')
        if los.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        los = torch.sum(los * mask) / torch.sum(mask)
        return los

    def on_train_begin(self):
        self.best_f1_score = 0
        self.best_precision = 0
        self.best_recall = 0

        self.best_epoch = 0
        self.init_time = time.time()
        self.start_time = time.time()
        print("-" * 5 + "Initializing the model" + "-" * 5)

    def on_epoch_begin(self):
        self.eval_start_time = time.time()

    def on_batch_begin(self, batch_x, batch_y, indices):
        pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails = self.model(batch_x)
        sub_heads_loss = self.loss(batch_y['sub_heads'], pred_sub_heads, batch_x['mask'])
        sub_tails_loss = self.loss(batch_y['sub_tails'], pred_sub_tails, batch_x['mask'])
        obj_heads_loss = self.loss(batch_y['obj_heads'], pred_obj_heads, batch_x['mask'])
        obj_tails_loss = self.loss(batch_y['obj_tails'], pred_obj_tails, batch_x['mask'])
        total_loss = (sub_heads_loss + sub_tails_loss) + (obj_heads_loss + obj_tails_loss)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.loss_sum += total_loss.item()

    def on_epoch_end(self):
        precision, recall, f1_score = metric(self.data_iter, self.rel_vocab, self.config, self.model)
        self.logging('epoch {:3d}, eval time: {:5.2f}s, f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}'.
                     format(self.epoch, time.time() - self.eval_start_time, f1_score, precision, recall))
        if f1_score > self.best_f1_score:
            self.best_f1_score = f1_score
            self.best_epoch = self.epoch
            self.best_precision = precision
            self.best_recall = recall
            self.logging("Saving the model, epoch: {:3d}, best f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}".
                         format(self.best_epoch, self.best_f1_score, precision, recall))
            # save the best model
            path = os.path.join(self.config.save_weights_dir, self.config.weights_save_name)
            torch.save(self.model.state_dict(), path)

    def on_train_end(self):
        self.logging("-" * 5 + "Finish training" + "-" * 5)
        self.logging("best epoch: {:3d}, best f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2}, total time: {:5.2f}s".
                     format(self.best_epoch, self.best_f1_score, self.best_precision, self.best_recall,
                            time.time() - self.init_time))
