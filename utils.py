import json
import os
import unicodedata
import codecs
from keras_bert import Tokenizer
import numpy as np
import torch


class HBTokenizer(Tokenizer):
    def _tokenize(self, text):
        if not self._cased:
            text = unicodedata.normalize('NFD', text)
            text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
            text = text.lower()
        spaced = ''
        for ch in text:
            if ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []
        for word in spaced.strip().split():
            tokens += self._word_piece_tokenize(word)
            tokens.append('[unused1]')
        return tokens


def get_tokenizer(vocab_path):
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return HBTokenizer(token_dict, cased=True)


def to_tup(triple_list):
    ret = []
    for triple in triple_list:
        ret.append(tuple(triple))
    return ret


def metric(data_iter, rel_vocab, config, model, output=False, h_bar=0.5, t_bar=0.5):
    if output:
        # check the result dir
        if not os.path.exists(config.result_dir):
            os.mkdir(config.result_dir)
        path = os.path.join(config.result_dir, config.result_save_name)
        fw = open(path, 'w')

    orders = ['subject', 'relation', 'object']
    correct_num, predict_num, gold_num = 0, 0, 0

    for batch_x, batch_y in data_iter:
        with torch.no_grad():
            token_ids = batch_x['token_ids']
            tokens = batch_x['tokens']
            mask = batch_x['mask']
            encoded_text = model.get_encoded_text(token_ids, mask)
            pred_sub_heads, pred_sub_tails = model.get_subs(encoded_text)
            sub_heads, sub_tails = np.where(pred_sub_heads.cpu()[0] > h_bar)[0], \
                                   np.where(pred_sub_tails.cpu()[0] > t_bar)[0]
            subjects = []
            for sub_head in sub_heads:
                sub_tail = sub_tails[sub_tails >= sub_head]
                if len(sub_tail) > 0:
                    sub_tail = sub_tail[0]
                    subject = tokens[sub_head: sub_tail]
                    subjects.append((subject, sub_head, sub_tail))

            if subjects:
                triple_list = []
                # [subject_num, seq_len, bert_dim]
                repeated_encoded_text = encoded_text.repeat(len(subjects), 1, 1)
                # [subject_num, 1, seq_len]
                sub_head_mapping = torch.Tensor(len(subjects), 1, encoded_text.size(1)).zero_()
                sub_tail_mapping = torch.Tensor(len(subjects), 1, encoded_text.size(1)).zero_()
                for subject_idx, subject in enumerate(subjects):
                    sub_head_mapping[subject_idx][0][subject[1]] = 1
                    sub_tail_mapping[subject_idx][0][subject[2]] = 1
                sub_tail_mapping = sub_tail_mapping.to(repeated_encoded_text)
                sub_head_mapping = sub_head_mapping.to(repeated_encoded_text)
                pred_obj_heads, pred_obj_tails = model.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping,
                                                                                 repeated_encoded_text)
                for subject_idx, subject in enumerate(subjects):
                    sub = subject[0]
                    sub = ''.join([i.lstrip("##") for i in sub])
                    sub = ' '.join(sub.split('[unused1]'))
                    obj_heads, obj_tails = np.where(pred_obj_heads.cpu()[subject_idx] > h_bar), np.where(
                        pred_obj_tails.cpu()[subject_idx] > t_bar)
                    for obj_head, rel_head in zip(*obj_heads):
                        for obj_tail, rel_tail in zip(*obj_tails):
                            if obj_head <= obj_tail and rel_head == rel_tail:
                                rel = rel_vocab.to_word[int(rel_head)]
                                obj = tokens[obj_head: obj_tail]
                                obj = ''.join([i.lstrip("##") for i in obj])
                                obj = ' '.join(obj.split('[unused1]'))
                                triple_list.append((sub, rel, obj))
                                break
                triple_set = set()
                for s, r, o in triple_list:
                    triple_set.add((s, r, o))
                pred_list = list(triple_set)
            else:
                pred_list = []

            pred_triples = set(pred_list)
            gold_triples = set(to_tup(batch_y['triples'][0]))

            correct_num += len(pred_triples & gold_triples)
            predict_num += len(pred_triples)
            gold_num += len(gold_triples)

            if output:
                result = json.dumps({
                    # 'text': ' '.join(tokens),
                    'triple_list_gold': [
                        dict(zip(orders, triple)) for triple in gold_triples
                    ],
                    'triple_list_pred': [
                        dict(zip(orders, triple)) for triple in pred_triples
                    ],
                    'new': [
                        dict(zip(orders, triple)) for triple in pred_triples - gold_triples
                    ],
                    'lack': [
                        dict(zip(orders, triple)) for triple in gold_triples - pred_triples
                    ]
                }, ensure_ascii=False)
                fw.write(result + '\n')

    print("correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".format(correct_num, predict_num, gold_num))

    precision = correct_num / (predict_num + 1e-10)
    recall = correct_num / (gold_num + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    return precision, recall, f1_score
