import json
from random import choice
from fastNLP import RandomSampler, DataSetIter, DataSet, TorchLoaderIter
from fastNLP.io import JsonLoader
from fastNLP import Vocabulary
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from utils import get_tokenizer

BERT_MAX_LEN = 512

tokenizer = get_tokenizer('data/vocab.txt')


def load_data(train_path, dev_path, test_path, rel_dict_path):
    """
    Load dataset from origin dataset using fastNLP
    :param train_path: train data path
    :param dev_path: dev data path
    :param test_path: test data path
    :param rel_dict_path: relation dictionary path
    :return: data_bundle(contain train, dev, test data), rel_vocab(vocabulary of relations), num_rels(number of relations)
    """
    paths = {'train': train_path, 'dev': dev_path, "test": test_path}
    loader = JsonLoader({"text": "text", "triple_list": "triple_list"})
    data_bundle = loader.load(paths)
    print(data_bundle)

    id2rel, rel2id = json.load(open(rel_dict_path))
    rel_vocab = Vocabulary(unknown=None, padding=None)
    rel_vocab.add_word_lst(list(id2rel.values()))
    print(rel_vocab)
    num_rels = len(id2rel)
    print("number of relations: " + str(num_rels))

    return data_bundle, rel_vocab, num_rels


def find_head_idx(source, target):
    """
    Find the index of the head of target in source
    :param source: tokenizer list
    :param target: target subject or object
    :return: the index of the head of target in source, if not found, return -1
    """
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


# def preprocess_data(config, dataset, rel_vocab, num_rels, is_test):
#     token_ids_list = []
#     masks_list = []
#     text_len_list = []
#     sub_heads_list = []
#     sub_tails_list = []
#     sub_head_list = []
#     sub_tail_list = []
#     obj_heads_list = []
#     obj_tails_list = []
#     triple_list = []
#     tokens_list = []
#     for item in range(len(dataset)):
#         # get tokenizer list
#         ins_json_data = dataset[item]
#         text = ins_json_data['text']
#         text = ' '.join(text.split()[:config.max_len])
#         tokens = tokenizer.tokenize(text)
#         if len(tokens) > BERT_MAX_LEN:
#             tokens = tokens[: BERT_MAX_LEN]
#         text_len = len(tokens)
#
#         if not is_test:
#             # build subject to relation_object map, s2ro_map[(sub_head, sub_tail)] = (obj_head, obj_tail, rel_idx)
#             s2ro_map = {}
#             for triple in ins_json_data['triple_list']:
#                 triple = (tokenizer.tokenize(triple[0])[1:-1], triple[1], tokenizer.tokenize(triple[2])[1:-1])
#                 sub_head_idx = find_head_idx(tokens, triple[0])
#                 obj_head_idx = find_head_idx(tokens, triple[2])
#                 if sub_head_idx != -1 and obj_head_idx != -1:
#                     sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
#                     if sub not in s2ro_map:
#                         s2ro_map[sub] = []
#                     s2ro_map[sub].append(
#                         (obj_head_idx, obj_head_idx + len(triple[2]) - 1, rel_vocab.to_index(triple[1])))
#
#             if s2ro_map:
#                 token_ids, segment_ids = tokenizer.encode(first=text)
#                 masks = segment_ids
#                 if len(token_ids) > text_len:
#                     token_ids = token_ids[:text_len]
#                     masks = masks[:text_len]
#                 token_ids = np.array(token_ids)
#                 masks = np.array(masks) + 1
#                 # sub_heads[i]: if index i is the head of any subjects in text
#                 sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
#                 for s in s2ro_map:
#                     sub_heads[s[0]] = 1
#                     sub_tails[s[1]] = 1
#                 # randomly select one subject in text and set sub_head and sub_tail
#                 sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
#                 sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
#                 sub_head[sub_head_idx] = 1
#                 sub_tail[sub_tail_idx] = 1
#                 obj_heads, obj_tails = np.zeros((text_len, num_rels)), np.zeros((text_len, num_rels))
#                 for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
#                     obj_heads[ro[0]][ro[2]] = 1
#                     obj_tails[ro[1]][ro[2]] = 1
#                 token_ids_list.append(token_ids)
#                 masks_list.append(masks)
#                 text_len_list.append(text_len)
#                 sub_heads_list.append(sub_heads)
#                 sub_tails_list.append(sub_tails)
#                 sub_head_list.append(sub_head)
#                 sub_tail_list.append(sub_tail)
#                 obj_heads_list.append(obj_heads)
#                 obj_tails_list.append(obj_tails)
#                 triple_list.append(ins_json_data['triple_list'])
#                 tokens_list.append(tokens)
#         else:
#             token_ids, segment_ids = tokenizer.encode(first=text)
#             masks = segment_ids
#             if len(token_ids) > text_len:
#                 token_ids = token_ids[:text_len]
#                 masks = masks[:text_len]
#             token_ids = np.array(token_ids)
#             masks = np.array(masks) + 1
#             # initialize these variant with 0
#             sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
#             sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
#             obj_heads, obj_tails = np.zeros((text_len, num_rels)), np.zeros((text_len, num_rels))
#             token_ids_list.append(token_ids)
#             masks_list.append(masks)
#             text_len_list.append(text_len)
#             sub_heads_list.append(sub_heads)
#             sub_tails_list.append(sub_tails)
#             sub_head_list.append(sub_head)
#             sub_tail_list.append(sub_tail)
#             obj_heads_list.append(obj_heads)
#             obj_tails_list.append(obj_tails)
#             triple_list.append(ins_json_data['triple_list'])
#             tokens_list.append(tokens)
#
#     data_dict = {'token_ids': token_ids_list,
#                  'marks': masks_list,
#                  'text_len': text_len_list,
#                  'sub_heads': sub_heads_list,
#                  'sub_tails': sub_tails_list,
#                  'sub_head': sub_head_list,
#                  'sub_tail': sub_tail_list,
#                  'obj_heads': obj_heads_list,
#                  'obj_tails': obj_tails_list,
#                  'triple_list': triple_list,
#                  'tokens': tokens_list}
#     process_dataset = DataSet(data_dict)
#     process_dataset.set_input('token_ids', 'marks', 'text_len', 'sub_head', 'sub_tail', 'tokens')
#     process_dataset.set_target('sub_heads', 'sub_tails', 'obj_heads', 'obj_tails', 'triple_list')
#     return process_dataset


class MyDataset(Dataset):
    def __init__(self, config, dataset, rel_vocab, num_rels, is_test):
        self.config = config
        self.dataset = dataset
        self.rel_vocab = rel_vocab
        self.num_rels = num_rels
        self.is_test = is_test
        self.tokenizer = tokenizer

    def __getitem__(self, item):
        """
        The way of reading data, so that we can get data through MyDataset[i]
        :param item: index number
        :return: the item st text attribute in dataset
        """
        # get tokenizer list
        ins_json_data = self.dataset[item]
        text = ins_json_data['text']
        text = ' '.join(text.split()[:self.config.max_len])
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > BERT_MAX_LEN:
            tokens = tokens[: BERT_MAX_LEN]
        text_len = len(tokens)

        if not self.is_test:
            # build subject to relation_object map, s2ro_map[(sub_head, sub_tail)] = (obj_head, obj_tail, rel_idx)
            s2ro_map = {}
            for triple in ins_json_data['triple_list']:
                triple = (self.tokenizer.tokenize(triple[0])[1:-1], triple[1], self.tokenizer.tokenize(triple[2])[1:-1])
                sub_head_idx = find_head_idx(tokens, triple[0])
                obj_head_idx = find_head_idx(tokens, triple[2])
                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                    if sub not in s2ro_map:
                        s2ro_map[sub] = []
                    s2ro_map[sub].append(
                        (obj_head_idx, obj_head_idx + len(triple[2]) - 1, self.rel_vocab.to_index(triple[1])))

            if s2ro_map:
                token_ids, segment_ids = self.tokenizer.encode(first=text)
                masks = segment_ids
                if len(token_ids) > text_len:
                    token_ids = token_ids[:text_len]
                    masks = masks[:text_len]
                token_ids = np.array(token_ids)
                masks = np.array(masks) + 1
                # sub_heads[i]: if index i is the head of any subjects in text
                sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
                for s in s2ro_map:
                    sub_heads[s[0]] = 1
                    sub_tails[s[1]] = 1
                # randomly select one subject in text and set sub_head and sub_tail
                sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
                sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
                sub_head[sub_head_idx] = 1
                sub_tail[sub_tail_idx] = 1
                obj_heads, obj_tails = np.zeros((text_len, self.num_rels)), np.zeros((text_len, self.num_rels))
                for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
                    obj_heads[ro[0]][ro[2]] = 1
                    obj_tails[ro[1]][ro[2]] = 1
                return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, \
                       ins_json_data['triple_list'], tokens
            else:
                return None
        else:
            token_ids, segment_ids = self.tokenizer.encode(first=text)
            masks = segment_ids
            if len(token_ids) > text_len:
                token_ids = token_ids[:text_len]
                masks = masks[:text_len]
            token_ids = np.array(token_ids)
            masks = np.array(masks) + 1
            # initialize these variant with 0
            sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
            sub_head, sub_tail = np.zeros(text_len), np.zeros(text_len)
            obj_heads, obj_tails = np.zeros((text_len, self.num_rels)), np.zeros((text_len, self.num_rels))
            return token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, \
                   ins_json_data['triple_list'], tokens

    def __len__(self):
        return len(self.dataset)


def my_collate_fn(batch):
    """
    Merge data in one batch
    :param batch: the batch size
    :return: a dictionary
    """
    batch = list(filter(lambda x: x is not None, batch))
    batch.sort(key=lambda x: x[2], reverse=True)
    token_ids, masks, text_len, sub_heads, sub_tails, sub_head, sub_tail, obj_heads, obj_tails, triples, tokens = zip(
        *batch)
    cur_batch = len(batch)
    max_text_len = max(text_len)
    batch_token_ids = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_masks = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_sub_heads = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_sub_tails = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_sub_head = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_sub_tail = torch.Tensor(cur_batch, max_text_len).zero_()
    batch_obj_heads = torch.Tensor(cur_batch, max_text_len, 24).zero_()
    batch_obj_tails = torch.Tensor(cur_batch, max_text_len, 24).zero_()

    for i in range(cur_batch):
        batch_token_ids[i, :text_len[i]].copy_(torch.from_numpy(token_ids[i]))
        batch_masks[i, :text_len[i]].copy_(torch.from_numpy(masks[i]))
        batch_sub_heads[i, :text_len[i]].copy_(torch.from_numpy(sub_heads[i]))
        batch_sub_tails[i, :text_len[i]].copy_(torch.from_numpy(sub_tails[i]))
        batch_sub_head[i, :text_len[i]].copy_(torch.from_numpy(sub_head[i]))
        batch_sub_tail[i, :text_len[i]].copy_(torch.from_numpy(sub_tail[i]))
        batch_obj_heads[i, :text_len[i], :].copy_(torch.from_numpy(obj_heads[i]))
        batch_obj_tails[i, :text_len[i], :].copy_(torch.from_numpy(obj_tails[i]))

    return {'token_ids': batch_token_ids,
            'mask': batch_masks,
            'sub_head': batch_sub_head,
            'sub_tail': batch_sub_tail,
            'tokens': tokens}, \
           {'sub_heads': batch_sub_heads,
            'sub_tails': batch_sub_tails,
            'obj_heads': batch_obj_heads,
            'obj_tails': batch_obj_tails,
            'triples': triples,
            }


def get_data_iter(config, dataset, rel_vocab, num_rels, is_test=False, num_workers=0, collate_fn=my_collate_fn):
    """
    Build a data Iterator that combines a dataset and a sampler, and provides single- or multi-process iterators
    over the dataset.
    :param config: configuration
    :param dataset: certain dataset in data bundle processed by fastNLP
    :param rel_vocab: vocabulary of relations
    :param num_rels: the number of relations
    :param is_test: if not test, use RandomSampler; if test, use SequentialSampler
    :param num_workers: how many subprocesses to use for data loading
    :param collate_fn: merges a list of samples to form a mini-batch
    :return: a dataloader
    """
    dataset = MyDataset(config, dataset, rel_vocab, num_rels, is_test)
    # dataset = preprocess_data(config, dataset, rel_vocab, num_rels, is_test)
    # print(dataset)
    if not is_test:
        sampler = RandomSampler()
        data_iter = TorchLoaderIter(dataset=dataset,
                                    collate_fn=collate_fn,
                                    batch_size=config.batch_size,
                                    num_workers=num_workers,
                                    pin_memory=True)
        # data_iter = DataSetIter(dataset=dataset,
        #                         batch_size=config.batch_size,
        #                         num_workers=num_workers,
        #                         sampler=sampler,
        #                         pin_memory=True)
    else:
        data_iter = TorchLoaderIter(dataset=dataset,
                                    collate_fn=collate_fn,
                                    batch_size=1,
                                    num_workers=num_workers,
                                    pin_memory=True)
        # data_iter = DataSetIter(dataset=dataset,
        #                         batch_size=1,
        #                         num_workers=num_workers,
        #                         pin_memory=True)
    return data_iter

# class DataPreFetcher(object):
#     def __init__(self, loader):
#         self.loader = iter(loader)
#         # self.stream = torch.cuda.Stream()
#         self.preload()
#
#     def preload(self):
#         try:
#             self.next_data = next(self.loader)
#         except StopIteration:
#             self.next_data = None
#             return
#         # with torch.cuda.stream(self.stream):
#         #     for k, v in self.next_data.items():
#         #         if isinstance(v, torch.Tensor):
#         #             self.next_data[k] = self.next_data[k].cuda(non_blocking=True)
#
#     def next(self):
#         # torch.cuda.current_stream().wait_stream(self.stream)
#         data = self.next_data
#         self.preload()
#         return data
