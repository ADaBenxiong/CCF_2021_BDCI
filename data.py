import os
import sys
import torch
import torch.nn as nn
import numpy as np
import math

import logging

from typing import List

from transformers import PreTrainedTokenizer
import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

logger = logging.getLogger(__name__)
from xpinyin import Pinyin
p_wrong = Pinyin()

import networkx as nx

from ddparser import DDParser
ddp = DDParser(prob = True, use_pos = True)

class InputExample(object):
    def __init__(self, query1, query2, label = None):
        self.query1 = query1
        self.query2 = query2
        self.label = label

class InputFeatures(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

def read_text_pair(data_path, is_test = False):
    '''Read datas.'''
    count_ = 0
    examples = []
    with open(data_path, 'r', encoding = 'utf-8') as f:
        for line in f:
            # print(line)
            data = line.rstrip().split("\t")
            # print(data)
            if is_test == False:
                if len(data) != 3:
                    continue
                examples.append(
                    InputExample(
                        query1 = data[0],
                        query2 = data[1],
                        label = data[2],
                    )
                )
            else:
                if len(data) != 2:
                    continue
                examples.append(
                    InputExample(
                        query1 = data[0],
                        query2 = data[1],
                        label = 0,
                    )
                )
            # if count_ > 10:
            #     break
            # count_ += 1
    return examples

def convert_examples_to_features(
        examples:List[InputExample],
        max_length:int,
        tokenizer:PreTrainedTokenizer,
) -> List[InputFeatures]:

    features = []
    count_ = 0
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc = "convert examples to features"):
        text_a = example.query1
        text_b = example.query2
        # punct = ['.', '?', '!', ',', '。', '？', '！', '，']
        # if text_a[-1] in punct:
        #     text_a = text_a[:-1]
        # if text_b[-1] in punct:
        #     text_b = text_b[:-1]
        label = int(example.label)

        # out = ddp.parse(text_a)
        # length = len(out[0]['word'])
        # edges = []
        # for idx in range(length):
        #     num_head = out[0]['head'][idx]
        #     if num_head != 0 and out[0]['deprel'][idx] != 'MT':
        #         # edges.append((idx, num_head - 1))
        #         edges.append((out[0]['word'][idx], out[0]['word'][num_head - 1]))
        # graph_a = nx.Graph(edges)
        # out = ddp.parse(text_b)
        # length = len(out[0]['word'])
        # edges = []
        # for idx in range(length):
        #     num_head = out[0]['head'][idx]
        #     if num_head != 0 and out[0]['deprel'][idx] != 'MT':
        #         # edges.append((idx, num_head - 1))
        #         edges.append((out[0]['word'][idx], out[0]['word'][num_head - 1]))
        # graph_b = nx.Graph(edges)

        # if len(graph_a.nodes()) - len(graph_b.nodes()) == -1 and set(graph_a.nodes()) < set(graph_b.nodes()):
        #     print(count_)
        #     print(text_a + "   " + text_b)
        # if len(graph_a.nodes()) - len(graph_b.nodes()) == 1 and set(graph_b.nodes()) < set(graph_a.nodes()):
        #     print(text_a + "   " + text_b)
        #     print(count_)

        if '被' in text_a:
            out = ddp.parse(text_a)

            length = len(out[0]['word'])

            entity = ['n', 'f', 's', 'nz', 'nw', 'r', 'PER', 'LOC', 'ORG', 'TIME']
            subject = []
            object = []

            edges = []
            for idx in range(length):
                num_head = out[0]['head'][idx]
                if num_head != 0:
                    edges.append((idx, num_head - 1))

                if out[0]['postag'][idx] in entity and out[0]['deprel'][idx] == 'SBV':
                    subject.append(idx)
                if out[0]['postag'][idx] in entity and out[0]['deprel'][idx] == 'VOB':
                    object.append(idx)
                if out[0]['postag'][idx] in entity and out[0]['deprel'][idx] == 'POB':
                    object.append(idx)

            graph = nx.Graph(edges)

            for sub in subject:
                for obj in object:

                    set_sub = []
                    set_sub_ = []
                    set_obj = []
                    set_obj_ = []

                    for idx in range(len(out[0]['head'])):
                        if out[0]['head'][idx] - 1 == sub:
                            set_sub.append(idx)
                        if idx == sub:
                            set_sub.append(idx)
                    for i in range(set_sub[0], set_sub[-1] + 1):
                        set_sub_.append(i)

                    for idx in range(len(out[0]['head'])):
                        if out[0]['head'][idx] - 1 == obj:
                            set_obj.append(idx)
                        if idx == obj:
                            set_obj.append(idx)
                    for i in range(set_obj[0], set_obj[-1] + 1):
                        set_obj_.append(i)
                    sub_ = set_sub_[0]
                    obj_ = set_obj_[0]
                    number_path = nx.shortest_path(graph, source=sub, target=obj)
                    token_path = [out[0]['word'][idx] for idx in number_path]
                    token_output = out[0]['word']
                    token_output[number_path[-2]] = token_path[1]
                    token_output[number_path[1]] = ''

                    a = ''.join([out[0]['word'][idx] for idx in set_sub_])
                    b = ''.join([out[0]['word'][idx] for idx in set_obj_])

                    for i in set_sub_:
                        token_output[i] = ''
                    for i in set_obj_:
                        token_output[i] = ''

                    token_output[obj_] = a
                    token_output[sub_] = b

                    if len(number_path) == 4 and '被' in token_path:
                        print(text_a)
                        text_a = ''.join(token_output)
                        print(text_a + '\n')

        if '被' in text_b:
            out = ddp.parse(text_b)

            length = len(out[0]['word'])

            entity = ['n', 'f', 's', 'nz', 'nw', 'r', 'PER', 'LOC', 'ORG', 'TIME']
            subject = []
            object = []

            edges = []
            for idx in range(length):
                num_head = out[0]['head'][idx]
                if num_head != 0:
                    edges.append((idx, num_head - 1))

                if out[0]['postag'][idx] in entity and out[0]['deprel'][idx] == 'SBV':
                    subject.append(idx)
                if out[0]['postag'][idx] in entity and out[0]['deprel'][idx] == 'VOB':
                    object.append(idx)
                if out[0]['postag'][idx] in entity and out[0]['deprel'][idx] == 'POB':
                    object.append(idx)

            graph = nx.Graph(edges)

            for sub in subject:
                for obj in object:

                    set_sub = []
                    set_sub_ = []
                    set_obj = []
                    set_obj_ = []

                    for idx in range(len(out[0]['head'])):
                        if out[0]['head'][idx] - 1 == sub:
                            set_sub.append(idx)
                        if idx == sub:
                            set_sub.append(idx)
                    for i in range(set_sub[0], set_sub[-1] + 1):
                        set_sub_.append(i)

                    for idx in range(len(out[0]['head'])):
                        if out[0]['head'][idx] - 1 == obj:
                            set_obj.append(idx)
                        if idx == obj:
                            set_obj.append(idx)
                    for i in range(set_obj[0], set_obj[-1] + 1):
                        set_obj_.append(i)
                    sub_ = set_sub_[0]
                    obj_ = set_obj_[0]
                    number_path = nx.shortest_path(graph, source=sub, target=obj)
                    token_path = [out[0]['word'][idx] for idx in number_path]
                    token_output = out[0]['word']
                    token_output[number_path[-2]] = token_path[1]
                    token_output[number_path[1]] = ''

                    a = ''.join([out[0]['word'][idx] for idx in set_sub_])
                    b = ''.join([out[0]['word'][idx] for idx in set_obj_])

                    for i in set_sub_:
                        token_output[i] = ''
                    for i in set_obj_:
                        token_output[i] = ''

                    token_output[obj_] = a
                    token_output[sub_] = b

                    if len(number_path) == 4 and '被' in token_path:
                        print(text_b)
                        text_b = ''.join(token_output)
                        print(text_b + '\n')

        bpe_tokens_a = tokenizer.tokenize(text_a)
        bpe_tokens_b = tokenizer.tokenize(text_b)

        bpe_tokens = [tokenizer.cls_token] + bpe_tokens_a + [tokenizer.sep_token] + bpe_tokens_b + [tokenizer.sep_token]

        a_mask = [1] * (len(bpe_tokens_a) + 2) + [0] * (max_length - (len(bpe_tokens_a) + 2))
        b_mask = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1) + [0] * (max_length - len(bpe_tokens))
        a_mask = a_mask[:max_length]
        b_mask = b_mask[:max_length]
        assert isinstance(bpe_tokens, list)

        input_ids = tokenizer.convert_tokens_to_ids(bpe_tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1)

        padding = [0] * (max_length - len(input_ids))
        input_ids += padding
        attention_mask += padding
        token_type_ids += padding

        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        token_type_ids = token_type_ids[:max_length]

        # print(text_a)
        # print(text_b)
        # lenb = sum(idx == 1 for idx in token_type_ids) - 1
        # lena = sum(idx == 1 for idx in attention_mask) - lenb - 3
        # print(input_ids)
        # print(attention_mask)
        # print(token_type_ids)
        # print(lena)
        # print(lenb)
        # print(us)

        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length

        features.append(InputFeatures(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids, label = label))

        count_ += 1
        # if count_ > 100:
        #     break

    return features
