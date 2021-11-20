import json
import collections

import time
import random
import os
import numpy as np

import argparse
import logging

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    PreTrainedTokenizer,
    BertConfig,
    BertPreTrainedModel,
    BertTokenizer,
    BertModel,
    RobertaConfig,
    RobertaTokenizer,
    RobertaModel,
    get_linear_schedule_with_warmup,
)

# import tqdm
from tqdm import tqdm, trange
from typing import List

from xpinyin import Pinyin
p_wrong = Pinyin()

from data import convert_examples_to_features, read_text_pair
from model import QuestionMatching

logger = logging.getLogger(__name__)

def set_seed(args):
    """sets random seed"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)

#adversarial training
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1, emb_name = 'word_embeddings'):
        #emb_name should be replaced with the parameter name of embedding in the corresponding model
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name = 'word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def bad_case_analysis(args, preds, labels):
    bad_case = preds ^ labels   #bad_case的列表
    logger.info("**** Bad Case Analysis ****")
    txt_dir = os.path.join(args.output_dir, 'bad_case_analysis.txt')
    data_path = args.dev_set

    count_ = 1 #记录行数
    with open(txt_dir, 'w') as fw:
        with open(data_path, 'r', encoding = 'utf-8') as fr:
            for line in tqdm(fr, desc="Writing bad case"):
                data = line.rstrip().split("\t")
                if len(data) != 3:
                    continue
                query1 = data[0]
                query2 = data[1]
                label = data[2]
                if bad_case[count_ - 1] == 1:
                    fw.write(str(count_) + "\t" + query1 + "\t" + query2 + "\t" + label + "\n")
                count_ += 1

def load_and_cache_examples(args, tokenizer, evaluate = False, test = False, misspelling = False):

    if evaluate:
        cached_mode = "dev"
        data_path = args.dev_set
    elif test:
        cached_mode = "test"
        data_path = args.test_set
    else:
        cached_mode = "train"
        data_path = args.train_set
    assert not (evaluate and test)

    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            cached_mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        logger.info("Loading cached data")
        features = torch.load(cached_features_file)

    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)

        if test:
            examples = read_text_pair(data_path, is_test = True)
        else:
            examples = read_text_pair(data_path)
        logger.info("Training number: %s", str(len(examples)))

        features = convert_examples_to_features(
            examples,
            args.max_seq_length,
            tokenizer
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    if misspelling:
        if test:
            examples = read_text_pair(data_path, is_test = True)
        else:
            examples = read_text_pair(data_path)

        preds = None    #错别字判断
        for (ex_index, example) in tqdm(enumerate(examples), desc="detect misspelling"):
            text_a = example.query1
            text_b = example.query2
            label = int(example.label)

            pinyin_a = p_wrong.get_pinyin(text_a)
            pinyin_b = p_wrong.get_pinyin(text_b)
            if pinyin_a == pinyin_b:
                yin_same = True  #判断是否是同音字
            else:
                yin_same = False

            if preds is None:
                if yin_same:
                    preds = np.array([1])
                else:
                    preds = np.array([0])
            else:
                if yin_same:
                    preds = np.append(preds, [1])
                else:
                    preds = np.append(preds, [0])

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype = torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype = torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype = torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype = torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)
    if misspelling == True:
        return dataset, preds
    else:
        return dataset

def ArgParse():
    parser = argparse.ArgumentParser()

    #Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type ",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    #Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization.\
        Sequences longer than this will be truncated, sequences shorter will be padded."
    )

    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training, evaluation and testing sets"
    )
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")

    parser.add_argument("--train_set", type=str, required=True, help="The full path of train_set_file")
    parser.add_argument("--dev_set", type=str, required=True, help="The full path of dev_set_file")
    parser.add_argument("--test_set", type=str, required=True, help="The full path of test_set_file")
    # parser.add_argument("--save_dir", default='./checkpoint', type=str,
    #                     help="The output directory where the model checkpoints will be written.")

    parser.add_argument("--num_train_epochs", default=1, type=int, help="Total number of training epochs to perform.")
    parser.add_argument('--max_steps', default=-1, type=int, help="If > 0, set total number of training steps to perform.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Number of updates steps to accumulate before preforming a backward/update pass.")

    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proption over the training process.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument('--adam_betas', default='(0.9, 0.999)', type=str, help='betas for Adam optimizer')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--no_clip_grad_norm", action="store_true", help="whether not to clip grad norm")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")

    parser.add_argument("--logging_steps", default=1000, type=int, help="Step interval for evaluation.")
    parser.add_argument('--save_steps', default=10000, type=int, help="Step interval for saving checkpoint.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test on the test set")
    parser.add_argument("--do_misspelling", action="store_true", help="Whether to run do misspelling")

    parser.add_argument("--do_fgm", action="store_true", help="Whether to run Adv-FGM training.")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--rdrop_coef",
        default=0.0,
        type=float,
        help="The coefficient of KL-Divergence loss in R-Drop paper, for more detail please refer to https://arxiv.org/abs/2106.14448), if rdrop_coef > 0 then R-Drop works"
    )

    parser.add_argument("--result_file", type=str, required=True, help="The result file name")

    args = parser.parse_args()
    return args

def evaluate(args, model, tokenizer, test = False, misspelling = False):
    eval_outputs_dirs = (args.output_dir, )
    results = {}

    for eval_output_dir in eval_outputs_dirs:
        if misspelling == True:
            eval_dataset, misspelling_preds = load_and_cache_examples(args, tokenizer, evaluate=not test, test=test, misspelling=misspelling)
        else:
            eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=not test, test=test, misspelling=misspelling)

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler = eval_sampler, batch_size = args.eval_batch_size, num_workers = 0)

        # 开始验证 !
        logger.info("**** Running evaluation ****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc = "Evaluating"):

            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids":batch[0],
                    "attention_mask":batch[1],
                    "token_type_ids":batch[2],
                    "labels":batch[3],
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis = 0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis = 0)


        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis = 1)

        if misspelling is True:
            preds = np.array((preds, misspelling_preds))
            preds = preds.max(axis = 0)


        acc = simple_accuracy(preds, out_label_ids)
        bad_case_analysis(args, preds, out_label_ids)

        result = {"eval_acc":acc, "eval_loss":eval_loss}
        results.update(result)

        #将结果写入文件中
        if not test:
            logger.info("**** Eval results ****")
            txt_dir = os.path.join(args.output_dir, 'dev_result.txt')
            with open(txt_dir, 'w') as f:
                for key in sorted(result.keys()):
                    logger.info(" %s = %s", key, str(result[key]))
                    f.write("%s = %s\n" % (key, str(result[key])))
        elif test:
            logger.info("**** Test results ****")
            txt_dir = os.path.join(args.output_dir, "test_result.txt")
            with open(txt_dir, 'w') as f:
                f.write("no message")

    if test:
        return results, preds
    else:
        return results

def main():

    #一、参数配置

    #读取配置信息参数
    args = ArgParse()

    #Output文件夹是否已经存在
    if(
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome".format(
                args.output_dir
            )
        )

    #训练的设备信息
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()

    # 设置打印的日志信息
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "device: %s, n_gpu: %s, 16-bits training: %s",
        device,
        args.n_gpu,
        args.fp16,
    )

    #设置种子
    set_seed(args)

    #打印训练配置信息
    logger.info("Parameters information: ")
    print(args)

    #二、模型配置信息

    #加载预训练模型的config、tokenizer和model
    config = BertConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels = 2,
        cache_dir = args.cache_dir if args.cache_dir else None,
    )

    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir = args.cache_dir if args.cache_dir else None,
    )

    model = QuestionMatching.from_pretrained(
        args.model_name_or_path,
        from_tf = bool(".ckpt" in args.model_name_or_path),
        config = config,
        rdrop_coef = args.rdrop_coef,
        cache_dir = args.cache_dir if args.cache_dir else None,
    )

    #数据预处理

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    best_steps = 0

    #评估
    if args.do_eval:
        output_dir = args.output_dir
        logger.info("Evaluate the following checkpoints: %s", output_dir)

        model = QuestionMatching.from_pretrained(
            output_dir,
            from_tf = bool(".ckpt" in args.model_name_or_path),
            config = config,
            rdrop_coef=args.rdrop_coef,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        model.to(args.device)
        result = evaluate(args, model, tokenizer)
        result = dict((k, v) for k, v in result.items())
        logger.info(result)


if __name__ == "__main__":
    main()


















