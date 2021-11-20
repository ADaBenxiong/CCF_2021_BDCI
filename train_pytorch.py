import json
import collections

import time
import random
import os
import numpy as np
import math

import argparse
import logging

import torch
from torch.optim.optimizer import Optimizer
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
from model import QuestionMatching, QuestionMatchingAvaragePooler, QuestionMatchingLast3EmbeddingCls

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

#指数移动平均
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)
                # exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                            N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    # p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(p_data_fp32, alpha = -group['weight_decay'] * group['lr'])

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    # p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p_data_fp32.addcdiv_(exp_avg, denom, value = -step_size * group['lr'])
                else:
                    # p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p_data_fp32.add_(exp_avg, alpha = -step_size * group['lr'])

                p.data.copy_(p_data_fp32)

        return loss

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

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
    # print(all_input_ids)
    # print(all_attention_mask)
    # print(all_token_type_ids)
    # print(all_label_ids)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)
    # print(dataset)
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
    parser.add_argument("--do_misspelling", action="store_true", help="Whether to do misspelling")
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

def evaluate(args, model, tokenizer, ema, test = False, misspelling = False):
    ema.apply_shadow()
    #evaluate
    ema.restore()
    eval_outputs_dirs = (args.output_dir, )
    results = {}

    for eval_output_dir in eval_outputs_dirs:
        if misspelling == True:
            eval_dataset, misspelling_preds = load_and_cache_examples(args, tokenizer, evaluate = not test, test = test, misspelling = misspelling)
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

        if misspelling == True:
            preds = np.array((preds, misspelling_preds))
            preds = preds.max(axis = 0)


        acc = simple_accuracy(preds, out_label_ids)

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

def train(args, train_dataset, model, tokenizer, ema):
    '''Train the model'''

    if args.do_fgm:
        fgm = FGM(model)

    #一、加载训练数据，计算训练批次
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler = train_sampler, batch_size = args.train_batch_size, num_workers = 0)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # 二、设置优化器以及优化器参数的变化
    no_decay = ["bias", "LayerNorm.weight"]
    # 对于bias以及LayerNorm.weight不进行权重衰减，对于其余的权重进行权重衰减
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    # 更新梯度
    exec('args.adam_betas = ' + args.adam_betas)

    #optimizer = RAdam(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-6)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=args.adam_betas, eps=args.adam_epsilon)

    assert not ((args.warmup_steps > 0) and (args.warmup_proportion > 0)), "--only can set one of --warmup_steps and --warm_ratio "
    if args.warmup_proportion > 0:
        args.warmup_steps = int(t_total * args.warmup_proportion)
    # 更新优化器
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    #开始训练
    logger.info("**** Running training ****")
    logger.info("  Num Examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size = %d", args.train_batch_size)
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    def evaluate_model(train_preds, train_label_ids, args, model, tokenizer, best_steps, best_dev_acc, global_step):

        train_preds = np.argmax(train_preds, axis = 1)
        train_acc = simple_accuracy(train_preds, train_label_ids)

        train_preds = None
        train_label_ids = None
        results = evaluate(args, model, tokenizer, ema)
        logger.info(
            "train acc: %s, dev acc: %s, loss: %s, global steps: %s",
            str(train_acc),
            str(results["eval_acc"]),
            str(results["eval_loss"]),
            str(global_step)
        )

        #保存效果更佳的模型
        if results["eval_acc"] >= best_dev_acc:
            best_dev_acc = results["eval_acc"]
            best_steps = global_step
            logger.info("achieve BEST dev acc: %s at global step: %s", str(best_dev_acc), str(best_steps))

            #Save the model with the best validation set
            output_dir = args.output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )
            model_to_save.save_pretrained(output_dir)
            # tokenizer.save_vocabulary(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)

            # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            # logger.info("Saving optimizer and scheduler states to %s", output_dir)

            txt_dir = os.path.join(output_dir, 'best_dev_results.txt')
            with open(txt_dir, 'w') as f:
                rs = 'global_steps: {}; dev_acc: {}'.format(global_step, best_dev_acc)
                f.write(rs)

        return train_preds, train_label_ids, train_acc, best_steps, best_dev_acc

    def save_model(args, model, tokenizer):
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )
        model_to_save.save_pretrained(output_dir)   #save config.json, pytorch_model.bin 文件
        # tokenizer.save_vocabulary(output_dir)   #save vocab.txt 文件
        tokenizer.save_pretrained(output_dir)   #save added_tokens.json, special_token_map.json, tokenizer_config.json, vocab.txt 文件
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

        # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler,pt"))
        # logger.info("Saving optimizer and scheduler states to %s", output_dir)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0    #平均loss， 打印loss
    best_dev_acc = 0.0
    best_steps = 0
    train_preds = None  #预测的结果
    train_label_ids = None  #真实的标签

    model.zero_grad()   #将参数的梯度设置为0
    train_iterator = trange(int(args.num_train_epochs), desc = "Epoch", disable=False)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc = "Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids":batch[0],
                "attention_mask":batch[1],
                "token_type_ids":batch[2],
                "labels":batch[3],
            }
            outputs = model(**inputs)

            loss = outputs[0]
            logits = outputs[1]

            if train_preds is None:
                train_preds = logits.detach().cpu().numpy()
                train_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                train_preds = np.append(train_preds, logits.detach().cpu().numpy(), axis = 0)
                train_label_ids = np.append(train_label_ids, inputs['labels'].detach().cpu().numpy(), axis = 0)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            if not args.no_clip_grad_norm:
                if not args.do_fgm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()

            if args.do_fgm:
                fgm.attack()
                outputs_adv = model(**inputs)
                loss_adv =outputs_adv[0]
                loss_adv = loss_adv.mean() / args.gradient_accumulation_steps
                loss_adv.backward()
                fgm.restore()

            #gradient_accumulation_step
            if (step + 1) % args.gradient_accumulation_steps == 0:

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                ema.update()

                #logging
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:

                    train_preds, train_labels_ids, train_acc, best_steps, best_dev_acc = \
                        evaluate_model(train_preds, train_label_ids, args, model, tokenizer, best_steps, best_dev_acc, global_step)
                    logger.info(
                        "Average loss: %s, average acc: %s at global step: %s",
                        str((tr_loss - logging_loss) / args.logging_steps),
                        str(train_acc),
                        str(global_step),
                    )
                    logging_loss = tr_loss

                #save model
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_model(args, model, tokenizer)


            #Reach the maximum number of steps and jump out of the loop
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    train_preds, train_label_ids, train_acc, best_steps, best_dev_acc = \
        evaluate_model(train_preds, train_label_ids, args, model, tokenizer, best_steps, best_dev_acc, global_step)
    save_model(args, model, tokenizer)

    return global_step, tr_loss / max(global_step, 1), best_steps

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

    # print(args.device)
    # print(args.n_gpu)

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
    print("*" * 100)


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
    #model = QuestionMatchingLast3EmbeddingCls.from_pretrained(
        args.model_name_or_path,
        from_tf = bool(".ckpt" in args.model_name_or_path),
        config = config,
        rdrop_coef = args.rdrop_coef,
        cache_dir = args.cache_dir if args.cache_dir else None,
    )

    # print("*" * 100)
    # logger.info("model information:")
    # print(config)
    # print(tokenizer)
    # print(model)
    # print("*" * 100)

    #数据预处理

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    best_steps = 0

    #模型配置完成、开始进行训练
    #训练
    if args.do_train:
        ema = EMA(model, 0.999)  # 使用了EMA
        ema.register()
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate = False)
        global_step, tr_loss, best_steps = train(args, train_dataset, model, tokenizer, ema)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    #评估
    if args.do_eval:
        output_dir = args.output_dir
        logger.info("Evaluate the following checkpoints: %s", output_dir)

        model = QuestionMatching.from_pretrained(
        #model=QuestionMatchingLast3EmbeddingCls.from_pretrained(
            output_dir,
            from_tf = bool(".ckpt" in args.model_name_or_path),
            config = config,
            rdrop_coef=args.rdrop_coef,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        model.to(args.device)
        ema = EMA(model, 0.999)  # 使用了EMA
        ema.register()

        result = evaluate(args, model, tokenizer, ema)
        result = dict((k, v) for k, v in result.items())
        logger.info(result)

    #测试
    if args.do_test:
        checkpoint_dir = os.path.join(args.output_dir)
        if best_steps:
            logger.info("best steps of eval acc is the following checkpoints: %s", best_steps)

        model = QuestionMatching.from_pretrained(
        # model = QuestionMatchingLast3EmbeddingCls.from_pretrained(
            checkpoint_dir,
            from_tf = bool(".ckpt" in args.model_name_or_path),
            config = config,
            rdrop_coef = args.rdrop_coef,
            cache_dir = args.cache_dir if args.cache_dir else None,
        )
        model.to(args.device)
        ema = EMA(model, 0.999)  # 使用了EMA
        ema.register()

        result, preds = evaluate(args, model, tokenizer, ema, test = True, misspelling = args.do_misspelling)

        result = dict((k, v) for k, v in result.items())
        logger.info(result)

        with open(os.path.join(args.output_dir, args.result_file), "w", encoding = "utf-8") as f:
            for pred in preds:
                f.write(str(pred) + "\n")

if __name__ == "__main__":
    main()


















