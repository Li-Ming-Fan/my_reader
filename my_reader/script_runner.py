# -*- coding:utf8 -*-


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.append('..')


from vocab import Vocab
from data_batcher import DataBatcher

from data_utils import example_generator, do_batch_std
from model_doc_qa import ModelDocQA

import model_utils


#
data_all = {}
data_all["train"] = [#'../data/demo/trainset/search.train.json',
                     '../data_v2/preprocessed/trainset/search.train.json',
                     '../data_v2/preprocessed/trainset/zhidao.train.json']
data_all["dev"] = [#'../data/demo/devset/search.dev.json',
                   '../data_v2/preprocessed/devset/search.dev.json',
                   '../data_v2/preprocessed/devset/zhidao.dev.json']
data_all["test"] = [#'../data/demo/testset/search.test.json',
                    '../data_v2/preprocessed/testset/search.test1.json',
                    '../data_v2/preprocessed/testset/zhidao.test1.json']
#
demo_data = {}
demo_data["train"] = ['../data_v2/demo/trainset/search.train.json']
demo_data["dev"] = ['../data_v2/demo/devset/search.dev.json']
demo_data["test"] = ['../data_v2/demo/testset/search.test.json']
#
dir_vocab_all = "../data_v2/vocab_all"
dir_vocab_demo = "../data_v2/vocab_demo"
#

#
debug_paras = {}
debug_paras["encd_size"] = 32
debug_paras["att_dim"] = 32
debug_paras["feat_size"] = 32
debug_paras["emb_dim"] = 32
#
debug_paras["max_p_len"] = 32
debug_paras["max_q_len"] = 32
debug_paras["max_a_len"] = 32
#

#
model_paras = {}
model_paras["encd_size"] = 256
model_paras["att_dim"] = 256
model_paras["feat_size"] = 256
model_paras["emb_dim"] = 256
#
model_paras["max_p_len"] = 500
model_paras["max_q_len"] = 60
model_paras["max_a_len"] = 200
#
def assign_paras_from_dict(settings, paras_dict):
    #
    for key in paras_dict:
        settings.__dict__[key] = paras_dict[key]
    #
    print(settings)
    #

#
import argparse
def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension')
    #
    parser.add_argument('--mode', choices=['prepare', 'train', 'eval', 'predict', 'convert'],
                        default = 'predict', help='run mode')
    parser.add_argument('--note', type=str, default = "note_something",
                        help='note_something')
    parser.add_argument('--debug', type=int, default = 1, 
                        help='debug or not')
    #
    parser.add_argument('--gpu', type=str, default = '0',
                        help='specify gpu device')
    parser.add_argument('--gpu_batch_split', type=list, default = [12, 24],
                        help='gpu_batch_split')
    #
    parser.add_argument('--batch_size', type = int, default = 32, 
                        help = 'batch_size')
    parser.add_argument('--check_period', type = int, default = 100, 
                        help = 'check_period')
    #
    parser.add_argument('--restart', type = bool, default = False,
                        help='restart')
    parser.add_argument('--opt', type = str, default = "adam",
                        help='optimizer_type')
    #
    parser.add_argument('--base_dir', default = None,
                        help='the base dir')
    #
    parser.add_argument('--max_p_num', type=int, default = 5,
                        help='max passage num in one sample')
    parser.add_argument('--max_p_len', type=int, default = 500,
                        help='max length of passage')
    parser.add_argument('--max_q_len', type=int, default = 60,
                        help='max length of question')
    parser.add_argument('--max_a_len', type=int, default = 200,
                        help='max length of answer')    
    #
    parser.add_argument('--vocab_dir', default = None,
                        help='vocab_dir')
    parser.add_argument('--emb_dim', type=int, default= 256,
                        help='emb_dim')
    parser.add_argument('--embed_file', default = None,
                        help='embed_file')
    parser.add_argument('--token_min_cnt', type=int, default = 5,
                        help='token_min_cnt')
    #
    return parser.parse_args()

#
if __name__ == '__main__':
    """
    """
    args = parse_args()
    #
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #

    # data
    dir_vocab_args = args.vocab_dir
    dir_base_args = args.base_dir
    if args.debug == 1:
        train_files = demo_data["train"]
        dev_files = demo_data["dev"]
        test_files = demo_data["test"]
        #
        args.vocab_dir = dir_vocab_demo
        args.base_dir = "../task_mrc_demo"
        #
        assign_paras_from_dict(args, debug_paras)
        #
    else:
        train_files = data_all["train"]
        dev_files = data_all["dev"]
        test_files = data_all["test"]
        #
        args.vocab_dir = dir_vocab_all
        args.base_dir = "../task_mrc_all"
        #
        assign_paras_from_dict(args, model_paras)
        #
    #
    if dir_vocab_args is not None:
        args.vocab_dir = dir_vocab_args
    #
    if dir_base_args is not None:
        args.model_dir = dir_base_args
    #
    # data files
    print('checking the data files...')
    for data_path in train_files + dev_files + test_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    #
    # directories
    print('preparing the directories...')
    args.model_dir = os.path.join(args.base_dir, "model")
    args.log_dir = os.path.join(args.base_dir, "log")
    args.result_dir = os.path.join(args.base_dir, "result")
    #
    for dir_path in [args.base_dir, args.model_dir, args.log_dir, args.result_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    #
    # vocab
    print('loading vocabulary ...')
    tokens_file = os.path.join(args.vocab_dir, "vocab_tokens.txt")
    vocab = Vocab(lower=True)
    vocab.add_tokens_from_file(tokens_file)
    #
    print('assigning embeddings...')
    vocab.load_pretrained_embeddings(args.embed_file)
    #
    # more settings
    args.vocab = vocab
    #
    args.keep_prob = 0.7
    #
    args.check_period_batch = args.check_period
    #
    args.learning_rate_base = 0.0005
    args.warmup_steps = 5000
    args.decay_steps = 5000
    args.decay_rate = 0.99
    args.staircase = True
    #
    args.optimizer_type = args.opt # "adam"  # "sgd"  # "adagrad"
    args.momentum = 0.9
    args.adagrad_init_acc = 0.1
    args.rand_unif_init_mag = 0.02
    args.trunc_norm_init_std = 1e-4
    #
    # mode
    if args.mode == "train":
        example_gen = lambda single_pass: example_generator(train_files, True, args.max_p_len, single_pass)
        batch_stder = lambda items: do_batch_std(items, vocab, args)
        batcher = DataBatcher(example_gen, batch_stder, args.batch_size, single_pass=False)
        #
        model = ModelDocQA(args)
        model.prepare_for_train(args.model_dir)
        #
        model_utils.do_train(model, batcher, args)
        model.close_logger()
        #
    elif args.mode == "eval":
        example_gen = lambda single_pass: example_generator(dev_files, True, args.max_p_len, single_pass)
        batch_stder = lambda items: do_batch_std(items, vocab, args)
        batcher = DataBatcher(example_gen, batch_stder, args.batch_size, single_pass=True)  # batch_size
        #
        model = ModelDocQA(args)
        model.prepare_for_train(args.model_dir)
        model.assign_dropout_keep_prob(1.0)
        #
        model_utils.do_eval(model, batcher, args,
                            result_dir = args.result_dir,
                            result_prefix = "eval",
                            save_full_info = False)
        model.close_logger()
        #
    elif args.mode == "predict":
        example_gen = lambda single_pass: example_generator(test_files, False, args.max_p_len, single_pass)
        batch_stder = lambda items: do_batch_std(items, vocab, args)
        batcher = DataBatcher(example_gen, batch_stder, args.batch_size, single_pass=True)
        #
        pb_file = os.path.join(args.model_dir + "_best", "model_frozen.pb")
        model = ModelDocQA(args)
        model.prepare_for_prediction_with_pb(pb_file)
        #
        model_utils.do_predict(model, batcher, args,
                               result_dir = args.result_dir,
                               result_prefix = "pred",
                               save_full_info = False)
        model.close_logger()
        #
    elif args.mode == "convert":
        args.is_train = False
        model = ModelDocQA(args)
        model.load_ckpt_and_save_pb_file(model, args.model_dir + "_best")
        print("load_ckpt_and_save_pb_file() finished")
    else:
        print("args.mode must be [train|eval|predict|convert]")
        #
    
    
        
    
