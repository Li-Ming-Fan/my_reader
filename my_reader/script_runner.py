# -*- coding:utf8 -*-


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.append('..')


from Zeras.vocab import Vocab
from Zeras.data_batcher import DataBatcher

from data_utils import example_generator, do_batch_std
from model_doc_qa import ModelDocQA
from model_settings import ModelSettings

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
    parser.add_argument('--mode', choices=['train', 'eval', 'predict', 'convert'],
                        default = 'train', help='run mode')
    parser.add_argument('--note', type=str, default = "note_something",
                        help='note_something')
    parser.add_argument('--debug', type=int, default = 1, 
                        help='debug or not')
    #
    parser.add_argument('--gpu', type=str, default = '0',
                        help='specify gpu device')
    #
    parser.add_argument('--restart', type = bool, default = False,
                        help='restart')
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

    # settings
    settings = ModelSettings()
    settings.gpu_available = args.gpu

    # data
    if args.debug == 1:
        train_files = demo_data["train"]
        dev_files = demo_data["dev"]
        test_files = demo_data["test"]
        #
        settings.tokens_file = os.path.join(dir_vocab_demo, "vocab_tokens.txt")
        settings.base_dir = "../task_mrc_demo"
        #
        assign_paras_from_dict(settings, debug_paras)
        #
    else:
        train_files = data_all["train"]
        dev_files = data_all["dev"]
        test_files = data_all["test"]
        #
        settings.tokens_file = os.path.join(dir_vocab_all, "vocab_tokens.txt")
        settings.base_dir = "../task_mrc_all"
        #
        assign_paras_from_dict(settings, model_paras)
        #
    #
    # data files
    print('checking the data files...')
    for data_path in train_files + dev_files + test_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    #
    # vocab
    print('loading vocabulary ...')    
    vocab = Vocab(lower=True)
    vocab.add_tokens_from_file(settings.tokens_file)
    #
    print('assigning embeddings...')
    vocab.load_pretrained_embeddings(settings.emb_file)
    #
    # more settings
    settings.adagrad_init_acc = 0.1
    settings.rand_unif_init_mag = 0.02
    settings.trunc_norm_init_std = 1e-4
    #
    # model & vocab
    settings.model_tag = "multidocqa"
    settings.vocab = vocab
    #
    # mode
    if args.mode == "train":
        example_gen = lambda single_pass: example_generator(train_files, True, settings.max_p_len, single_pass)
        batch_stder = lambda items: do_batch_std(items, vocab, settings)
        batcher = DataBatcher(example_gen, batch_stder, settings.batch_size, single_pass=False)
        #
        settings.is_train = True
        settings.check_settings()
        #
        model = ModelDocQA(settings)
        model.prepare_for_train_and_valid(settings.model_dir)
        #
        model_utils.do_train(model, batcher, settings)
        model.close_logger()
        #
    elif args.mode == "eval":
        example_gen = lambda single_pass: example_generator(dev_files, True, settings.max_p_len, single_pass)
        batch_stder = lambda items: do_batch_std(items, vocab, settings)
        batcher = DataBatcher(example_gen, batch_stder, settings.batch_size, single_pass=True)  # batch_size
        #
        settings.is_train = True
        settings.check_settings()
        settings.create_or_reset_log_file()
        #
        model = ModelDocQA(settings)
        model.prepare_for_train_and_valid(settings.model_dir)
        model.assign_dropout_keep_prob(1.0)
        #
        model_utils.do_eval(model, batcher, settings,
                            result_dir = settings.result_dir,
                            result_prefix = "eval",
                            save_full_info = False)
        model.close_logger()
        #
    elif args.mode == "predict":
        example_gen = lambda single_pass: example_generator(test_files, False, settings.max_p_len, single_pass)
        batch_stder = lambda items: do_batch_std(items, vocab, settings)
        batcher = DataBatcher(example_gen, batch_stder, settings.batch_size, single_pass=True)
        #
        settings.is_train = False
        settings.check_settings()
        settings.create_or_reset_log_file()
        #
        pb_file = os.path.join(settings.model_dir + "_best", "model_frozen.pb")
        model = ModelDocQA(settings)
        model.prepare_for_prediction_with_pb(pb_file)
        #
        model_utils.do_predict(model, batcher, settings,
                               result_dir = settings.result_dir,
                               result_prefix = "pred",
                               save_full_info = False)
        model.close_logger()
        #
    elif args.mode == "convert":
        settings.is_train = False
        settings.check_settings()
        settings.create_or_reset_log_file()
        #
        model = ModelDocQA(settings)
        ModelDocQA.load_ckpt_and_save_pb_file(model, settings.model_dir + "_best")
        print("load_ckpt_and_save_pb_file() finished")
    else:
        print("args.mode must be [train|eval|predict|convert]")
        #
    
    
        
    
