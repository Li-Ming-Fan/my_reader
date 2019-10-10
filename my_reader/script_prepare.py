# -*- coding:utf8 -*-


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import pickle

from vocab import Vocab
from data_utils import example_generator, build_vocab

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
import argparse
def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension')
    #
    parser.add_argument('--mode', choices=['prepare', 'train', 'eval', 'predict'],
                        default = 'train', help='run mode')
    parser.add_argument('--note', type=str,
                        default = "note_something", help='note_something')
    parser.add_argument('--debug', type=int,
                        default = 1, help='debug or not')
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
    parser.add_argument('--vocab_dir', type=str, default= None,
                        help='the dir to save vocabulary')
    parser.add_argument('--emb_dim', type=int, default= 64,
                        help='emb_dim')
    parser.add_argument('--embed_file', type=str, default = None,
                        help='file of pre-trained embeddings')
    parser.add_argument('--token_min_cnt', type=int, default = 5,
                        help='max length of answer')
    #
    return parser.parse_args()


if __name__ == '__main__':
    """
    """
    args = parse_args()
    #
    
    #
    t0 = time.time()
    #
    # data
    dir_vocab_args = args.vocab_dir
    if args.debug == 1:
        train_files = demo_data["train"]
        dev_files = demo_data["dev"]
        test_files = demo_data["test"]
        #
        args.vocab_dir = dir_vocab_demo
    else:
        train_files = data_all["train"]
        dev_files = data_all["dev"]
        test_files = data_all["test"]
        #
        args.vocab_dir = dir_vocab_all
    #
    if dir_vocab_args is not None:
        args.vocab_dir = dir_vocab_args
    #
    # data files
    print('checking the data files...')
    for data_path in train_files + dev_files + test_files:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)
    #
    # directories
    print('preparing the directories...')
    for dir_path in [args.vocab_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    #
    # vocab
    print('building vocabulary ...')
    vocab = Vocab(lower=True)
    #
    filelist = train_files + dev_files + test_files
    example_gen = example_generator(filelist, False, args.max_p_len, single_pass=True)
    #
    vocab = build_vocab(example_gen, vocab)
    #
    unfiltered_vocab_size = vocab.size()
    print("unfiltered_vocab_size: %d" % unfiltered_vocab_size)
    print("token_min_cnt: %d" % args.token_min_cnt)
    #
    vocab.filter_tokens_by_cnt(min_cnt = args.token_min_cnt)
    filtered_num = unfiltered_vocab_size - vocab.size()
    print('after filter {} tokens, the final vocab size is {}'.format(filtered_num, vocab.size()))
    #
    
    #
    # embeddings
    print('assigning embeddings...')
    print("emb_dim: %d" % args.emb_dim)
    if args.embed_file is None:
        print("randomly_init_embeddings")
        vocab.randomly_init_embeddings(args.emb_dim)
    else:
        print("load_pretrained_embeddings")
        vocab.load_pretrained_embeddings(args.embed_file)
    #
    
    #
    # save
    print('saving vocab ...')
    file_vocab_pkl = os.path.join(args.vocab_dir, 'vocab.pkl')
    with open(file_vocab_pkl, 'wb') as fout:
        pickle.dump(vocab, fout)
    #
    tokens_file = os.path.join(args.vocab_dir, "vocab_tokens.txt")
    vocab.save_tokens_to_file(tokens_file)
    #
    t1 = time.time()
    print('done with preparing, cost time: %d' % ( t1-t0 ))
    #
