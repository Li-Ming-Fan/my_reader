

#import os
import json
import random

from collections import Counter
from collections import namedtuple

from vocab import Vocab

#
def example_generator(filelist, is_train_data, max_p_len, single_pass):
    """
    """
    while True:
        assert filelist, ("Error: Empty filelist") # check filelist isn't empty
        #
        if single_pass:
            filelist = sorted(filelist)
        else:
            random.shuffle(filelist)
        #
        for f in filelist:
            fin = open(f,  "r", encoding="utf-8")
            for lidx, line in enumerate(fin):
                sample = json.loads(line.strip())
                #
                # sample_str = json.dumps(sample, ensure_ascii=False, indent=4)
                # print(sample_str)
                #
                # print(sample.keys())
                # if lidx == 1: break
                """
                dict_keys(['documents', 'answer_spans', 'answer_docs', 'fake_answers',
                           'question', 'segmented_answers', 'answers', 'entity_answers',
                           'segmented_question', 
                           'question_type', 'match_scores', 'fact_or_opinion', 'question_id'])
                """
                #
                if is_train_data:
                    if len(sample['answer_spans']) == 0:
                        continue
                    if sample['answer_spans'][0][1] >= max_p_len:
                        continue
                #
                if 'answer_docs' in sample:
                    sample['answer_passages'] = sample['answer_docs']   # list, passage idx
                #
                # sample['question_tokens']
                # sample['passages']
                #
                sample['question_tokens'] = sample['segmented_question']
                #
                sample['passages'] = []
                for d_idx, doc in enumerate(sample['documents']):
                    if is_train_data:
                        most_related_para = doc['most_related_para']
                        sample['passages'].append(
                            {'passage_tokens': doc['segmented_paragraphs'][most_related_para],
                             'is_selected': doc['is_selected']} )
                        #                        
                    else:
                        para_infos = []
                        for para_tokens in doc['segmented_paragraphs']:
                            question_tokens = sample['segmented_question']
                            common_with_question = Counter(para_tokens) & Counter(question_tokens)
                            correct_preds = sum(common_with_question.values())
                            if correct_preds == 0:
                                recall_wrt_question = 0
                            else:
                                recall_wrt_question = float(correct_preds) / len(question_tokens)
                            para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))
                        #
                        para_infos.sort(key=lambda x: (-x[1], x[2]))
                        #
                        # " 最相近的 1 篇 passage "
                        #
                        para_end_idx = min(1, len(para_infos))
                        #
                        for para_idx in range(para_end_idx):
                            candidate_passage_tokens = para_infos[para_idx][0]
                            sample['passages'].append({'passage_tokens': candidate_passage_tokens})
                        #
                #
                # print(len(sample['passages']))
                #
                yield sample
                #
            # for line
        # for file
        #
        if single_pass:
            print("example_generator completed reading all datafiles. No more data.")
            break
        #

#
def do_batch_std(list_samples, vocab, settings):
    """
    """    
    batch_data = {}
    batch_data["data_raw"] = []
    batch_data["batch_questions_full"] = []
    batch_data["batch_passages_full"] = []
    #
    batch_data["batch_passage_idx"] = []
    batch_data["batch_start_label"] = []
    batch_data["batch_end_label"] = []
    #
    batch_data["batch_question_len"] = []
    batch_data["batch_passage_len"] = []
    #
    max_passage_num = max([len(sample['passages']) for sample in list_samples])
    max_passage_num = min(settings.max_p_num, max_passage_num)
    #
    for sidx, sample in enumerate(list_samples):
        #
        # data_raw
        batch_data["data_raw"].append(sample)
        #
        # question
        sample['question_token_ids'] = vocab.convert_tokens_to_ids(sample['question_tokens'])
        #
        batch_data['batch_questions_full'].append(sample['question_token_ids'])  #
        batch_data["batch_question_len"].append(len(sample['question_token_ids']))
        #
        # passage
        sample_passages = []
        sample_passage_len = []
        for pidx in range(max_passage_num):
            if pidx < len(sample['passages']):
                #
                passage_token_ids = vocab.convert_tokens_to_ids(
                        sample['passages'][pidx]["passage_tokens"])
                sample['passages'][pidx]['passage_token_ids'] = passage_token_ids
                #
                passage_token_ids = sample['passages'][pidx]['passage_token_ids']  #
                sample_passages.append(passage_token_ids)
                sample_passage_len.append(len(passage_token_ids))
                #
            else:
                #
                sample_passages.append([])     #
                sample_passage_len.append(0)   #
                #
        #
        # passage
        batch_data['batch_passages_full'].append(sample_passages)
        batch_data["batch_passage_len"].append(sample_passage_len)
        #
        # label
        if 'answer_passages' in sample and len(sample['answer_passages']):
            batch_data["batch_passage_idx"].append(sample['answer_passages'][0])
            batch_data['batch_start_label'].append(sample['answer_spans'][0][0])
            batch_data['batch_end_label'].append(sample['answer_spans'][0][1])
        else:
            # fake span for some samples, only valid for testing
            batch_data["batch_passage_idx"].append(0)
            batch_data['batch_start_label'].append(0)
            batch_data['batch_end_label'].append(0)
        #
    #
    # paddding
    batch_data = dynamic_padding(batch_data, settings.max_q_len, settings.max_p_len)
    #
    return batch_data

#
def dynamic_padding(batch_data, max_q_len, max_p_len, pad_id=0):
    """
    """
    #
    pad_q_len = min(max_q_len, max(batch_data["batch_question_len"]))
    pad_p_len = min(max_p_len, max([max(item) for item in batch_data["batch_passage_len"]]))
    #
    batch_data['batch_questions'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                       for ids in batch_data['batch_questions_full']]
    batch_data['batch_passages'] = [ [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in passage]
                                           for passage in batch_data["batch_passages_full"] ]
    #
    return batch_data

#
def build_vocab(example_gen, vocab):
    """
    """
    for sample in example_gen:
        vocab.add_tokens_from_corpus([ sample["question_tokens"] ])
        vocab.add_tokens_from_corpus([ item["passage_tokens"] for item in sample["passages"] ])
    #
    return vocab
    

#
if __name__ == "__main__":
    
    file_path = "../data_v2/demo/trainset/search.train.json"
    
    # example
    data_gen = example_generator([ file_path ], True, 100, single_pass = True)
    #
    example = next(data_gen)
    print(example)
    #
    
    # vocab
    vocab = Vocab(lower=True)
    vocab = build_vocab(data_gen, vocab)
    print(vocab.size())
    #
    
    #
    # batch
    data_gen = example_generator([ file_path ], True, 100, single_pass = True)
    #
    list_examples = []
    for idx in range(2):
        example = next(data_gen)
        list_examples.append(example)
    #
    Settings = namedtuple('Settings', ['max_p_num', 'max_p_len', 'max_q_len'])
    settings = Settings(3, 20, 10)
    #
    batch = do_batch_std(list_examples, vocab, settings)
    #
    print(batch)
    #
    
    
