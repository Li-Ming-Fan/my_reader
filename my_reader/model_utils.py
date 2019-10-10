
import os
import json

from utils import compute_bleu_rouge
from utils import normalize


#
def do_eval(model, batcher, settings, result_dir=None, result_prefix=None, save_full_info=False):
    """
    """
    pred_answers, ref_answers = [], []
    total_loss, total_num = 0, 0
    
    count = 0
    while True:
        #
        batch = batcher.get_next_batch()  
        if batch is None: break
        #
        results = model.run_eval_one_batch(batch)
        count += 1
        print(count)
        #
        loss = results["loss"]
        idx_passage = results["idx_passage"]
        idx_start = results["idx_start"]
        idx_end = results["idx_end"]
        # pred_prob = results["pred_prob"]
        #
        batch_size = len(idx_passage)
        total_loss += loss * batch_size
        total_num += batch_size
        #
        sidx = 0
        for sidx in range(batch_size):
            #            
            sample = batch['data_raw'][sidx]
            idx_p_curr = idx_passage[sidx]
            idx_s_curr = idx_start[sidx]
            idx_e_curr = idx_end[sidx]
            # prob_curr = pred_prob[sidx]
            #
            pred_a = ''.join(sample['passages'][idx_p_curr]['passage_tokens'][idx_s_curr: idx_e_curr + 1])
            #
            if save_full_info:
                sample['pred_answers'] = [pred_a]
                pred_answers.append(sample)
            else:
                pred_answers.append({'question_id': sample['question_id'],
                                     'question_type': sample['question_type'],
                                     'answers': [ pred_a ],
                                     'entity_answers': [[]],
                                     'yesno_answers': []})
            if 'answers' in sample:
                ref_answers.append({'question_id': sample['question_id'],
                                    'question_type': sample['question_type'],
                                    'answers': sample['answers'],
                                    'entity_answers': [[]],
                                    'yesno_answers': []})
    #
    # saving
    if result_dir is not None and result_prefix is not None:
        result_file = os.path.join(result_dir, result_prefix + '.json')
        with open(result_file, 'w', encoding="utf-8") as fout:
            for pred_answer in pred_answers:
                fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')
        #
        model.logger.info('saving {} results to {}'.format(result_prefix, result_file))
        #
    
    # 
    # metric
    # this average loss is invalid on test set, since we don't have true start_id and end_id
    ave_loss = 1.0 * total_loss / total_num
    #
    if len(ref_answers) > 0:
        pred_dict, ref_dict = {}, {}
        for pred, ref in zip(pred_answers, ref_answers):
            question_id = ref['question_id']
            if len(ref['answers']) > 0:
                pred_dict[question_id] = normalize(pred['answers'])
                ref_dict[question_id] = normalize(ref['answers'])
        #
        bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
    else:
        bleu_rouge = None
    #
    print("ave_loss: %g" % ave_loss)
    print("bleu_rouge:")
    print(bleu_rouge)
    #
    model.logger.info('ave_loss: {}'.format(ave_loss))
    model.logger.info('bleu_rouge: {}'.format(bleu_rouge))
    #
    return ave_loss, bleu_rouge

#
def do_train(model, train_batcher, settings):
    """
    """
    total_num, total_loss = 0, 0
    sect_batch_loss = 0.0
    sect_range = settings.check_period_batch
    #
    loss_best = 10000
    dir_best = os.path.join(settings.base_dir, "model_best")
    if not os.path.exists(dir_best):
        os.makedirs(dir_best)
    #
    # count = 0 
    while True:
        #
        batch = train_batcher.get_next_batch()
        if batch is None: break
        #
        results = model.run_train_one_batch(batch)
        # count += 1
        # print(count)
        #
        loss = results["loss"]
        global_step = results["global_step"]
        lr = results["lr"]
        #
        batch_size = len(batch["batch_questions"])
        total_loss += loss * batch_size
        total_num += batch_size
        #
        sect_batch_loss += loss
        #        
        if global_step % sect_range == 0:
            print("curr, lr, loss: %d, %g, %g" % (global_step, lr, loss) )
            #
            sect_mean_loss = sect_batch_loss / sect_range
            model.logger.info('average loss from batch {} to {} is {}'.format(
                global_step - sect_range + 1, global_step, sect_mean_loss))
            sect_batch_loss = 0
            #
            model.save_ckpt(settings.model_dir, "docqa", global_step)
            if sect_mean_loss < loss_best:
                model.save_ckpt_best(dir_best, "docqa", global_step)
                loss_best = sect_mean_loss
            #
            
    return 1.0 * total_loss / total_num


#
def do_predict(model, batcher, settings, result_dir=None, result_prefix=None, save_full_info=False):
    """
    """
    pred_answers = []
    total_num = 0
    
    count = 0
    while True:
        #
        batch = batcher.get_next_batch()  
        if batch is None: break
        #
        results = model.predict_one_batch_with_pb(batch)
        count += 1
        print(count)
        #
        idx_passage = results[0]
        idx_start = results[1]
        idx_end = results[2]
        # pred_prob = results["pred_prob"]
        #
        batch_size = len(idx_passage)
        total_num += batch_size
        #
        sidx = 0
        for sidx in range(batch_size):
            #            
            sample = batch['data_raw'][sidx]
            idx_p_curr = idx_passage[sidx]
            idx_s_curr = idx_start[sidx]
            idx_e_curr = idx_end[sidx]
            # prob_curr = pred_prob[sidx]
            #
            pred_a = ''.join(sample['passages'][idx_p_curr]['passage_tokens'][idx_s_curr: idx_e_curr + 1])
            #
            if save_full_info:
                sample['pred_answers'] = [pred_a]
                pred_answers.append(sample)
            else:
                pred_answers.append({'question_id': sample['question_id'],
                                     'question_type': sample['question_type'],
                                     'answers': [ pred_a ],
                                     'entity_answers': [[]],
                                     'yesno_answers': []})
    #
    # saving
    if result_dir is not None and result_prefix is not None:
        result_file = os.path.join(result_dir, result_prefix + '.json')
        with open(result_file, 'w', encoding="utf-8") as fout:
            for pred_answer in pred_answers:
                fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')
        #
        model.logger.info('saving {} results to {}'.format(result_prefix, result_file))
        #
    
    #
    print("prediction finished")
    #

