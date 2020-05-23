# Create time 02/22/2019
# Author: CHEN Wang, CSE, CUHK
# Email: wchen@cse.cuhk.edu.hk
# The code is revised from https://github.com/memray/seq2seq-keyphrase/blob/master/keyphrase/keyphrase_utils.py

import math
import string
import re
import json
import argparse
import time
import numpy as np
from nltk.stem.porter import *
from data_utils import in_context, ken_in_context
from onmt.utils.logging import init_logger, logger
from data_utils import KEY_SEPERATOR, P_START, A_START, P_END, A_END


def evaluate_func(opts, do_stem=True):
    """
    calculate the macro-averaged and micro-averaged precesions, recalls and F1 scores
    """
    # print out the arguments
    logger.info("Parameters:")
    logger.info("do_stem: {}".format(do_stem))
    for k, v in opts.__dict__.items():
        logger.info("{}: {}".format(k, v))

    context_file = open(opts.src, encoding='utf-8')
    context_lines = context_file.readlines()

    target_file = open(opts.tgt, encoding='utf-8')
    target_lines = target_file.readlines()

    preds_file = open(opts.output, encoding='utf-8')
    preds_lines = preds_file.readlines()

    # the number of examples should be the same
    assert len(context_lines) == len(preds_lines)
    assert len(preds_lines) == len(target_lines)

    stemmer = PorterStemmer()

    # collect the statistics of the number of present and absent predictions
    num_present_preds_statistics = [0] * 51
    num_absent_preds_statistics = [0] * 51

    macro_metrics = {'total': [], 'present': [], 'absent': []}
    cnt = 1
    present_correctly_matched_at = {'5': [], '10': [], 'M': []}
    absent_correctly_matched_at = {'5': [], '10': [], 'M': []}
    total_correctly_matched_at = {'5': [], '10': [], 'M': []}

    present_target_nums_list = []
    absent_target_nums_list = []
    total_target_nums_list = []

    present_preds_nums_list = []
    absent_preds_nums_list = []
    total_preds_nums_list = []

    dupRatio_list = []
    p_dupRatio_list = []
    a_dupRatio_list = []
    logger.info("Only evaluate the top {} predictions".format(opts.preds_cutoff_num))

    for context, targets, preds in zip(context_lines, target_lines, preds_lines):
        # preprocess predictions and targets to a list ['key1a key1b', 'key2a key2b']

        # still use preprocessing steps
        # targets = process_keyphrase(targets.strip(), limit_num=False, fine_grad=True, replace_digit=False)
        # preds = preds.replace(KEY_SEPERATOR, ';')
        # preds = process_keyphrase(preds.strip(), limit_num=False, fine_grad=True, replace_digit=False)
        #
        # # preprocess context in a fine-gradularity: [word1, word2,..., wordk,...]
        # context = ' '.join(get_tokens(context, fine_grad=True, replace_digit=False))

        # do not use preprocessing steps
        targets = targets.replace('/', KEY_SEPERATOR)
        targets = targets.replace('/', KEY_SEPERATOR)
        targets = targets.replace("<eokp>", '')
        targets = targets.replace(P_START, KEY_SEPERATOR)
        targets = targets.replace(A_START, KEY_SEPERATOR)
        targets = targets.replace('<blank>', '')
        targets = targets.replace(P_END, KEY_SEPERATOR)
        targets = targets.replace(A_END, KEY_SEPERATOR)

        targets = targets.split(KEY_SEPERATOR)
        targets = [tmp_key.strip() for tmp_key in targets if len(tmp_key.strip()) != 0]
        #targets = [tmp_key.strip() if len(tmp_key.strip()) != 0 else "empty present kp placeholder" for tmp_key in targets]

        # calculate no stem dup ratios
        orig_preds = [tmp_key.strip() for tmp_key in preds.split(KEY_SEPERATOR) if len(tmp_key.strip()) != 0]
        p_orig_preds = {}
        a_orig_preds = {}
        for tmp_key in orig_preds:
            if P_START in tmp_key:
                if tmp_key in p_orig_preds:
                    p_orig_preds[tmp_key] += 1
                else:
                    p_orig_preds[tmp_key] = 1
            elif A_START in tmp_key:
                if tmp_key in a_orig_preds:
                    a_orig_preds[tmp_key] += 1
                else:
                    a_orig_preds[tmp_key] = 1
        if len(p_orig_preds) > 0:
            p_total_num = sum([p_orig_preds[key] for key in p_orig_preds])
            p_dupRatio = (p_total_num - len(p_orig_preds)) * 1.0 / p_total_num
            p_dupRatio_list.append(p_dupRatio)
        if len(a_orig_preds) > 0:
            a_total_num = sum([a_orig_preds[key] for key in a_orig_preds])
            a_dupRatio = (a_total_num - len(a_orig_preds)) * 1.0 / a_total_num
            a_dupRatio_list.append(a_dupRatio)

        # remove eokp token
        preds = preds.replace('/', KEY_SEPERATOR)
        preds = preds.replace("<eokp>", '')
        preds = preds.replace(P_START, KEY_SEPERATOR)
        preds = preds.replace(A_START, KEY_SEPERATOR)
        preds = preds.replace('<blank>', '')
        preds = preds.replace(P_END, KEY_SEPERATOR)
        preds = preds.replace(A_END, KEY_SEPERATOR)

        preds = preds.split(KEY_SEPERATOR)
        preds = [tmp_key.strip() for tmp_key in preds if len(tmp_key.strip()) != 0]
        #preds = [tmp_key.strip() if len(tmp_key.strip()) != 0 else "empty present kp placeholder" for tmp_key in preds]

        # if len(preds) > 0:
        # cut off the preds
        if opts.preds_cutoff_num > 0:
            preds = preds[:opts.preds_cutoff_num]

        # stem words in context, target, pred, if needed
        if do_stem:
            context = ' '.join([stemmer.stem(w) for w in context.strip().split()])
            # the gold keyphrases of SemEval testing dataset are already stemmed
            if 'semeval' in opts.tgt.lower():
                targets = [keyphrase for keyphrase in targets if keyphrase != opts.ap_splitter]
            else:
                targets = [' '.join([stemmer.stem(w) for w in keyphrase.split()]) for keyphrase in targets if keyphrase != opts.ap_splitter]

            preds = [' '.join([stemmer.stem(w) for w in keyphrase.split()]) for keyphrase in preds if keyphrase != opts.ap_splitter]
        else:
            context = context.strip()

        if opts.filter_dot_comma_unk:
            # preds = [keyphrase for keyphrase in preds if ',' not in keyphrase and '.' not in keyphrase and '<unk>' not in keyphrase]
            # currently we only remove the keyphrase with <unk> token
            preds = [keyphrase for keyphrase in preds if '<unk>' not in keyphrase]

        context_list = context.split()

        # get the present_tgt_keyphrase, absent_tgt_keyphrase
        present_tgt_set = set()
        absent_tgt_set = set()
        total_tgt_set = set(targets)
        total_tgt_list = []
        for tgt in targets:
            # check whether the tgt is in context
            if opts.match_method == 'word_match':
                tgt_list = tgt.split()
                match = in_context(context_list, tgt_list)
            else:
                match = tgt in context
            # put in the corresponding tgt set
            if match:
                if tgt not in present_tgt_set:
                    total_tgt_list.append(tgt)
                    present_tgt_set.add(tgt)
            else:
                if tgt not in absent_tgt_set:
                    total_tgt_list.append(tgt)
                    absent_tgt_set.add(tgt)

        # store the nums of tgt
        present_target_nums_list.append(len(present_tgt_set))
        absent_target_nums_list.append(len(absent_tgt_set))
        total_target_nums_list.append(len(total_tgt_set))

        # get the present_pred_keyphrase, absent_pred_keyphrase
        present_preds = []
        present_preds_set = set()
        absent_preds = []
        absent_preds_set = set()
        total_preds = []

        single_word_maxnum = opts.single_word_maxnum
        # split to present and absent predictions and also delete the repeated predictions
        for pred in preds:
            # # only keep single_word_maxnum single word keyphrases
            # single_word_maxnum = -1 means we keep all the single word phrase
            if single_word_maxnum != -1 and len(pred.split()) == 1:
                if single_word_maxnum > 0:
                    single_word_maxnum -= 1
                else:
                    continue
            if opts.match_method == 'word_match':
                match = in_context(context_list, pred.split())
                ken_match = ken_in_context(context_list, [pred.split()])
                if match != ken_match:
                    logger.info("{} is not consistently recognized as a present key.".format(pred))
            else:
                match = pred in context
            # put in the corresponding preds set
            if match:
                if pred not in present_preds_set:
                    total_preds.append(pred)
                    present_preds.append(pred)
                    present_preds_set.add(pred)
            else:
                if pred not in absent_preds_set:
                    total_preds.append(pred)
                    absent_preds.append(pred)
                    absent_preds_set.add(pred)

        # store the nums of preds
        present_preds_nums_list.append(len(present_preds))
        absent_preds_nums_list.append(len(absent_preds))
        total_preds_nums_list.append(len(total_preds))

        if len(preds) > 0:
            dupRatio = (len(preds) - len(total_preds)) * 1.0 / len(preds)
            dupRatio_list.append(dupRatio)

        if len(present_preds_set) < 50:
            num_present_preds_statistics[len(present_preds_set)] += 1
        else:
            num_present_preds_statistics[50] += 1
        if len(absent_preds_set) < 50:
            num_absent_preds_statistics[len(absent_preds_set)] += 1
        else:
            num_absent_preds_statistics[50] += 1

        # get the total_correctly_matched indicators
        total_correctly_matched = [1 if total_pred in total_tgt_set else 0 for total_pred in total_preds]
        # get the present_correctly_matched indicators
        present_correctly_matched = [1 if present_pred in present_tgt_set else 0 for present_pred in present_preds]
        # get the absent_correctly_matched indicators
        absent_correctly_matched = [1 if absent_pred in absent_tgt_set else 0 for absent_pred in absent_preds]

        # macro metric calculating
        macro_metrics['total'].append(
            macro_metric_fc(total_tgt_set, total_correctly_matched))
        macro_metrics['present'].append(
            macro_metric_fc(present_tgt_set, present_correctly_matched))
        macro_metrics['absent'].append(
            macro_metric_fc(absent_tgt_set, absent_correctly_matched))

        cnt += 1

        if cnt % 1000 == 0:
            logger.info('{} papers evaluation complete!'.format(cnt))

    # compute the corpus evaluation
    # macro_ave_fc(micro_metrics['total'], keyphrase_type='total')
    num_ex = len(total_target_nums_list)
    logger.info('# Total tgt num=%d' % sum(total_target_nums_list))
    logger.info('# Present tgt num=%d' % sum(present_target_nums_list))
    logger.info('# Absent tgt num=%d' % sum(absent_target_nums_list))
    logger.info('# ave Total tgt num=%.2f' % (sum(total_target_nums_list) * 1.0 / num_ex))
    logger.info('# ave Present tgt num=%.2f' % (sum(present_target_nums_list) * 1.0 / num_ex))
    logger.info('# ave Absent tgt num=%.2f' % (sum(absent_target_nums_list) * 1.0 / num_ex))

    logger.info('# Total preds num=%d' % sum(total_preds_nums_list))
    logger.info('# Present preds num=%d' % sum(present_preds_nums_list))
    logger.info('# Absent preds num=%d' % sum(absent_preds_nums_list))
    logger.info('# ave Total preds num=%.2f' % (sum(total_preds_nums_list) * 1.0 / num_ex))
    logger.info('# ave Present preds num=%.2f' % (sum(present_preds_nums_list) * 1.0 / num_ex))
    logger.info('# ave Absent preds num=%.2f' % (sum(absent_preds_nums_list) * 1.0 / num_ex))

    logger.info('# ex num={}, ave DupRatio={:.5f}'.format(len(dupRatio_list), (sum(dupRatio_list) * 1.0 / len(dupRatio_list)) if len(dupRatio_list) != 0 else 0.0))
    logger.info(
        '# present ex num={}, ave p_DupRatio={:.5f}'.format(len(p_dupRatio_list), (sum(p_dupRatio_list) * 1.0 / len(p_dupRatio_list)) if len(p_dupRatio_list) != 0 else 0.0))
    logger.info(
        '# absent ex num={}, ave a_DupRatio={:.5f}'.format(len(a_dupRatio_list), (sum(a_dupRatio_list) * 1.0 / len(a_dupRatio_list)) if len(a_dupRatio_list) != 0 else 0.0))

    logger.info('# Min Total preds num=%d' % min(total_preds_nums_list))
    logger.info('# Min Present preds num=%d' % min(present_preds_nums_list))
    logger.info('# Min Absent preds num=%d' % min(absent_preds_nums_list))
    logger.info('num_present_preds_statistics: ')
    logger.info(' '.join([str(idx) + ':' + str(num) for idx, num in enumerate(num_present_preds_statistics) if num != 0]))
    logger.info('num_absent_preds_statistics: ')
    logger.info(' '.join([str(idx) + ':' + str(num) for idx, num in enumerate(num_absent_preds_statistics) if num != 0]))

    map_score_fc(macro_metrics['total'], total_target_nums_list, keyphrase_type='total')
    map_score_fc(macro_metrics['present'], present_target_nums_list, keyphrase_type='present')
    map_score_fc(macro_metrics['absent'], absent_target_nums_list, keyphrase_type='absent')

    micro_ave_fc(macro_metrics['total'], keyphrase_type='total')
    micro_ave_fc(macro_metrics['present'], keyphrase_type='present')
    micro_ave_fc(macro_metrics['absent'], keyphrase_type='absent')


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r, target_num=None):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    if target_num:
        return np.sum(out)*1.0/target_num
    else:
        return np.mean(out)


def mean_average_precision(rs, target_nums_list):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    if target_nums_list:
        return np.mean([average_precision(r, target_num) for r, target_num in zip(rs, target_nums_list)])
    else:
        return np.mean([average_precision(r) for r in rs])


def map_score_fc(macro_metrics, target_lens_list=None, keyphrase_type='', at_keys=None):
    assert keyphrase_type != ''
    # configure the considered prediction numbers
    if at_keys is None:
        at_keys = [5, 10, 15, 50]
    else:
        at_keys = [int(at_key) for at_key in at_keys]
    # calculate the map scores
    map = {}
    logger.info('\n')
    logger.info('Begin' + '=' * 20 + keyphrase_type + '=' * 20 + 'Begin')
    output_str = ''
    for at_key in at_keys:
        assert type(at_key) is int
        correctly_matched_at_k = [macro_metric['correctly_matched'][:at_key] for macro_metric in macro_metrics]
        map[at_key] = mean_average_precision(correctly_matched_at_k, target_lens_list)
        output_str += '@%d=%f, ' % (at_key, map[at_key])

    output_str = 'MAP_%s:\t\t' % (keyphrase_type) + output_str
    logger.info(output_str)
    logger.info('End' + '=' * 20 + keyphrase_type + '=' * 20 + 'End')


def macro_metric_fc(tgt_set, correctly_matched, at_keys=None):
    metric_dict = {}
    metric_dict['target_number'] = len(tgt_set)
    metric_dict['prediction_number'] = len(correctly_matched)
    metric_dict['correctly_matched'] = correctly_matched

    # configure the considered prediction numbers
    if at_keys is None:
        at_keys = [5, 10, 'M']
    else:
        at_keys = [int(at_key) for at_key in at_keys if at_key != 'M']

    for topk in at_keys:
        if topk != 'M':
            valid_pred_num = topk
        else:
            valid_pred_num = len(correctly_matched)
        metric_dict['correct_number@%s' % topk] = sum(correctly_matched[:valid_pred_num])
        metric_dict['p@%s' % topk] = float(sum(correctly_matched[:valid_pred_num])) / float(valid_pred_num) if valid_pred_num != 0 else 0.0

        if len(tgt_set) != 0:
            metric_dict['r@%s' % topk] = float(sum(correctly_matched[:valid_pred_num])) / float(len(tgt_set)) if len(tgt_set) != 0 else 0.0
        else:
            metric_dict['r@%s' % topk] = 0

        if metric_dict['p@%s' % topk] + metric_dict['r@%s' % topk] != 0:
            metric_dict['f1@%s' % topk] = 2 * metric_dict['p@%s' % topk] * metric_dict['r@%s' % topk] / float(
                metric_dict['p@%s' % topk] + metric_dict['r@%s' % topk])
        else:
            metric_dict['f1@%s' % topk] = 0
    return metric_dict


def micro_ave_fc(macro_metrics, keyphrase_type='total', at_keys=None):
    # configure the considered prediction numbers
    if at_keys is None:
        at_keys = [5, 10, 'M']
    else:
        at_keys = [int(at_key) for at_key in at_keys if at_key != 'M']

    logger.info('\n')
    logger.info('Begin' + '='*20 + keyphrase_type + '='*20 + 'Begin')
    real_test_size = len(macro_metrics)
    logger.info("real_test_size: {}".format(real_test_size))
    overall_score = {}
    for topk in at_keys:
        correct_number = sum([m['correct_number@%s' % topk] for m in macro_metrics])
        overall_target_number = sum([m['target_number'] for m in macro_metrics])
        # sum([min(m['prediction_number'], k) for m in micro_metrics])
        if topk != 'M':
            overall_prediction_number = sum([topk for _ in macro_metrics])
        else:
            overall_prediction_number = sum([m['prediction_number'] for m in macro_metrics])

        # if real_test_size * k > overall_prediction_number:
        #     overall_prediction_number = real_test_size * k

        # Compute the Macro Measures, by averaging the macro-score of each prediction
        overall_score['p@%s' % topk] = float(sum([m['p@%s' % topk] for m in macro_metrics])) / float(real_test_size)
        overall_score['r@%s' % topk] = float(sum([m['r@%s' % topk] for m in macro_metrics])) / float(real_test_size)
        overall_score['f1@%s' % topk] = float(sum([m['f1@%s' % topk] for m in macro_metrics])) / float(real_test_size)

        # Print basic statistics
        output_str = 'Overall - valid testing data=%d, Number of Target=%d, Number of Prediction=%d, Number of Correct=%d' % (
            real_test_size,
            overall_target_number,
            overall_prediction_number, correct_number
        )
        logger.info(output_str)
        # Print macro-average performance
        overall_score['std_f1@%s' % topk] = 0.0
        if (overall_score["p@%s" % topk] + overall_score["r@%s" % topk]) != 0:
            overall_score['std_f1@%s' % topk] = 2 * overall_score["p@%s" % topk] * overall_score["r@%s" % topk] / (overall_score["p@%s" % topk] + overall_score["r@%s" % topk])
        output_str = 'Macro_%s_%s:\t\tP@%s=%f, R@%s=%f, F1@%s=%f, std_F1@%s=%f' % (
            keyphrase_type, topk,
            topk, overall_score['p@%s' % topk],
            topk, overall_score['r@%s' % topk],
            topk, overall_score['f1@%s' % topk],
            topk, overall_score['std_f1@%s' % topk]
        )
        logger.info(output_str)

        # Print micro-average performance
        overall_score['micro_p@%s' % topk] = correct_number / float(overall_prediction_number) if overall_prediction_number != 0 else 0
        overall_score['micro_r@%s' % topk] = correct_number / float(overall_target_number) if overall_target_number != 0 else 0
        if overall_score['micro_p@%s' % topk] + overall_score['micro_r@%s' % topk] > 0:
            overall_score['micro_f1@%s' % topk] = 2 * overall_score['micro_p@%s' % topk] * overall_score[
                'micro_r@%s' % topk] / float(overall_score['micro_p@%s' % topk] + overall_score['micro_r@%s' % topk])
        else:
            overall_score['micro_f1@%s' % topk] = 0

        output_str = 'Micro_%s_%s:\t\tP@%s=%f, R@%s=%f, F1@%s=%f' % (
            keyphrase_type, topk,
            topk, overall_score['micro_p@%s' % topk],
            topk, overall_score['micro_r@%s' % topk],
            topk, overall_score['micro_f1@%s' % topk]
        )
        logger.info(output_str)
        logger.info('End' + '=' * 20 + keyphrase_type + '=' * 20 + 'End')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluation')
    # Same with the arguments of the translating
    parser.add_argument('--src', '-src', type=str,
                        default='data\\full_data\\word_kp20k_testing_context.txt')
    parser.add_argument('--tgt', '-tgt', type=str,
                        default='data\\full_data\\kp20k_testing_keyword.txt')
    parser.add_argument('--output', '-output', type=str,
                        default='log\\translation_log\\merged_predictions.txt')
    # Specific arguments for evaluation
    parser.add_argument('--preds_cutoff_num', '-preds_cutoff_num', type=int, default=-1)
    parser.add_argument('--single_word_maxnum', '-single_word_maxnum', type=int, default=-1,
                        help=""""The maximum number of preserved single word predictions. 
                        The default is -1 which means preserve all the single word predictions.""")
    parser.add_argument('--filter_dot_comma_unk', '-filter_dot_comma_unk', type=bool, default=True,
                        help="""Whether to filter out the predictions with dot, comma and unk symbol. The default is true.""")
    parser.add_argument('--match_method', '-match_method', type=str, default='word_match',
                        choices=['word_match'])
    parser.add_argument('--ap_splitter', '-ap_splitter', type=str, default='<absent_end>')
    parser.add_argument('--log_file', '-log_file', type=str, default='logs\\evaluation.log')

    opts = parser.parse_args()
    init_logger(opts.log_file)
    evaluate_func(opts=opts)