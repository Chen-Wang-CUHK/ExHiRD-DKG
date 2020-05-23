import csv
import os
from nltk.stem.porter import *
import argparse
from tqdm import tqdm


STEMMER = PorterStemmer()

def collect_all_eval_results(log_dir_base, model_name, win_size=0, true_ave_type='Macro', seed_list=(343, 3435, 34350), log_prefix=""):
    # Note that the Micro and Macro are reversed, we choose to report the true Macro ave results.
    assert true_ave_type in ['Macro', 'Micro']
    ave_type = true_ave_type
    # csv_name = os.path.join(log_dir_base, '{}_translate_eval_bs{}_{}.csv'.format(model_name, bs, true_ave_type))
    csv_name = os.path.join(log_dir_base, '{}_history{}_translate_eval_{}.csv'.format(model_name, win_size, true_ave_type))
    with open(csv_name, 'w', encoding='utf-8', newline='') as csvfile:
        def csv_initial(seed, model_name):
            fieldnames = ['model_name']
            rslt_dict = {'model_name': "seed{}_{}".format(seed, model_name)}

            # for DupRatio
            for dataset in ['inspec', 'krapivin', 'semeval', 'kp20k']:
                # for DupRatio
                fn = '{}_ave_DupRatio'.format(dataset)
                fieldnames.append(fn)
                rslt_dict[fn] = -1
            fn = 'blank_ave_DupRatio'
            fieldnames.append(fn)
            rslt_dict[fn] = '--'

            for key_type in ['total', 'present', 'absent']:
                for dataset in ['inspec', 'krapivin', 'semeval', 'kp20k']:
                    # # for DupRatio
                    # if key_type == 'total':
                    #     fn = '{}_ave_DupRatio'.format(dataset)
                    #     fieldnames.append(fn)
                    #     rslt_dict[fn] = -1
                    # for F1 scores
                    metric_type = 'F1'

                    fn = '{}_{}_{}_M'.format(dataset, key_type, metric_type)
                    fieldnames.append(fn)
                    rslt_dict[fn] = 0.0

                    fn = '{}_{}_{}_5'.format(dataset, key_type, metric_type)
                    fieldnames.append(fn)
                    rslt_dict[fn] = 0.0

                fn = 'blank1_{}'.format(key_type)
                fieldnames.append(fn)
                rslt_dict[fn] = '--'

                for dataset in ['inspec', 'krapivin', 'semeval', 'kp20k']:
                    fn = '{}_ave_{}_preds_num'.format(dataset, key_type)
                    fieldnames.append(fn)
                    rslt_dict[fn] = 0.0

                fn = 'blank2_{}'.format(key_type)
                fieldnames.append(fn)
                rslt_dict[fn] = '--'
            return fieldnames, rslt_dict
        fieldnames, _ = csv_initial(seed_list[0], model_name)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for seed in seed_list:
            _, rslt_dict = csv_initial(seed, model_name)

            log_dir = os.path.join(log_dir_base, "seed{}".format(seed))
            for dataset in ['inspec', 'krapivin', 'semeval', 'kp20k']:
                # ExHiRD_s_seed34_inspec_history0.log
                eval_log = log_prefix + "{}_seed{}_{}_history{}.log".format(model_name, seed, dataset, win_size)
                eval_log = os.path.join(log_dir, eval_log)
                # # for catSeq
                # eval_log = "{}_{}_translate.log".format(dataset, model_name)
                # eval_log = os.path.join(log_dir_base, eval_log)
                if os.path.isfile(eval_log):
                    eval_file = open(eval_log, encoding='utf-8')
                    for line in eval_file.readlines():
                        # if "MAP" in line:
                        #     key_type, scores = line.strip().split(':')
                        #     key_type = key_type.strip().split('_')[-1]
                        #
                        #     scores = scores.strip().split(',')
                        #     map_5 = float(scores[0].strip().split('=')[-1].strip())
                        #     fn = '{}_{}_map_5'.format(dataset, key_type)
                        #     rslt_dict[fn] = map_5
                        #     map_10 = float(scores[1].strip().split('=')[-1].strip())
                        #     fn = '{}_{}_map_10'.format(dataset, key_type)
                        #     rslt_dict[fn] = map_10

                        # for DupRatio
                        if ' DupRatio' in line:
                            DupRatio = line.strip().split('DupRatio=')[-1]
                            DupRatio = float(DupRatio)
                            rslt_dict['{}_ave_DupRatio'.format(dataset)] = DupRatio

                        for key_type in ['total', "present", "absent"]:
                            if "ave {} preds num=".format(key_type) in line.lower():
                                ave_num = line.strip().split("num=")[-1]
                                ave_num = float(ave_num)
                                rslt_dict['{}_ave_{}_preds_num'.format(dataset, key_type)] = ave_num

                            if "{}_{}_5:".format(ave_type, key_type) in line:
                                f1_5 = line.strip().split('std_F1@5=')[-1]
                                f1_5 = float(f1_5.strip())
                                rslt_dict['{}_{}_F1_5'.format(dataset, key_type)] = f1_5

                            if "{}_{}_M:".format(ave_type, key_type) in line:
                                f1_M = line.strip().split('std_F1@M=')[-1]
                                f1_M = float(f1_M.strip())
                                rslt_dict['{}_{}_F1_M'.format(dataset, key_type)] = f1_M
            writer.writerow(rslt_dict)


def get_our_splitted_from_catseq(catseq_all_gt_file, catseq_all_pred_file, our_test_gt_file, saved_file, saved_gt_file):
    catseq_all_gt_lines = open(catseq_all_gt_file, encoding='utf-8').readlines()
    catseq_all_pred_lines = open(catseq_all_pred_file, encoding='utf-8').readlines()
    assert len(catseq_all_gt_lines) == len(catseq_all_pred_lines)
    our_test_gt_lines = open(our_test_gt_file, encoding='utf-8').readlines()
    match_catseq_pred_lines = []
    match_catseq_gt_lines = []
    for i in tqdm(len(our_test_gt_lines)):
        our_gt_set = our_test_gt_lines[i].strip().split(';')
        our_gt_set = [key.strip() for key in our_gt_set if len(key.strip()) != 0 and ',' not in key and ')' not in key and '(' not in key]
        our_gt_set = set(our_gt_set)
        matched = []
        matched_idx = []
        matched_gt = []
        ratio_list = []
        for j in range(len(catseq_all_gt_lines)):
            catseq_all_gt_line = catseq_all_gt_lines[j]
            catseq_all_gt_set = catseq_all_gt_line.strip().split('/')
            catseq_all_gt_set = [key.strip() for key in catseq_all_gt_set if len(key.strip()) != 0]
            catseq_all_gt_set = set(catseq_all_gt_set)
            if 'semeval' in catseq_all_pred_file.lower():
                stemmed_keys = [' '.join([STEMMER.stem(w) for w in key.split()]) for key in catseq_all_gt_set]
                catseq_all_gt_set = set(stemmed_keys)
            dup_ratio = len(our_gt_set & catseq_all_gt_set) * 1.0 / len(our_gt_set | catseq_all_gt_set)
            ratio_list.append(dup_ratio)
            # if dup_ratio >= 0.9:
            #     matched_idx.append(j)
            #     matched.append(catseq_all_pred_lines[j])
            #     matched_gt.append(catseq_all_gt_lines[j])

        max_ratio = max(ratio_list)
        for j in range(len(ratio_list)):
            if ratio_list[j] == max_ratio:
                matched_idx.append(j)
                matched.append(catseq_all_pred_lines[j].strip() + '\n')
                matched_gt.append(catseq_all_gt_lines[j].strip() + '\n')

        if len(matched) == 1:
            match_catseq_pred_lines = match_catseq_pred_lines + matched
            match_catseq_gt_lines = match_catseq_gt_lines + matched_gt
        else:
            print("Multiple matches!")
            raise NotImplementedError
    saved_file = open(saved_file, 'w', encoding='utf-8')
    saved_file.writelines(match_catseq_pred_lines)

    saved_gt_file = open(saved_gt_file, 'w', encoding='utf-8')
    saved_gt_file.writelines(match_catseq_gt_lines)


def get_our_splitted_from_catseq(catseq_all_gt_file, catseq_all_pred_file, our_test_gt_file, saved_file, saved_gt_file):
    catseq_all_gt_lines = open(catseq_all_gt_file, encoding='utf-8').readlines()
    catseq_all_pred_lines = open(catseq_all_pred_file, encoding='utf-8').readlines()
    assert len(catseq_all_gt_lines) == len(catseq_all_pred_lines)
    our_test_gt_lines = open(our_test_gt_file, encoding='utf-8').readlines()
    match_catseq_pred_lines = []
    match_catseq_gt_lines = []
    total_matched_idx = []
    for i in tqdm(range(len(our_test_gt_lines))):
        our_gt_set = our_test_gt_lines[i].strip().split(';')
        our_gt_set = [key.strip() for key in our_gt_set if len(key.strip()) != 0 and ',' not in key and ')' not in key and '(' not in key]
        our_gt_set = set(our_gt_set)
        matched = []
        matched_idx = []
        matched_gt = []
        ratio_list = []
        max_ratio_idx = []
        max_ratio = 0.0
        for j in range(len(catseq_all_gt_lines)):
            catseq_all_gt_line = catseq_all_gt_lines[j]
            if catseq_all_gt_line == '':
                continue
            catseq_all_gt_set = catseq_all_gt_line.strip().split('/')
            catseq_all_gt_set = [key.strip() for key in catseq_all_gt_set if len(key.strip()) != 0]
            catseq_all_gt_set = set(catseq_all_gt_set)
            if 'semeval' in catseq_all_pred_file.lower():
                stemmed_keys = [' '.join([STEMMER.stem(w) for w in key.split()]) for key in catseq_all_gt_set]
                catseq_all_gt_set = set(stemmed_keys)

            union_cnt = len(our_gt_set | catseq_all_gt_set)
            dup_ratio = len(our_gt_set & catseq_all_gt_set) * 1.0 / union_cnt if union_cnt != 0 else 0.0
            ratio_list.append(dup_ratio)

            if dup_ratio > max_ratio:
                max_ratio = dup_ratio
                max_ratio_idx = [j]
            elif dup_ratio == max_ratio:
                max_ratio_idx.append(j)

            if dup_ratio == 1.0:
                break
            # if dup_ratio >= 0.9:
            #     matched_idx.append(j)
            #     matched.append(catseq_all_pred_lines[j])
            #     matched_gt.append(catseq_all_gt_lines[j])

        # max_ratio = max(ratio_list)
        if max_ratio >= 0.2:
            # for j in max_ratio_idx:
            j = max_ratio_idx[0]
            matched_idx.append(j)
            matched.append(catseq_all_pred_lines[j].strip() + '\n')
            matched_gt.append(catseq_all_gt_lines[j].strip() + '\n')
        elif 'kp20k' in catseq_all_pred_file.lower():
            matched_idx.append(-1)
            matched.append('\n')
            matched_gt.append('\n')
        else:
            print("Multiple matches!")
            raise NotImplementedError

        if len(matched) == 1:
            if matched_idx[0] != -1:
                catseq_all_gt_lines[matched_idx[0]] = ''
                total_matched_idx.append(matched_idx[0])
            match_catseq_pred_lines.append(matched[0])
            match_catseq_gt_lines.append(matched_gt[0])
        else:
            print("Multiple matches!")
            raise NotImplementedError
    print("total_matched_idx num: {}".format(len(total_matched_idx)))
    saved_file = open(saved_file, 'w', encoding='utf-8')
    saved_file.writelines(match_catseq_pred_lines)

    saved_gt_file = open(saved_gt_file, 'w', encoding='utf-8')
    saved_gt_file.writelines(match_catseq_gt_lines)



# def get_catseq_test_context(our_test_context_file, our_test_gt_file, catseq_all_gt_file, saved_file, saved_gt_file):
#     assert 'kp20k' in catseq_all_gt_file.lower()
#
#     our_context_lines = open(our_test_context_file, encoding='utf-8').readlines()
#     our_test_gt_lines = open(our_test_gt_file, encoding='utf-8').readlines()
#     catseq_all_gt_lines = open(catseq_all_gt_file, encoding='utf-8').readlines()
#
#     assert len(our_context_lines) == len(our_test_gt_lines)
#     match_our_context_lines = []
#     match_our_gt_lines = []
#     for i in tqdm(range(len(catseq_all_gt_lines))):
#         catseq_gt_set = catseq_all_gt_lines[i].strip().split('/')
#         catseq_gt_set = [key.strip() for key in catseq_gt_set if len(key.strip()) != 0]
#         catseq_gt_set = set(catseq_gt_set)
#         matched_context = []
#         matched_idx = []
#         matched_gt = []
#         ratio_list = []
#         for j in range(len(catseq_all_gt_lines)):
#             our_gt_line = our_test_gt_lines[j]
#             our_gt_set = our_gt_line.strip().split(';')
#             our_gt_set = [key.strip() for key in our_gt_set if len(key.strip()) != 0 and ',' not in key and ')' not in key and '(' not in key]
#             our_gt_set = set(our_gt_set)
#             dup_ratio = len(our_gt_set & catseq_gt_set) * 1.0 / len(our_gt_set | catseq_gt_set)
#             ratio_list.append(dup_ratio)
#
#         max_ratio = max(ratio_list)
#         if max_ratio != 0:
#             for j in range(len(ratio_list)):
#                 if ratio_list[j] == max_ratio:
#                     matched_idx.append(j)
#                     matched_context.append(our_context_lines[j].strip() + '\n')
#                     matched_gt.append(our_test_gt_lines[j].strip() + '\n')
#         else:
#             print("Multiple matches!")
#             raise NotImplementedError
#
#         if len(matched_context) == 1 or max_ratio == 1.0:
#             match_our_context_lines.append(matched_context[0])
#             match_our_gt_lines.append(matched_gt[0])
#         else:
#             print("Multiple matches!")
#             raise NotImplementedError
#     saved_file = open(saved_file, 'w', encoding='utf-8')
#     saved_file.writelines(match_our_context_lines)
#
#     saved_gt_file = open(saved_gt_file, 'w', encoding='utf-8')
#     saved_gt_file.writelines(match_our_gt_lines)


def get_present_context(tgt_keys_file, tgt_pid_file, pred_keys_file, pred_pid_file, saved_file):
    tgt_keys_lines = open(tgt_keys_file, encoding='utf-8').readlines()
    tgt_pid_lines = open(tgt_pid_file, encoding='utf-8').readlines()

    pred_keys_lines = open(pred_keys_file, encoding='utf-8').readlines()
    pred_pid_lines = open(pred_pid_file, encoding='utf-8').readlines()

    assert len(tgt_keys_lines) == len(tgt_pid_lines)
    assert len(pred_keys_lines) == len(pred_pid_lines)
    assert len(pred_keys_lines) == len(tgt_keys_lines)

    saved_file = open(saved_file, 'w', encoding='utf-8')
    for i in tqdm(range(len(tgt_keys_lines))):
        # print(i)
        tgt_present_keys = get_present_keys(tgt_keys_lines[i], tgt_pid_lines[i])
        pred_present_keys = get_present_keys(pred_keys_lines[i], pred_pid_lines[i])

        total_present_keys = tgt_present_keys + pred_present_keys
        saved_file.write(' ; '.join(total_present_keys) + '\n')


def get_present_keys(keys_line, pids_line):
    keys = keys_line.strip().split('/')
    keys = [key.strip() for key in keys]

    pids = pids_line.strip().split('/')
    pids = [ind.strip() for ind in pids]

    # if len(keys) != len(pids):
    #     print(' ')
    assert len(keys) == len(pids)
    # if len(keys) < len(pids):
    #     # assert keys_line.strip()[-1] == '/'
    #     if keys_line.strip()[-1] != '/':
    #         print(' ')
    #     pids = pids[:-1]

    present_keys = []
    for j in range(len(keys)):
        if pids[j] == 'True':
            if keys[j] != '':
                present_keys.append(keys[j])
            else:
                present_keys.append("empty present kp placeholder")

    return present_keys


if __name__ == '__main__':
    # for ExHiRD-s
    model_name = "ExHiRD_s"
    for win_size in [0]:
        log_dir_base = "logs\\reruned_evaluation_log\\{}\\history{}".format(model_name, win_size)
        collect_all_eval_results(log_dir_base=log_dir_base, model_name=model_name, win_size=win_size, seed_list=[34, 343, 3435])

    # for ExHiRD-s
    model_name = "ExHiRD_h"
    for win_size in [1, 4]:
        log_dir_base = "logs\\reruned_evaluation_log\\{}\\history{}".format(model_name, win_size)
        collect_all_eval_results(log_dir_base=log_dir_base, model_name=model_name, win_size=win_size,
                                 seed_list=[34, 343, 3435])