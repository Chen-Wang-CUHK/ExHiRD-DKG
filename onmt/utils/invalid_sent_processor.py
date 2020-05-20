"""
Author: wchen@cse.cuhk.edu.hk
This includes
"""
import torch

from onmt.utils.misc import aeq


def valid_src_compress(src, sent_nums, sent_lens):
    """
    Select all the valid sentences and remove the invalid ones.
    :param sent_batch: [batch, s_num, s_len, *]
    :param sent_nums: [batch]
    :param sent_lens: [batch, s_num]
    :return valid_src: [s_total, * ...], s_total = s_nums.sum().item()
            valid_sent_lens: [s_total]
    """
    # remove the padded empty sentence
    batch = src.size(0)
    s_num = src.size(1)
    s_len = src.size(2)

    batch_1 = sent_nums.size(0)

    batch_2 = sent_lens.size(0)
    s_num_1 = sent_lens.size(1)
    s_len_1 = sent_lens.max().item()

    # check the arguments
    aeq(batch, batch_1, batch_2)
    aeq(s_num, s_num_1)
    aeq(s_len, s_len_1)

    # valid_total_sent_num
    s_total = sent_nums.sum().item()
    valid_src = []
    valid_sent_lens = []
    for i in range(batch):
        valid_src.append(src[i, :sent_nums[i]])
        valid_sent_lens.append(sent_lens[i, :sent_nums[i]])
    # [s_total, s_len, *]
    valid_src = torch.cat(valid_src, dim=0)
    # [s_total]
    valid_sent_lens = torch.cat(valid_sent_lens, dim=0)

    return valid_src, valid_sent_lens


def recover_src(valid_src, sent_nums):
    """
    Recover the compressed valid_src
    :param valid_src: [s_total, * ...], s_total == sent_nums.sum().item()
    :param sent_nums: [batch]
    :return  recovered_src: [batch, s_num, * ...]
    """

    batch = sent_nums.size(0)
    s_num = sent_nums.max().item()
    s_total = sent_nums.sum().item()

    s_total_ = valid_src.size(0)

    # check the args
    aeq(s_total, s_total_)

    # recover the word memory bank and the word final state
    recovered_src = []
    start = 0
    for i in range(batch):
        s_num_i = sent_nums[i]

        # for word memory bank
        # [s_num_i, s_len, 2 * hidden_size]
        recovered = valid_src[start: start + s_num_i]
        if s_num_i < s_num:
            padded_dims = [s_num - s_num_i] + list(valid_src.size())[1:]
            padded = torch.zeros(padded_dims).cuda()
            recovered = torch.cat([recovered, padded], dim=0)
        # [s_num, * ...]
        recovered_src.append(recovered)

        start = start + s_num_i
    # [batch, s_num, * ...]
    recovered_src = torch.stack(recovered_src, dim=0)
    return recovered_src

