import torch
import torch.nn as nn

import onmt
import onmt.inputters as inputters
from onmt.utils.misc import aeq, sequence_mask
from onmt.utils.loss import LossComputeBase

# add by wchen
from data_utils import P_START, A_START, KEY_SEPERATOR
EPS = 1e-8

class CopyGenerator(nn.Module):
    """An implementation of pointer-generator networks (See et al., 2017)
    (https://arxiv.org/abs/1704.04368), which consider copying words
    directly from the source sequence.

    The copy generator is an extended version of the standard
    generator that computes three values.

    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.

    The model returns a distribution over the extend dictionary,
    computed as

    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`


    .. mermaid::

       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O


    Args:
       input_size (int): size of input representation
       output_size (int): size of output vocabulary
       pad_idx (int)
    """

    def __init__(self, input_size, output_size, pad_idx):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.linear_copy = nn.Linear(input_size, 1)
        self.pad_idx = pad_idx

    def forward(self, hidden, attn, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by compying
        source words.

        Args:
           hidden (`FloatTensor`): hidden outputs `[batch*tlen, input_size]`
           attn (`FloatTensor`): attn for each `[batch*tlen, input_size]`
           src_map (`FloatTensor`):
             A sparse indicator matrix mapping each source word to
             its index in the "extended" vocab containing.
             `[src_len, batch, extra_words]`
        """
        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.pad_idx] = -float('inf')
        prob = torch.softmax(logits, 1)

        # Probability of copying p(z=1) batch.
        p_copy = torch.sigmoid(self.linear_copy(hidden))
        # Probability of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy)
        mul_attn = torch.mul(attn, p_copy)
        copy_prob = torch.bmm(
            mul_attn.view(-1, batch, slen).transpose(0, 1),
            src_map.transpose(0, 1)
        ).transpose(0, 1)
        copy_prob = copy_prob.contiguous().view(-1, cvocab)
        return torch.cat([out_prob, copy_prob], 1)


class CopyGeneratorLoss(nn.Module):
    """ Copy generator criterion """

    def __init__(self, vocab_size, force_copy, unk_index=0,
                 ignore_index=-100, eps=1e-20):
        super(CopyGeneratorLoss, self).__init__()
        self.force_copy = force_copy
        self.eps = eps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.unk_index = unk_index

    def forward(self, scores, align, target):
        """
        scores (FloatTensor): (batch_size*tgt_len) x dynamic vocab size
        align (LongTensor): (batch_size*tgt_len)
        target (LongTensor): (batch_size*tgt_len)
        """
        # probabilities assigned by the model to the gold targets
        vocab_probs = scores.gather(1, target.unsqueeze(1)).squeeze(1)

        # probability of tokens copied from source
        copy_ix = align.unsqueeze(1) + self.vocab_size
        copy_tok_probs = scores.gather(1, copy_ix).squeeze(1)
        # Set scores for unk to 0 and add eps
        copy_tok_probs[align == self.unk_index] = 0
        copy_tok_probs += self.eps  # to avoid -inf logs

        # find the indices in which you do not use the copy mechanism
        non_copy = align == self.unk_index
        if not self.force_copy:
            non_copy = non_copy | (target != self.unk_index)

        probs = torch.where(
            non_copy, copy_tok_probs + vocab_probs, copy_tok_probs
        )

        loss = -probs.log()  # just NLLLoss; can the module be incorporated?
        # Drop padding.
        loss[target == self.ignore_index] = 0
        return loss


class CopyGeneratorLossCompute(LossComputeBase):
    """
    Copy Generator Loss Computation.
    """

    def __init__(self, criterion, generator, tgt_vocab, normalize_by_length):
        super(CopyGeneratorLossCompute, self).__init__(criterion, generator)
        self.tgt_vocab = tgt_vocab
        self.normalize_by_length = normalize_by_length

    def _make_shard_state(self, batch, output, range_, attns):
        """ See base class for args description. """
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")

        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
            "copy_attn": attns.get("copy"),
            "align": batch.alignment[range_[0] + 1: range_[1]]
        }

    def _compute_loss(self, batch, output, target, copy_attn, align):
        """
        Compute the loss. The args must match self._make_shard_state().
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """
        target = target.view(-1)
        align = align.view(-1)
        scores = self.generator(
            self._bottle(output), self._bottle(copy_attn), batch.src_map
        )
        loss = self.criterion(scores, align, target)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        scores_data = inputters.TextDataset.collapse_copy_scores(
            self._unbottle(scores.clone(), batch.batch_size),
            batch, self.tgt_vocab, batch.dataset.src_vocabs)
        scores_data = self._bottle(scores_data)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = target.clone()
        unk = self.criterion.unk_index
        correct_mask = (target_data == unk) & (align != unk)
        offset_align = align[correct_mask] + len(self.tgt_vocab)
        target_data[correct_mask] += offset_align

        # Compute sum of perplexities for stats
        stats = self._stats(loss.sum().clone().item(), scores_data, target_data)

        # this part looks like it belongs in CopyGeneratorLoss
        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            tgt_lens = batch.tgt.ne(self.padding_idx).sum(0).float()
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

        return loss, stats


class HREDCopyGeneratorLossCompute(LossComputeBase):
    """
    Copy Generator Loss Computation.
    """

    def __init__(self, criterion, generator, tgt_vocab, normalize_by_length,
                 add_orthg=False, lambda_orthogonal=0.0, add_cover=False, lambda_cover=0.0,
                 add_exclusive_loss=False, lambda_ex=0.0, ex_loss_win_size=1,
                 lambda_first_word_nll=1.0, lambda_valid_words_nll=1.0, add_te_loss=False, lambda_te=0.0):
        super(HREDCopyGeneratorLossCompute, self).__init__(criterion, generator)
        self.tgt_vocab = tgt_vocab
        self.normalize_by_length = normalize_by_length
        self.add_orthg = add_orthg
        self.lambda_orthogonal = lambda_orthogonal
        self.add_cover = add_cover
        self.lambda_cover = lambda_cover
        self.add_exclusive_loss = add_exclusive_loss
        self.lambda_ex = lambda_ex
        self.ex_loss_win_size = ex_loss_win_size
        self.lambda_first_word_nll = lambda_first_word_nll
        self.lambda_valid_words_nll = lambda_valid_words_nll
        self.add_te_loss = add_te_loss
        self.lambda_te = lambda_te

    @property
    def unk_idx(self):
        return self.criterion.unk_index

    @property
    def eps(self):
        return self.criterion.eps

    @property
    def force_copy(self):
        return self.criterion.force_copy

    @property
    def vocab_size(self):
        return self.criterion.vocab_size

    def _make_shard_state(self, batch, output, range_, attns=None):
        """ See base class for args description. """
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")
        # "range_" is ignored
        b_size, tgt_s_num, tgt_s_len = batch.tgt.size()
        output_target = batch.tgt[:, :, 1:]
        # [tgt_s_num, tgt_s_len - 1, b_size]
        output_target = output_target.transpose(0, 1).transpose(1, 2)
        # [tgt_s_num * (tgt_s_len - 1), b_size]
        output_target = output_target.contiguous().view(-1, b_size)

        # [tgt_s_num, tgt_s_len - 1, b_size]
        output_align = batch.alignment[:, 1:]
        # [tgt_s_num * (tgt_s_len - 1), b_size]
        output_align = output_align.contiguous().view(-1, b_size)

        # for sent coverage
        # [b_size, tgt_s_num, sent_memory_bank_len] -> [tgt_s_num, b_size, sent_memory_bank_len]
        sent_attns = attns.get("sent_std_attn")
        sent_attns = sent_attns.transpose(0, 1) if sent_attns is not None else None
        sent_coverage = attns.get("sent_coverage")
        sent_coverage = sent_coverage.transpose(0, 1) if sent_coverage is not None else None

        return {
            "output": output,
            "target": output_target,
            "copy_attn": attns.get("copy"),
            "align": output_align,
            "orthog_states": attns.get("orthog_states"),
            "sent_attns": sent_attns,
            "sent_coverage": sent_coverage,
            "target_attns": attns.get("target_attns")
        }

    def _compute_orthogonal_loss(self, batch, orthog_states):
        """
        The orthogonal loss computation function
        :param batch: the current batch
        :param orthog_states: the orthog_states from the sent level decoder
        :return: a scalar, the orthogonal loss
        """
        # [b_size, s_num, tgt_s_len-1]
        valid_tgt = batch.tgt[:, :, 1:]
        b_size, s_num, _ = valid_tgt.size()
        b_size1, s_num1, _ = orthog_states.size()
        aeq(b_size, b_size1)
        aeq(s_num, s_num1)

        # obtain the mask
        # [b_size, s_num]
        mask = valid_tgt.ne(self.padding_idx).sum(dim=-1).ne(0)
        mask = mask.float()
        # [b_size, 1, s_num]
        mask = mask.unsqueeze(1)
        # [b_size, s_num, s_num]
        mask_2d = torch.bmm(mask.transpose(1, 2), mask)

        # compute the loss
        # [b_size, s_num, s_num]
        identity = torch.eye(s_num).unsqueeze(0).repeat(b_size, 1, 1).to(orthog_states.device)
        # [b_size, s_num, s_num]
        orthogonal_loss_ = torch.bmm(orthog_states, orthog_states.transpose(1, 2)) - identity
        orthogonal_loss_ = orthogonal_loss_ * mask_2d
        # [b_size]
        orthogonal_loss = torch.norm(orthogonal_loss_.view(b_size, -1), p=2, dim=1)
        return orthogonal_loss

    def _compute_coverage_loss(self, batch, sent_attns, sent_coverage):
        """
        :param batch: the current batch
        :param sent_attns: the sent attentions of each step, [s_num, b_size, sent_memory_bank_len]
        :param sent_coverage: the sent coverages, [s_num, b_size, sent_memory_bank_len]
        :return: coverage loss, [1]
        """
        assert sent_attns is not None
        assert sent_coverage is not None

        # [b_size, s_num]
        valid_tgt = batch.tgt[:, :, 1]
        b_size, s_num = valid_tgt.size()
        # obtain the mask
        # [b_size, s_num]
        mask = valid_tgt.ne(self.padding_idx).float()
        # [s_num, b_size]
        mask = mask.transpose(0, 1)
        mask = mask.unsqueeze(-1)

        s_num1, b_size1, mem_len = sent_attns.size()
        s_num2, b_size2, mem_len_ = sent_attns.size()
        aeq(s_num, s_num1, s_num2)
        aeq(b_size, b_size1, b_size2)
        aeq(mem_len, mem_len_)
        # apply the mask
        sent_attns = sent_attns * mask
        sent_coverage = sent_coverage * mask
        # compute coverage loss
        # p_t
        sent_attns = sent_attns[1:]
        # c_t = p_0 + p_1 + p_2 + ... + p_(t-1)
        sent_coverage = sent_coverage[:-1]
        cover_loss = torch.where(sent_attns < sent_coverage, sent_attns, sent_coverage)
        return cover_loss.sum()

    def _compute_exclusive_loss(self, first_word_scores, first_word_targets, first_aligns, win_size):
        """
        compute the exclusive loss
        :param first_word_scores: the predicted scores for the first valid word, [b_size, tgt_s_num, dy_vocab_size]
        :param first_word_targets: the target words for the first valid word, [b_size, tgt_s_num]
        :param first_aligns: the alignments for the first valid word, [b_size, tgt_s_num]
        :param win_size: the window size for exclusive loss
        :return:
        """
        b_size, tgt_s_num, dy_vocab_size = first_word_scores.size()
        b_size_, tgt_s_num_ = first_word_targets.size()
        aeq(b_size, b_size_)
        aeq(tgt_s_num, tgt_s_num_)

        if win_size == -1 or win_size > (tgt_s_num - 1):
            win_size = tgt_s_num - 1

        exclusive_tokens = []
        exclusive_aligns = []
        for i in range(1, tgt_s_num):
            if i < win_size:
                tokens_tmp = first_word_targets[:, :i]
                filled_tmp = torch.full([b_size, win_size - i], self.padding_idx,
                                        dtype=torch.long, device=tokens_tmp.device)
                tokens_tmp = torch.cat([tokens_tmp, filled_tmp], dim=1)

                aligns_tmp = first_aligns[:, :i]
                filled_tmp = torch.full([b_size, win_size - i], self.unk_idx,
                                        dtype=torch.long, device=aligns_tmp.device)
                aligns_tmp = torch.cat([aligns_tmp, filled_tmp], dim=1)
            else:
                tokens_tmp = first_word_targets[:, i - win_size:i]
                aligns_tmp = first_aligns[:, i - win_size:i]

            exclusive_tokens.append(tokens_tmp)
            exclusive_aligns.append(aligns_tmp)

        # [b_size, tgt_s_num-1, win_size]
        exclusive_tokens = torch.stack(exclusive_tokens, dim=1)
        # [b_size * tgt_s_num-1 * win_size]
        exclusive_tokens = exclusive_tokens.view(-1)
        # [b_size, tgt_s_num-1, win_size]
        exclusive_aligns = torch.stack(exclusive_aligns, dim=1)
        # [b_size * tgt_s_num-1 * win_size]
        exclusive_aligns = exclusive_aligns.view(-1)

        # [b_size, tgt_s_num, dy_vocab_size] -> [b_size, tgt_s_num - 1, dy_vocab_size]
        first_word_scores = first_word_scores[:, 1:, :]
        # [b_size, tgt_s_num - 1, win_size, dy_vocab_size]
        first_word_scores = first_word_scores.unsqueeze(2).expand(-1, -1, win_size, -1).contiguous()
        # [b_size * tgt_s_num - 1 * win_size, dy_vocab_size]
        first_word_scores = first_word_scores.view(-1, dy_vocab_size)

        """
        scores (FloatTensor): (batch_size*tgt_len) x dynamic vocab size
        align (LongTensor): (batch_size*tgt_len)
        target (LongTensor): (batch_size*tgt_len)
        """
        # probabilities assigned by the model to the gold targets
        vocab_probs = first_word_scores.gather(1, exclusive_tokens.unsqueeze(1)).squeeze(1)

        # probability of tokens copied from source
        copy_ix = exclusive_aligns.unsqueeze(1) + self.vocab_size
        copy_tok_probs = first_word_scores.gather(1, copy_ix).squeeze(1)
        # Set scores for unk to 0 and add eps
        copy_tok_probs[exclusive_aligns == self.unk_idx] = 0

        # find the indices in which you do not use the copy mechanism
        non_copy = exclusive_aligns == self.unk_idx
        if not self.force_copy:
            non_copy = non_copy | (exclusive_tokens != self.unk_idx)

        probs = torch.where(
            non_copy, copy_tok_probs + vocab_probs, copy_tok_probs
        )

        loss = -(1 - probs + self.eps).log()
        # Drop padding in exclusive tokens.
        loss[exclusive_tokens == self.padding_idx] = 0
        # Drop padding in original tokens.
        loss = loss.view(b_size, tgt_s_num-1, win_size)
        gt_first_word_targets = first_word_targets[:, 1:].unsqueeze(2).expand_as(loss)
        loss[gt_first_word_targets == self.padding_idx] = 0
        # Drop the same exclusive tokens with original tokens
        exclusive_tokens = exclusive_tokens.view(b_size, tgt_s_num-1, win_size)
        loss[gt_first_word_targets == exclusive_tokens] = 0
        return loss.sum()

    def _compute_te_loss(self, target_attns):
        """
        :param target_attns: a tuple (stacked_target_attns, target_attns_lens, src_states_target_list)
        :return: target encoding loss
        """
        # stacked_target_attns: [b_size, max_sep_num, sample_size+1]
        # target_attns_lens: [b_size]
        # src_states_target_list: [b_size]
        stacked_target_attns, target_attns_lens, src_states_target_list = target_attns
        b_size, max_sep_num, cls_num = stacked_target_attns.size()
        device = stacked_target_attns.device

        gt_tensor = torch.Tensor(src_states_target_list).view(b_size, 1).repeat(1, max_sep_num).to(device)

        # class_dist_flat: [b_size * max_sep_num, sample_size+1]
        class_dist_flat = stacked_target_attns.view(-1, cls_num)
        log_dist_flat = torch.log(class_dist_flat + EPS)
        target_flat = gt_tensor.view(-1, 1)
        # [b_size * max_sep_num, 1]
        losses_flat = -torch.gather(log_dist_flat, dim=1, index=target_flat.long())
        losses = losses_flat.view(b_size, max_sep_num)

        mask = sequence_mask(torch.Tensor(target_attns_lens)).to(device)
        losses = losses * mask.float()
        losses = losses.sum(dim=1)
        return losses

    def _compute_loss(self, batch, output, target, copy_attn, align,
                      orthog_states=None, sent_attns=None, sent_coverage=None, target_attns=None):
        """
        Compute the loss. The args must match self._make_shard_state().
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
            orthog_states: the word level decoding initialization state, [batch_size, s_num, h_size]
            sent_attns: the sent-level attention scores, [s_num, batch_size, sent_memory_bank_len]
            sent_coverage: the sent-level coverage scores, [s_num, batch_size, sent_memory_bank_len]
            target_attns: a tuple (stacked_target_attns, target_attns_lens, src_states_target_list)
        """
        # =================== 1. Compute the generation loss =========================
        b_size, tgt_s_num, tgt_s_len = batch.tgt.size()
        valid_tgt_s_len = tgt_s_len - 1
        # [tgt_s_num * valid_tgt_s_len, b_size] -> [tgt_s_num * valid_tgt_s_len * b_size]
        target = target.view(-1)
        # [tgt_s_num * valid_tgt_s_len, b_size] -> [tgt_s_num * valid_tgt_s_len * b_size]
        align = align.view(-1)
        # [tgt_s_num * valid_tgt_s_len * b_size, dy_vocab_size]
        scores = self.generator(
            self._bottle(output), self._bottle(copy_attn), batch.src_map
        )
        # [tgt_s_num * valid_tgt_s_len * b_size]
        gen_loss = self.criterion(scores, align, target)
        # give a different weight to the nll loss of the first valid word
        # [tgt_s_num, valid_tgt_s_len, b_size]
        gen_loss = gen_loss.view(tgt_s_num, valid_tgt_s_len, b_size)
        target = target.view(gen_loss.size())

        if self.lambda_valid_words_nll != 1:
            mask = torch.ones(gen_loss.size()).to(gen_loss.device) * self.lambda_valid_words_nll
            eos_idx = batch.dataset.fields['src'].vocab.stoi['</s>']
            mask[target == eos_idx] = 1.0

            p_start_idx = batch.dataset.fields['src'].vocab.stoi[P_START]
            mask[target == p_start_idx] = 1.0

            a_start_idx = batch.dataset.fields['src'].vocab.stoi[A_START]
            mask[target == a_start_idx] = 1.0

            sep_idx = batch.dataset.fields['src'].vocab.stoi[KEY_SEPERATOR]
            mask[target == sep_idx] = 1.0

            mask[target == self.padding_idx] = 0.0

            gen_loss = gen_loss * mask

        elif self.lambda_first_word_nll != 1:
            mask = torch.ones([tgt_s_num, valid_tgt_s_len, b_size]).to(gen_loss.device)
            mask[:, 1, :] = self.lambda_first_word_nll
            mask[target == self.padding_idx] = 0.0
            gen_loss = gen_loss * mask
        target = target.view(-1)
        # =================== 1. Compute the generation loss =========================

        # =================== 2. Compute the exclusive loss ==========================
        _, dy_vocab_size = scores.size()
        first_word_scores = scores.view(tgt_s_num, valid_tgt_s_len, b_size, dy_vocab_size)
        # [tgt_s_num, b_size, dy_vocab_size]
        first_word_scores = first_word_scores[:, 1, :, :]
        # [b_size, tgt_s_num, dy_vocab_size]
        first_word_scores = first_word_scores.transpose(0, 1)

        # [tgt_s_num, b_size]
        first_word_targets = target.view(tgt_s_num, valid_tgt_s_len, b_size)[:, 1, :]
        # [b_size, tgt_s_num]
        first_word_targets = first_word_targets.transpose(0, 1)

        # [tgt_s_num, b_size]
        first_aligns = align.view(tgt_s_num, valid_tgt_s_len, b_size)[:, 1, :]
        # [b_size, tgt_s_num]
        first_aligns = first_aligns.transpose(0, 1)

        exclusive_loss = torch.zeros(1).to(gen_loss.device)
        if self.add_exclusive_loss:
            exclusive_loss = self._compute_exclusive_loss(first_word_scores,
                                                          first_word_targets,
                                                          first_aligns,
                                                          self.ex_loss_win_size)
        # =================== 2. Compute the exclusive loss ==========================

        # =================== 3. Compute the orthogonal loss =========================
        orthogonal_loss = torch.zeros(1).to(gen_loss.device)
        if self.add_orthg:
            orthogonal_loss = self._compute_orthogonal_loss(batch, orthog_states)

        # =================== 4. Compute the sent-level coverage loss ================
        cover_loss = torch.zeros(1).to(gen_loss.device)
        if self.add_cover:
            cover_loss = self._compute_coverage_loss(batch, sent_attns, sent_coverage)

        # =================== 5. Compute the target encoding like loss ================
        target_enc_loss = torch.zeros(1).to(gen_loss.device)
        if self.add_te_loss and target_attns[0] is not None:
            target_enc_loss = self._compute_te_loss(target_attns)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        scores_data = inputters.TextDataset.collapse_copy_scores(
            self._unbottle(scores.clone(), batch.batch_size),
            batch, self.tgt_vocab, batch.dataset.src_vocabs)
        scores_data = self._bottle(scores_data)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = target.clone()
        unk = self.criterion.unk_index
        correct_mask = (target_data == unk) & (align != unk)
        offset_align = align[correct_mask] + len(self.tgt_vocab)
        target_data[correct_mask] += offset_align

        # this part looks like it belongs in CopyGeneratorLoss
        if self.normalize_by_length:
            raise NotImplementedError
            # # Compute Loss as NLL divided by seq length
            # tgt_lens = batch.tgt.ne(self.padding_idx).sum(0).float()
            # # Compute Total Loss per sequence in batch
            # loss = loss.view(-1, batch.batch_size).sum(0)
            # # Divide by length of each sequence and sum
            # loss = torch.div(loss, tgt_lens).sum()
        else:
            gen_loss = gen_loss.sum()
            orthogonal_loss = orthogonal_loss.sum()
            cover_loss = cover_loss.sum()
            exclusive_loss = exclusive_loss.sum()
            target_enc_loss = target_enc_loss.sum()

        # Compute the total loss
        total_loss = gen_loss + self.lambda_orthogonal * orthogonal_loss + \
                     self.lambda_cover * cover_loss + \
                     self.lambda_ex * exclusive_loss + \
                     self.lambda_te * target_enc_loss

        # Compute sum of perplexities for stats
        stats = self._stats(gen_loss.clone().item(),
                            scores_data, target_data,
                            orthogonal_loss.clone().item(),
                            cover_loss.clone().item(),
                            exclusive_loss.clone().item(),
                            target_enc_loss.clone().item(),
                            total_loss.clone().item(),
                            batch.batch_size)

        return total_loss, stats

    # def _stats(self, loss, scores, target, orthogonal_loss=None, cover_loss=None, te_loss=None, total_loss=None, b_size=None):
    #     """
    #     Args:
    #         loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
    #         scores (:obj:`FloatTensor`): a score for each possible output
    #         target (:obj:`FloatTensor`): true targets
    #         orthogonal_loss (:obj:`FloatTensor'): the computed orthogonal loss
    #         cover_loss (:obj:`FloatTensor'): the computed coverage loss
    #         cover_loss (:obj:`Float'): the computed target encoding loss
    #         total_loss (:obj:`FloatTensor'): the computed total loss: loss + lambda_orthogonal * orthogonal_loss
    #         b_size: the batch size
    #     Returns:
    #         :obj:`onmt.utils.Statistics` : statistics for this batch.
    #     """
    #     pred = scores.max(1)[1]
    #     non_padding = target.ne(self.padding_idx)
    #     num_correct = pred.eq(target).masked_select(non_padding).sum().item()
    #     num_non_padding = non_padding.sum().item()
    #     return onmt.utils.Statistics(loss.item(), num_non_padding, num_correct, orthogonal_loss.item(), cover_loss.item(), te_loss.item(), total_loss.item(), b_size)


class CatSeqDCopyGeneratorLossCompute(LossComputeBase):
    """
    Copy Generator Loss Computation.
    """
    def __init__(self, criterion, generator, tgt_vocab, normalize_by_length,
                 add_orthg=True, lambda_orthogonal=0.03, add_te_loss=True, lambda_te=0.03):
        super(CatSeqDCopyGeneratorLossCompute, self).__init__(criterion, generator)
        self.tgt_vocab = tgt_vocab
        self.normalize_by_length = normalize_by_length
        self.add_orthog = add_orthg
        self.lambda_orthogonal = lambda_orthogonal
        self.add_te_loss = add_te_loss
        self.lambda_te = lambda_te

    def _make_shard_state(self, batch, output, range_, attns):
        """ See base class for args description. """
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")

        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1]],
            "copy_attn": attns.get("copy"),
            "align": batch.alignment[range_[0] + 1: range_[1]],
            "sep_states": attns.get("sep_states"),
            "target_attns": attns.get("target_attns")
        }

    def _compute_orthogonal_loss(self, sep_states):
        """
        The orthogonal loss computation function
        sep_states: a tuple (stacked_sep_states, sep_states_lens)
        :return: a scalar, the orthogonal loss
        """
        # stacked_sep_states: [b_size, max_sep_num, src_h_size]
        stacked_sep_states, sep_states_lens = sep_states
        b_size, max_sep_num, src_h_size = stacked_sep_states.size()
        b_size_ = len(sep_states_lens)
        aeq(b_size, b_size_)

        device = stacked_sep_states.device

        # obtain the mask
        # [b_size, max_sep_num]
        mask = sequence_mask(torch.Tensor(sep_states_lens)).to(device)
        mask = mask.float()
        # [b_size, 1, max_sep_num]
        mask = mask.unsqueeze(1)
        # [b_size, max_sep_num, max_sep_num]
        mask_2d = torch.bmm(mask.transpose(1, 2), mask)

        # compute the loss
        # [b_size, max_sep_num, max_sep_num]
        identity = torch.eye(max_sep_num).unsqueeze(0).repeat(b_size, 1, 1).to(device)
        # [b_size, max_sep_num, max_sep_num]
        orthogonal_loss_ = torch.bmm(stacked_sep_states, stacked_sep_states.transpose(1, 2)) - identity
        orthogonal_loss_ = orthogonal_loss_ * mask_2d
        # [b_size]
        orthogonal_loss = torch.norm(orthogonal_loss_.view(b_size, -1), p=2, dim=1)
        return orthogonal_loss

    def _compute_te_loss(self, target_attns):
        """
        :param target_attns: a tuple (stacked_target_attns, target_attns_lens, src_states_target_list)
        :return: target encoding loss
        """
        # stacked_target_attns: [b_size, max_sep_num, sample_size+1]
        # target_attns_lens: [b_size]
        # src_states_target_list: [b_size]
        stacked_target_attns, target_attns_lens, src_states_target_list = target_attns
        b_size, max_sep_num, cls_num = stacked_target_attns.size()
        device = stacked_target_attns.device

        gt_tensor = torch.Tensor(src_states_target_list).view(b_size, 1).repeat(1, max_sep_num).to(device)

        # class_dist_flat: [b_size * max_sep_num, sample_size+1]
        class_dist_flat = stacked_target_attns.view(-1, cls_num)
        log_dist_flat = torch.log(class_dist_flat + EPS)
        target_flat = gt_tensor.view(-1, 1)
        # [b_size * max_sep_num, 1]
        losses_flat = -torch.gather(log_dist_flat, dim=1, index=target_flat.long())
        losses = losses_flat.view(b_size, max_sep_num)

        mask = sequence_mask(torch.Tensor(target_attns_lens)).to(device)
        losses = losses * mask.float()
        losses = losses.sum(dim=1)
        return losses


    def _compute_loss(self, batch, output, target, copy_attn, align, sep_states, target_attns):
        """
        Compute the loss. The args must match self._make_shard_state().
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
            sep_states: a tuple (stacked_sep_states, sep_states_lens)
            target_attns: a tuple (stacked_target_attns, target_attns_lens, src_states_target_list)
        """
        # 1. compute the generation loss
        target = target.view(-1)
        align = align.view(-1)
        scores = self.generator(
            self._bottle(output), self._bottle(copy_attn), batch.src_map
        )
        gen_loss = self.criterion(scores, align, target)

        # 2. Compute the orthogonal loss
        orthogonal_loss = torch.zeros(1).to(gen_loss.device)
        if self.add_orthog:
            orthogonal_loss = self._compute_orthogonal_loss(sep_states)

        # 3. Compute the target encoding loss
        target_enc_loss = torch.zeros(1).to(gen_loss.device)
        if self.add_te_loss and target_attns[0] is not None:
            target_enc_loss = self._compute_te_loss(target_attns)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        scores_data = inputters.TextDataset.collapse_copy_scores(
            self._unbottle(scores.clone(), batch.batch_size),
            batch, self.tgt_vocab, batch.dataset.src_vocabs)
        scores_data = self._bottle(scores_data)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0
        target_data = target.clone()
        unk = self.criterion.unk_index
        correct_mask = (target_data == unk) & (align != unk)
        offset_align = align[correct_mask] + len(self.tgt_vocab)
        target_data[correct_mask] += offset_align

        # this part looks like it belongs in CopyGeneratorLoss
        if self.normalize_by_length:
            raise NotImplementedError
            # # Compute Loss as NLL divided by seq length
            # tgt_lens = batch.tgt.ne(self.padding_idx).sum(0).float()
            # # Compute Total Loss per sequence in batch
            # gen_loss = gen_loss.view(-1, batch.batch_size).sum(0)
            # # Divide by length of each sequence and sum
            # gen_loss = torch.div(gen_loss, tgt_lens).sum()
        else:
            gen_loss = gen_loss.sum()
            orthogonal_loss = orthogonal_loss.sum()
            target_enc_loss = target_enc_loss.sum()

        # Compute the total loss
        total_loss = gen_loss + self.lambda_orthogonal * orthogonal_loss + self.lambda_te * target_enc_loss

        # Compute sum of perplexities for stats
        coverage_loss=0.0
        stats = self._stats(gen_loss.clone().item(),
                            scores_data, target_data,
                            orthogonal_loss.clone().item(),
                            coverage_loss,
                            target_enc_loss.clone().item(),
                            total_loss.clone().item(),
                            batch.batch_size)
        return total_loss, stats

    # def _stats(self, loss, scores, target, orthogonal_loss=0.0, te_loss=0.0, total_loss=0.0, b_size=0):
    #     """
    #     Args:
    #         loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
    #         scores (:obj:`FloatTensor`): a score for each possible output
    #         target (:obj:`FloatTensor`): true targets
    #         orthogonal_loss (:obj:`FloatTensor'): the computed orthogonal loss
    #         orthogonal_loss (:obj:`FloatTensor'): the computed orthogonal loss
    #         te_loss (:obj:`Float'): the computed target encoding loss
    #         total_loss (:obj:`FloatTensor'): the computed total loss: loss + lambda_orthogonal * orthogonal_loss
    #         b_size: the batch size
    #     Returns:
    #         :obj:`onmt.utils.Statistics` : statistics for this batch.
    #     """
    #     pred = scores.max(1)[1]
    #     non_padding = target.ne(self.padding_idx)
    #     num_correct = pred.eq(target).masked_select(non_padding).sum().item()
    #     num_non_padding = non_padding.sum().item()
    #     return onmt.utils.Statistics(loss, num_non_padding, num_correct, orthogonal_loss, cover_loss, te_loss, total_loss, b_size)