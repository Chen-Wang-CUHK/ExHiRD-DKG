""" Translation main class """
from __future__ import unicode_literals, print_function

import torch
import itertools
from torchtext.data import Field, NestedField


class HRTranslationBuilder(object):
    """
    Build a word-based hierarchical translation from the batch output
    of translator and the underlying dictionaries.

    Replacement based on "Addressing the Rare Word
    Problem in Neural Machine Translation" :cite:`Luong2015b`

    Args:
       data (DataSet):
       fields (dict of Fields): data fields
       n_best (int): number of translations produced
       replace_unk (bool): replace unknown words using attention
       has_tgt (bool): whether the batch have gold targets
    """

    def __init__(self, data, fields, n_best=1, replace_unk=False, has_tgt=False,
                 eokp_token='<eokp>', sep_token=';', p_end_token='P_END', a_end_token='A_END'):
        self.data = data
        self.fields = fields
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.has_tgt = has_tgt
        self.eokp_token = eokp_token
        self.sep_token = sep_token
        self.p_end_token = p_end_token
        self.a_end_token = a_end_token

    def _build_target_tokens(self, src, src_vocab, src_raw, pred, attn):
        tgt_field = self.fields["tgt"][0][1]
        src_field = self.fields["src"][0][1]
        vocab = tgt_field.vocab
        tokens = []
        # src [s_num, s_len]
        # pred: [tgt_s_num, max_kp_length]
        # attn: [tgt_s_num, max_kp_length, s_num * s_len]
        tgt_s_num, max_kp_length = pred.size()
        for i, phrase in enumerate(pred.split(1, dim=0)):
            phrase = phrase.squeeze(0)
            for j, tok_id in enumerate(phrase.split(1, dim=0)):
                if tok_id < len(vocab):
                    tok = vocab.itos[tok_id]
                else:
                    tok = src_vocab.itos[tok_id - len(vocab)]
                # replace unk token if required
                if tok == tgt_field.unk_token and self.replace_unk and attn is not None and src is not None:
                    # [s_num * s_len] or [src_length]
                    attn_ij = attn[i, j]
                    if src.dim() == 1:
                        _, max_index = attn_ij.max(0)
                        tok = src_raw[max_index.item()]
                    else:
                        flat_src = list(itertools.chain.from_iterable(src_raw))
                        # [s_num * s_len]
                        valid_src = src.view(-1) != src_field.vocab.stoi[src_field.pad_token]
                        assert valid_src.sum().item() == sum([len(src_sent_tmp) for src_sent_tmp in src_raw])
                        _, hr_max_index = attn_ij.max(0)
                        raw_max_index = valid_src[:hr_max_index].sum()
                        tok = flat_src[raw_max_index.item()]
                tokens.append(tok)
                if tok in [self.sep_token, tgt_field.eos_token, self.eokp_token, self.p_end_token, self.a_end_token]:
                    break
                if j == max_kp_length - 1:
                    tokens.append(self.sep_token)
            if tokens[-1] == tgt_field.eos_token:
                tokens = tokens[:-1]
                break
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert(len(translation_batch["gold_score"]) ==
               len(translation_batch["predictions"]))
        assert self.n_best == 1
        batch_size = batch.batch_size

        # predictions: [[tgt_s_num, max_kp_length], ...]
        # scores: [[1], ...]
        # attention: [[tgt_s_num, max_kp_length, s_num * s_len], ...]
        # gold_score: [batch_size]
        preds, pred_score, attn, gold_score, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["gold_score"],
                        batch.indices.data),
                    key=lambda x: x[-1])))

        # Sorting
        inds, perm = torch.sort(batch.indices)
        data_type = self.data.data_type
        assert data_type == 'text'
        if isinstance(self.fields['src'][0][1], NestedField):
            # [batch_size, s_num, s_len]
            src = batch.src[0].index_select(0, perm)
        else:
            # [src_length, batch_size]
            src = batch.src[0].index_select(1, perm)
            # [batch_size, src_length]
            src = src.transpose(0, 1)

        if self.has_tgt:
            if isinstance(self.fields['tgt'][0][1], NestedField):
                # [batch_size, tgt_s_num. tgt_s_len]
                tgt = batch.tgt.index_select(0, perm)
            else:
                # [tgt_length, batch_size]
                tgt = batch.tgt.index_select(1, perm)
                # [batch_size, tgt_length]
                tgt = tgt.transpose(0, 1)
        else:
            tgt = None

        translations = []
        for b in range(batch_size):
            src_vocab = self.data.src_vocabs[inds[b]] \
                if self.data.src_vocabs else None
            src_raw = self.data.examples[inds[b]].src

            pred_sents = self._build_target_tokens(
                src[b] if src is not None else None,
                src_vocab, src_raw,
                preds[b], attn[b])

            gold_sent = None
            if tgt is not None:
                gold_sent = self._build_target_tokens(
                    src[b] if src is not None else None,
                    src_vocab, src_raw,
                    tgt[b, :, 1:] if tgt is not None else None, None)

            translation = Translation(
                src[b] if src is not None else None,
                src_raw, pred_sents, attn[b], pred_score[b],
                gold_sent, gold_score[b]
            )
            translations.append(translation)

        return translations


class TranslationBuilder(object):
    """
    Build a word-based translation from the batch output
    of translator and the underlying dictionaries.

    Replacement based on "Addressing the Rare Word
    Problem in Neural Machine Translation" :cite:`Luong2015b`

    Args:
       data (DataSet):
       fields (dict of Fields): data fields
       n_best (int): number of translations produced
       replace_unk (bool): replace unknown words using attention
       has_tgt (bool): will the batch have gold targets
    """

    def __init__(self, data, fields, n_best=1, replace_unk=False,
                 has_tgt=False):
        self.data = data
        self.fields = fields
        self.n_best = n_best
        self.replace_unk = replace_unk
        self.has_tgt = has_tgt

    def _build_target_tokens(self, src, src_vocab, src_raw, pred, attn):
        tgt_field = self.fields["tgt"][0][1]
        src_field = self.fields["src"][0][1]
        vocab = tgt_field.vocab
        tokens = []
        for tok in pred:
            if tok < len(vocab):
                tokens.append(vocab.itos[tok])
            else:
                tokens.append(src_vocab.itos[tok - len(vocab)])
            if tokens[-1] == tgt_field.eos_token:
                tokens = tokens[:-1]
                break
        if self.replace_unk and attn is not None and src is not None:
            for i in range(len(tokens)):
                if tokens[i] == tgt_field.unk_token:
                    if src.dim() == 1:
                        _, max_index = attn[i].max(0)
                        tokens[i] = src_raw[max_index.item()]
                    else:
                        flat_src = list(itertools.chain.from_iterable(src_raw))
                        # [s_num * s_len]
                        valid_src = src.view(-1) != src_field.vocab.stoi[src_field.pad_token]
                        assert valid_src.sum().item() == sum([len(src_sent_tmp) for src_sent_tmp in src_raw])
                        _, max_index = attn[i].max(0)
                        raw_max_index = valid_src[:max_index].sum()
                        tokens[i] = flat_src[raw_max_index.item()]
        return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert(len(translation_batch["gold_score"]) ==
               len(translation_batch["predictions"]))
        batch_size = batch.batch_size

        preds, pred_score, attn, gold_score, indices = list(zip(
            *sorted(zip(translation_batch["predictions"],
                        translation_batch["scores"],
                        translation_batch["attention"],
                        translation_batch["gold_score"],
                        batch.indices.data),
                    key=lambda x: x[-1])))

        # Sorting
        inds, perm = torch.sort(batch.indices)
        data_type = self.data.data_type

        # change by wchen. Use batch first version for src for both seqE and HRE
        if data_type == 'text':
            if len(batch.src) == 2:
                # [src_len, batch_size]
                src = batch.src[0].index_select(1, perm)
                # [batch_size, src_len]
                src = src.transpose(0, 1)
            else:
                # [batch_size, s_num, s_len]
                src = batch.src[0].index_select(0, perm)
        else:
            src = None
        tgt = batch.tgt.index_select(1, perm) if self.has_tgt else None

        translations = []
        for b in range(batch_size):
            if data_type == 'text':
                src_vocab = self.data.src_vocabs[inds[b]] \
                    if self.data.src_vocabs else None
                # [src_len] for seqE
                # [s_num, s_len] for HRE
                src_raw = self.data.examples[inds[b]].src
            else:
                src_vocab = None
                src_raw = None
            # src[:, b] -> src[b]
            pred_sents = [self._build_target_tokens(
                src[b] if src is not None else None,
                src_vocab, src_raw,
                preds[b][n], attn[b][n])
                for n in range(self.n_best)]
            gold_sent = None
            if tgt is not None:
                gold_sent = self._build_target_tokens(
                    src[b] if src is not None else None,
                    src_vocab, src_raw,
                    tgt[1:, b] if tgt is not None else None, None)

            translation = Translation(
                src[b] if src is not None else None,
                src_raw, pred_sents, attn[b], pred_score[b],
                gold_sent, gold_score[b]
            )
            translations.append(translation)

        return translations


class Translation(object):
    """
    Container for a translated sentence.

    Attributes:
        src (`LongTensor`): src word ids
        src_raw ([str]): raw src words

        pred_sents ([[str]]): words from the n-best translations
        pred_scores ([[float]]): log-probs of n-best translations
        attns ([`FloatTensor`]) : attention dist for each translation
        gold_sent ([str]): words from gold translation
        gold_score ([float]): log-prob of gold translation

    """

    def __init__(self, src, src_raw, pred_sents,
                 attn, pred_scores, tgt_sent, gold_score):
        self.src = src
        self.src_raw = src_raw
        self.pred_sents = pred_sents
        self.attns = attn
        self.pred_scores = pred_scores
        self.gold_sent = tgt_sent
        self.gold_score = gold_score

    def log(self, sent_number):
        """
        Log translation.
        """

        output = '\nSENT {}: {}\n'.format(sent_number, self.src_raw)

        best_pred = self.pred_sents[0]
        best_score = self.pred_scores[0]
        pred_sent = ' '.join(best_pred)
        output += 'PRED {}: {}\n'.format(sent_number, pred_sent)
        output += "PRED SCORE: {:.4f}\n".format(best_score)

        if self.gold_sent is not None:
            tgt_sent = ' '.join(self.gold_sent)
            output += 'GOLD {}: {}\n'.format(sent_number, tgt_sent)
            output += ("GOLD SCORE: {:.4f}\n".format(self.gold_score))
        if len(self.pred_sents) > 1:
            output += '\nBEST HYP:\n'
            for score, sent in zip(self.pred_scores, self.pred_sents):
                output += "[{:.4f}] {}\n".format(score, sent)

        return output
