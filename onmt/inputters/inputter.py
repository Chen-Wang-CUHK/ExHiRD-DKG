# -*- coding: utf-8 -*-
import glob
import os
import codecs

from collections import Counter, defaultdict
from itertools import chain, cycle
from functools import partial

import torch
import torchtext.data
from torchtext.data import Field, NestedField
from torchtext.vocab import Vocab

from onmt.inputters.text_dataset import TextDataset
from onmt.inputters.image_dataset import ImageDataset
from onmt.inputters.audio_dataset import AudioDataset
from onmt.utils.logging import logger

import gc

# add by wchen
from nltk.tokenize import sent_tokenize
from onmt.inputters.my_fields import SentPosiField, WordSentIdField
from data_utils import KEY_SEPERATOR


def _getstate(self):
    return dict(self.__dict__, stoi=dict(self.stoi))


def _setstate(self, state):
    self.__dict__.update(state)
    self.stoi = defaultdict(lambda: 0, self.stoi)


Vocab.__getstate__ = _getstate
Vocab.__setstate__ = _setstate


def make_src(data, vocab):
    src_size = max([t.size(0) for t in data])
    src_vocab_size = max([t.max() for t in data]) + 1
    alignment = torch.zeros(src_size, len(data), src_vocab_size)
    for i, sent in enumerate(data):
        for j, t in enumerate(sent):
            alignment[j, i, t] = 1
    return alignment


def make_tgt(data, vocab):
    tgt_size = max([t.size(0) for t in data])
    alignment = torch.zeros(tgt_size, len(data)).long()
    for i, sent in enumerate(data):
        alignment[:sent.size(0), i] = sent
    return alignment


# add by wchen for HRED
def make_hr_src(data, vocab):
    """
    :param data: [[s1,s2,...], ... ,[s1,s2,...]], list of list of tensors
    :param vocab: not used
    :return:
    """
    batch = len(data)
    s_num = max([len(ex) for ex in data])
    s_len = max([max([sent.size(0) for sent in ex]) for ex in data])
    src_vocab_size = max([max([sent.max() for sent in ex]) for ex in data]) + 1
    alignment = torch.zeros(s_num, s_len, batch, src_vocab_size)
    for b, ex in enumerate(data):
        for i, sent in enumerate(ex):
            for j, t in enumerate(sent):
                alignment[i, j, b, t] = 1
    # [s_num * s_len, batch, src_vocab_size]
    alignment = alignment.view(s_num * s_len, batch, src_vocab_size)
    return alignment


# add by wchen for HRED
def make_hr_tgt(data, vocab):
    """
    :param data: [[s1,s2,...], ... ,[s1,s2,...]], list of list tensors
    :param vocab: not used
    :return:
    """
    batch = len(data)
    tgt_s_num = max([len(ex) for ex in data])
    tgt_s_len = max([max([sent.size(0) for sent in ex]) for ex in data])
    alignment = torch.zeros(tgt_s_num, tgt_s_len, batch).long()
    for b, ex in enumerate(data):
        for i, sent in enumerate(ex):
            alignment[i, :sent.size(0), b] = sent
    # # [tgt_s_num * tgt_s_len, batch]
    # alignment = alignment.view(-1, batch)
    return alignment


def make_img(data, vocab):
    c = data[0].size(0)
    h = max([t.size(1) for t in data])
    w = max([t.size(2) for t in data])
    imgs = torch.zeros(len(data), c, h, w).fill_(1)
    for i, img in enumerate(data):
        imgs[i, :, 0:img.size(1), 0:img.size(2)] = img
    return imgs


def make_audio(data, vocab):
    """ batch audio data """
    nfft = data[0].size(0)
    t = max([t.size(1) for t in data])
    sounds = torch.zeros(len(data), 1, nfft, t)
    for i, spect in enumerate(data):
        sounds[i, :, :, 0:spect.size(1)] = spect
    return sounds


# mix this with partial
def _feature_tokenize(
        string, layer=0, tok_delim=None, feat_delim=None, truncate=None):
    tokens = string.split(tok_delim)
    if truncate is not None:
        tokens = tokens[:truncate]
    if feat_delim is not None:
        tokens = [t.split(feat_delim)[layer] for t in tokens]
    return tokens


# add by wchen, hr feature tokenizer
# mix this with partial
def _hr_sent_tokenize(string, sent_delim=None, sent_num_truncate=None, pad_token=None, eokp_token=None, use_bi_end=False):
    if sent_delim is None:
        # src
        sents = sent_tokenize(string.strip())
    else:
        # tgt
        sents = string.strip().split(sent_delim)
        # since an additional pad token is added,
        # the total valid tokens for hr tgt need to change correspondingly
        # the filtering condition also need to change correspondingly
        if use_bi_end:
            sents = [sent.strip() + " " + pad_token for sent in sents if len(sent.strip()) != 0]
        else:
            sents = [sent.strip() + " " + sent_delim + " " + pad_token for sent in sents if len(sent.strip()) != 0]

    if sent_num_truncate is not None:
        sents = sents[:sent_num_truncate]

    # for tgt
    if len(sents) != 0 and eokp_token is not None and sent_delim is not None:
        # assert eokp_token in sents[-1], "{} token should be the ending token of the tgt!"
        # <eokp> is the ending token of original sents[-1], we do not add extra KEY_SEPERATOR
        # to indicate the ending of the keyphrase
        sents[-1] = sents[-1].replace(" " + sent_delim + " " + pad_token, " " + pad_token)

    return sents


def _my_tokenize(string, sent_delim=None, token_delim=None, sent_num_truncate=None, word_num_truncate=None):
    if sent_delim is None:
        # src
        sents = sent_tokenize(string.strip())
    else:
        # tgt
        sents = string.strip().split(sent_delim)

    if sent_num_truncate is not None:
        sents = sents[:sent_num_truncate]

    # word tokenization
    if word_num_truncate is not None:
        sents = [sent_i.strip().split(token_delim)[:word_num_truncate] for sent_i in sents]
    else:
        sents = [sent_i.strip().split(token_delim) for sent_i in sents]

    return list(chain.from_iterable(sents))


def _sent_posi_prep(string, sent_delim=None, token_delim=None, sent_num_truncate=None, word_num_truncate=None):
    if sent_delim is None:
        # src
        sents = sent_tokenize(string.strip())
    else:
        sents = string.strip().split(sent_delim)

    if sent_num_truncate is not None:
        sents = sents[:sent_num_truncate]

    if word_num_truncate is not None:
        sents = [sent_i.strip().split(token_delim)[:word_num_truncate] for sent_i in sents]
    else:
        sents = [sent_i.strip().split(token_delim) for sent_i in sents]

    if len(sents) != 0:
        sent_lens = [len(sent_i) for sent_i in sents]
        # get the forward sentence ending position
        f_sent_end_position = [sum(sent_lens[:i]) - 1 for i in range(1, len(sents) + 1)]
        # get the backward sentence ending position
        b_sent_end_position = [0] + [sum(sent_lens[:i]) for i in range(1, len(sents))]
        # merge
        assert len(f_sent_end_position) == len(b_sent_end_position)
        sent_end_position = [[fp, bp] for fp, bp in zip(f_sent_end_position, b_sent_end_position)]
        return sent_end_position
    else:
        return None


def _word_sent_id_prep(string, sent_delim=None, token_delim=None, sent_num_truncate=None, word_num_truncate=None):
    if sent_delim is None:
        # src
        sents = sent_tokenize(string.strip())
    else:
        sents = string.strip().split(sent_delim)

    if sent_num_truncate is not None:
        sents = sents[:sent_num_truncate]

    if word_num_truncate is not None:
        sents = [sent_i.strip().split(token_delim)[:word_num_truncate] for sent_i in sents]
    else:
        sents = [sent_i.strip().split(token_delim) for sent_i in sents]

    if len(sents) != 0:
        word_sent_ids = [i for i, sent_i in enumerate(sents) for _ in sent_i]
        return word_sent_ids
    else:
        return None


# def _sent_posi_postp(data):
#     """
#     postprocessing for the sent position data
#     :param data: [[[b1_s1_fp, b1_s1_bp], [b1_s2_fp, b1_s2_bp], ...], ...], list of examples' sent positions
#     :return: a padded list, we use [0, 0] to pad the sentence position of each example
#     """
#     assert isinstance(data, list)
#     max_sent_num = max([len(ex) for ex in data])
#     padded_data = [ex + [[0, 0]] * (max_sent_num - len(ex)) for ex in data]
#     return padded_data


def get_fields(
    src_data_type,
    n_src_feats,
    n_tgt_feats,
    pad='<blank>',
    bos='<s>',
    eos='</s>',
    dynamic_dict=False,
    src_truncate=None,
    tgt_truncate=None
):
    """
    src_data_type: type of the source input. Options are [text|img|audio].
    n_src_feats, n_tgt_feats: the number of source and target features to
        create a `torchtext.data.Field` for.
    pad, bos, eos: special symbols to use for fields.
    returns: A dictionary. The keys are strings whose names correspond to the
        keys of the dictionaries yielded by the make_examples methods of
        various dataset classes. The values are lists of (name, Field)
        pairs, where the name is a string which will become the name of
        an attribute of an example.
    """
    assert src_data_type in ['text', 'img', 'audio'], \
        "Data type not implemented"
    assert not dynamic_dict or src_data_type == 'text', \
        'it is not possible to use dynamic_dict with non-text input'
    fields = {'src': [], 'tgt': []}

    if src_data_type == 'text':
        feat_delim = u"￨" if n_src_feats > 0 else None
        for i in range(n_src_feats + 1):
            name = "src_feat_" + str(i - 1) if i > 0 else "src"
            tokenize = partial(
                _feature_tokenize,
                layer=i,
                truncate=src_truncate,
                feat_delim=feat_delim)
            use_len = i == 0
            feat = Field(
                pad_token=pad, tokenize=tokenize, include_lengths=use_len)
            fields['src'].append((name, feat))
    elif src_data_type == 'img':
        img = Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_img, sequential=False)
        fields["src"].append(('src', img))
    else:
        audio = Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_audio, sequential=False)
        fields["src"].append(('src', audio))

    if src_data_type == 'audio':
        # only audio has src_lengths
        length = Field(use_vocab=False, dtype=torch.long, sequential=False)
        fields["src_lengths"] = [("src_lengths", length)]

    # below this: things defined no matter what the data source type is
    feat_delim = u"￨" if n_tgt_feats > 0 else None
    for i in range(n_tgt_feats + 1):
        name = "tgt_feat_" + str(i - 1) if i > 0 else "tgt"
        tokenize = partial(
            _feature_tokenize,
            layer=i,
            truncate=tgt_truncate,
            feat_delim=feat_delim)

        feat = Field(
            init_token=bos,
            eos_token=eos,
            pad_token=pad,
            tokenize=tokenize)
        fields['tgt'].append((name, feat))

    indices = Field(use_vocab=False, dtype=torch.long, sequential=False)
    fields["indices"] = [('indices', indices)]

    if dynamic_dict:
        src_map = Field(
            use_vocab=False, dtype=torch.float,
            postprocessing=make_src, sequential=False)
        fields["src_map"] = [("src_map", src_map)]

        align = Field(
            use_vocab=False, dtype=torch.long,
            postprocessing=make_tgt, sequential=False)
        fields["alignment"] = [('alignment', align)]

    return fields


# wchen: use NestedField for both src and tgt
def get_nested_fields(
    src_data_type,
    n_src_feats,
    n_tgt_feats,
    pad='<blank>',
    bos='<s>',
    eos='</s>',
    eokp=None,
    dynamic_dict=False,
    src_truncate=None,
    src_sent_num_truncate=None,
    src_sent_length_truncate=None,
    tgt_truncate=None,
    seq_hr_src=False,
    hr_src=False,
    hr_tgt=False,
    use_bi_end=False,
    contain_title=False
):
    """
    src_data_type: type of the source input. Options are [text].
    n_src_feats, n_tgt_feats: the number of source and target features to
        create a `torchtext.data.Field` or a `torchtext.data.NestedField` for.
    pad, bos, eos: special symbols to use for fields.
    returns: A dictionary. The keys are strings whose names correspond to the
        keys of the dictionaries yielded by the make_examples methods of
        various dataset classes. The values are lists of (name, Field)
        pairs, where the name is a string which will become the name of
        an attribute of an example.
    """
    assert src_data_type in ['text'], \
        "Data type not implemented"
    assert not dynamic_dict or src_data_type == 'text', \
        'it is not possible to use dynamic_dict with non-text input'
    fields = {'src': [], 'tgt': []}

    if contain_title:
        fields['title'] = []

    feat_delim = u"￨" if n_src_feats > 0 else None
    for i in range(n_src_feats + 1):
        name = "src_feat_" + str(i - 1) if i > 0 else "src"
        use_len = i == 0
        if hr_src:
            # nesting field
            assert not seq_hr_src
            nesting_tokenize = partial(
                _feature_tokenize,
                layer=i,
                truncate=src_sent_length_truncate,
                feat_delim=feat_delim)
            nesting_field = Field(pad_token=pad, tokenize=nesting_tokenize)
            # nested field
            nested_tokenize = partial(
                _hr_sent_tokenize,
                sent_num_truncate=src_sent_num_truncate)
            feat = NestedField(nesting_field, include_lengths=use_len, tokenize=nested_tokenize)
        else:
            # changed from "_feature_tokenize" to "_my_tokenize" to make sure
            # the sequentially tokenized src be the same as the obtained tokens with _hr_sent_tokenize
            assert src_truncate is None, "If you want to sequentially truncate src by tokens," \
                                         "please use _feature_tokenize as the following comment lines."
            nesting_tokenize = partial(
                _my_tokenize,
                sent_delim=None, token_delim=None,
                sent_num_truncate=src_sent_num_truncate,
                word_num_truncate=src_sent_length_truncate)
            # nesting_tokenize = partial(
            #     _feature_tokenize,
            #     layer=i,
            #     truncate=src_truncate,
            #     feat_delim=feat_delim)
            feat = Field(pad_token=pad, tokenize=nesting_tokenize, include_lengths=use_len)
        fields['src'].append((name, feat))

        # for title
        if contain_title:
            name = "title_feat_" + str(i - 1) if i > 0 else "title"
            use_len = i == 0
            title_tokenize = partial(_my_tokenize, sent_delim=None, token_delim=None,
                                     sent_num_truncate=None, word_num_truncate=None)
            title_feat = Field(pad_token=pad, tokenize=title_tokenize, include_lengths=use_len)
            fields['title'].append((name, title_feat))


    # TODO: use hierarchical decoding structure
    # below this: things defined no matter what the data source type is
    feat_delim = u"￨" if n_tgt_feats > 0 else None
    for i in range(n_tgt_feats + 1):
        name = "tgt_feat_" + str(i - 1) if i > 0 else "tgt"
        tokenize = partial(
            _feature_tokenize,
            layer=i,
            truncate=tgt_truncate,
            feat_delim=feat_delim)

        if hr_tgt:
            nesting_field = Field(
                init_token=bos,
                pad_token=pad,
                tokenize=tokenize)
            nested_tokenize = partial(_hr_sent_tokenize,
                                      sent_delim=KEY_SEPERATOR, pad_token=pad, eokp_token=eokp, use_bi_end=use_bi_end)
            feat = NestedField(nesting_field, eos_token=eos, tokenize=nested_tokenize)
        else:
            nesting_field = Field(
                init_token=bos,
                eos_token=eos,
                pad_token=pad,
                tokenize=tokenize)
            feat = nesting_field
        fields['tgt'].append((name, feat))

    if seq_hr_src:
        # for sentence start and end position information
        sent_posi_prep = partial(
                _sent_posi_prep,
                sent_delim=None, token_delim=None,
                sent_num_truncate=src_sent_num_truncate,
                word_num_truncate=src_sent_length_truncate)
        src_sent_position = SentPosiField(preprocessing=sent_posi_prep,
                                          use_vocab=False,
                                          dtype=torch.long,
                                          sequential=False,
                                          include_lengths=True)
        fields["src_sent_position"] = [("src_sent_position", src_sent_position)]

        # for sentence id information
        word_sent_id_prep = partial(
                _word_sent_id_prep,
                sent_delim=None, token_delim=None,
                sent_num_truncate=src_sent_num_truncate,
                word_num_truncate=src_sent_length_truncate)
        src_word_sent_ids = WordSentIdField(preprocessing=word_sent_id_prep,
                                            use_vocab=False,
                                            dtype=torch.long,
                                            sequential=False,
                                            include_lengths=True)
        fields["src_word_sent_ids"] = [("src_word_sent_ids", src_word_sent_ids)]

    indices = Field(use_vocab=False, dtype=torch.long, sequential=False)
    fields["indices"] = [('indices', indices)]

    # currently the copy mechanism is not available for HRED
    # if hr_tgt:
    #     assert not dynamic_dict, "Currently the copy mechanism is not available for HRED"
    if dynamic_dict:
        if not hr_src:
            src_map = Field(
                use_vocab=False, dtype=torch.float,
                postprocessing=make_src, sequential=False)
        else:
            src_map = Field(
                use_vocab=False, dtype=torch.float,
                postprocessing=make_hr_src, sequential=False)
        fields["src_map"] = [("src_map", src_map)]

        if not hr_tgt:
            align = Field(
                use_vocab=False, dtype=torch.long,
                postprocessing=make_tgt, sequential=False)
        else:
            align = Field(
                use_vocab=False, dtype=torch.long,
                postprocessing=make_hr_tgt, sequential=False, batch_first=True)
        fields["alignment"] = [('alignment', align)]

    return fields


def load_fields_from_vocab(vocab, data_type="text"):
    """
    vocab: a list of (field name, torchtext.vocab.Vocab) pairs
    data_type: text, img, or audio
    returns: a dictionary whose keys are the field names and whose values
             are field objects with the vocab set to the corresponding vocab
             object from the input.
    """
    vocab = dict(vocab)
    n_src_features = sum('src_feat_' in k for k in vocab)
    n_tgt_features = sum('tgt_feat_' in k for k in vocab)
    # changed by wchen
    # fields = get_fields(data_type, n_src_features, n_tgt_features)
    fields = get_nested_fields(data_type, n_src_features, n_tgt_features)

    for k, vals in fields.items():
        for n, f in vals:
            if n in vocab:
                f.vocab = vocab[n]
    return fields


def old_style_vocab(vocab):
    """
    vocab: some object loaded from a *.vocab.pt file
    returns: whether the object is a list of pairs where the second object
        is a torchtext.vocab.Vocab object.

    This exists because previously only the vocab objects from the fields
    were saved directly, not the fields themselves, and the fields needed to
    be reconstructed at training and translation time.
    """
    return isinstance(vocab, list) and \
        any(isinstance(v[1], Vocab) for v in vocab)


def make_features(batch, side, data_type='text'):
    """
    Args:
        batch (Tensor): a batch of source or target data.
        side (str): for source or for target.
        data_type (str): type of the source input.
            Options are [text|img|audio].
    Returns:
        A sequence of src/tgt tensors with optional feature tensors
        of size (len x batch).
    """
    assert side in ['src', 'tgt', 'title']
    if isinstance(batch.__dict__[side], tuple):
        data = batch.__dict__[side][0]
    else:
        data = batch.__dict__[side]

    feat_start = side + "_feat_"
    keys = sorted([k for k in batch.__dict__ if feat_start in k])
    features = [batch.__dict__[k] for k in keys]
    levels = [data] + features

    if data_type == 'text':
        # changed by wchen: 2 -> -1
        return torch.cat([level.unsqueeze(-1) for level in levels], -1)
    else:
        return levels[0]


def filter_example(ex, use_src_len=True, use_tgt_len=True,
                   min_src_len=1, max_src_len=float('inf'),
                   min_tgt_len=1, max_tgt_len=float('inf')):
    """
    A generalized function for filtering examples based on the length of their
    src or tgt values. Rather than being used by itself as the filter_pred
    argument to a dataset, it should be partially evaluated with everything
    specified except the value of the example.
    """
    # add by wchen
    # src length
    if len(ex.src) == 0:
        src_len = 0
    elif type(ex.src[0]) is list:
        src_len = sum([len(sent_i) for sent_i in ex.src])
    else:
        src_len = len(ex.src)
    # tgt length
    if len(ex.tgt) == 0:
        tgt_len = 0
    elif type(ex.tgt[0]) is list:
        # since we add an extra pad token for each keyphrase, we discard it when we count valid tgt tokens
        tgt_len = sum([len(tgt_i[:-1]) for tgt_i in ex.tgt])
    else:
        tgt_len = len(ex.tgt)
    return (not use_src_len or min_src_len <= src_len <= max_src_len) and \
        (not use_tgt_len or min_tgt_len <= tgt_len <= max_tgt_len)


def build_dataset(fields, data_type, src,
                  src_dir=None, title=None, tgt=None,
                  src_seq_len=50, tgt_seq_len=50,
                  sample_rate=0, window_size=0, window_stride=0, window=None,
                  normalize_audio=True, use_filter_pred=True,
                  image_channel_size=3, src_seq_min_length=1, tgt_seq_min_length=1):
    """
    src: path to corpus file or iterator over source data
    tgt: path to corpus file, iterator over target data, or None
    seq_hr_src: add by wchen, whether to include the sentence forward and backward positions
    """
    dataset_classes = {
        'text': TextDataset, 'img': ImageDataset, 'audio': AudioDataset
    }
    assert data_type in dataset_classes
    assert src is not None
    if data_type == 'text':
        src_examples_iter = TextDataset.make_examples(src, "src")
    elif data_type == 'img':
        # there is a truncate argument as well, but it was never set to
        # anything besides None before
        src_examples_iter = ImageDataset.make_examples(
            src, src_dir, 'src', channel_size=image_channel_size
        )
    else:
        src_examples_iter = AudioDataset.make_examples(
            src, src_dir, "src", sample_rate,
            window_size, window_stride, window,
            normalize_audio, None)

    if title is None:
        title_examples_iter = None
    else:
        title_examples_iter = TextDataset.make_examples(title, "title")

    if tgt is None:
        tgt_examples_iter = None
    else:
        tgt_examples_iter = TextDataset.make_examples(tgt, "tgt")

    # the second conjunct means nothing will be filtered at translation time
    # if there is no target data
    if use_filter_pred and tgt_examples_iter is not None:
        filter_pred = partial(
            filter_example, use_src_len=data_type == 'text',
            max_src_len=src_seq_len, max_tgt_len=tgt_seq_len,
            min_src_len=src_seq_min_length, min_tgt_len=tgt_seq_min_length
        )
    else:
        filter_pred = None

    dataset_cls = dataset_classes[data_type]
    dataset = dataset_cls(
        fields, src_examples_iter, title_examples_iter, tgt_examples_iter, filter_pred=filter_pred)
    return dataset


def _build_field_vocab(field, counter, **kwargs):
    # this is basically copy-pasted from torchtext.
    # changed by wchen for NestedField
    if isinstance(field, NestedField):
        unk_token = field.unk_token if field.unk_token is not None else field.nesting_field.unk_token
        pad_token = field.pad_token if field.pad_token is not None else field.nesting_field.pad_token
        init_token = field.init_token if field.init_token is not None else field.nesting_field.init_token
        eos_token = field.eos_token if field.eos_token is not None else field.nesting_field.eos_token
        all_specials = [unk_token, pad_token, init_token, eos_token]
    else:
        all_specials = [
            field.unk_token, field.pad_token, field.init_token, field.eos_token
        ]
    specials = [tok for tok in all_specials if tok is not None]
    field.vocab = field.vocab_cls(counter, specials=specials, **kwargs)
    # add by wchen: if "field" is a NestedField, then the vocab should be shared with the nesting field
    if type(field) is NestedField:
        field.nesting_field.vocab = field.vocab


def build_vocab(train_dataset_files, fields, data_type, share_vocab,
                src_vocab_path, src_vocab_size, src_words_min_frequency,
                tgt_vocab_path, tgt_vocab_size, tgt_words_min_frequency):
    """
    Args:
        train_dataset_files: a list of train dataset pt file.
        fields (dict): fields to build vocab for.
        data_type: "text", "img" or "audio"?
        share_vocab(bool): share source and target vocabulary?
        src_vocab_path(string): Path to src vocabulary file.
        src_vocab_size(int): size of the source vocabulary.
        src_words_min_frequency(int): the minimum frequency needed to
                include a source word in the vocabulary.
        tgt_vocab_path(string): Path to tgt vocabulary file.
        tgt_vocab_size(int): size of the target vocabulary.
        tgt_words_min_frequency(int): the minimum frequency needed to
                include a target word in the vocabulary.

    Returns:
        Dict of Fields
    """
    counters = {k: Counter() for k, v in chain.from_iterable(fields.values())}

    # Load vocabulary
    if src_vocab_path:
        src_vocab = _read_vocab_file(src_vocab_path, "src")
        src_vocab_size = len(src_vocab)
        logger.info('Loaded source vocab has %d tokens.' % src_vocab_size)
        for i, token in enumerate(src_vocab):
            # keep the order of tokens specified in the vocab file by
            # adding them to the counter with decreasing counting values
            counters['src'][token] = src_vocab_size - i
    else:
        src_vocab = None

    if tgt_vocab_path:
        tgt_vocab = _read_vocab_file(tgt_vocab_path, "tgt")
        tgt_vocab_size = len(tgt_vocab)
        logger.info('Loaded source vocab has %d tokens.' % tgt_vocab_size)
        for i, token in enumerate(tgt_vocab):
            counters['tgt'][token] = tgt_vocab_size - i
    else:
        tgt_vocab = None

    for i, path in enumerate(train_dataset_files):
        dataset = torch.load(path)
        logger.info(" * reloading %s." % path)
        for ex in dataset.examples:
            for name, field in chain.from_iterable(fields.values()):
                has_vocab = (name == 'src' and src_vocab) or \
                    (name == 'tgt' and tgt_vocab)
                if field.sequential and not has_vocab:
                    val = getattr(ex, name, None)
                    # changed by wchen for HRED
                    if type(val[0]) is list:
                        for val_i in val:
                            counters[name].update(val_i)
                    else:
                        counters[name].update(val)

        # Drop the none-using from memory but keep the last
        if i < len(train_dataset_files) - 1:
            dataset.examples = None
            gc.collect()
            del dataset.examples
            gc.collect()
            del dataset
            gc.collect()

    for name, field in fields["tgt"]:
        _build_field_vocab(field, counters[name])
        logger.info(" * %s vocab size: %d." % (name, len(field.vocab)))
    if data_type == 'text':
        for name, field in fields["src"]:
            _build_field_vocab(field, counters[name])
            logger.info(" * %s vocab size: %d." % (name, len(field.vocab)))
        if share_vocab:
            # `tgt_vocab_size` is ignored when sharing vocabularies
            logger.info(" * merging src and tgt vocab...")
            src_field = fields['src'][0][1]
            tgt_field = fields['tgt'][0][1]
            _merge_field_vocabs(
                src_field, tgt_field, vocab_size=src_vocab_size,
                min_freq=src_words_min_frequency)
            if 'title' in fields:
                title_field = fields['title'][0][1]
                title_field.vocab = src_field.vocab
            logger.info(" * merged vocab size: %d." % len(src_field.vocab))

    return fields  # is the return necessary?


def _merge_field_vocabs(src_field, tgt_field, vocab_size, min_freq):
    # in the long run, shouldn't it be possible to do this by calling
    # build_vocab with both the src and tgt data?
    # changed by wchen for NestedField
    if isinstance(tgt_field, NestedField):
        unk_token = tgt_field.unk_token if tgt_field.unk_token is not None else tgt_field.nesting_field.unk_token
        pad_token = tgt_field.pad_token if tgt_field.pad_token is not None else tgt_field.nesting_field.pad_token
        init_token = tgt_field.init_token if tgt_field.init_token is not None else tgt_field.nesting_field.init_token
        eos_token = tgt_field.eos_token if tgt_field.eos_token is not None else tgt_field.nesting_field.eos_token
        specials = [unk_token, pad_token, init_token, eos_token]
    else:
        specials = [tgt_field.unk_token, tgt_field.pad_token,
                    tgt_field.init_token, tgt_field.eos_token]
    merged = sum(
        [src_field.vocab.freqs, tgt_field.vocab.freqs], Counter()
    )
    merged_vocab = Vocab(
        merged, specials=specials,
        max_size=vocab_size, min_freq=min_freq
    )
    src_field.vocab = merged_vocab
    # add by wchen
    if type(src_field) is NestedField:
        src_field.nesting_field.vocab = merged_vocab
    tgt_field.vocab = merged_vocab
    # add by wchen
    if type(tgt_field) is NestedField:
        tgt_field.nesting_field.vocab = merged_vocab
    assert len(src_field.vocab) == len(tgt_field.vocab)


def _read_vocab_file(vocab_path, tag):
    """
    Loads a vocabulary from the given path.
    :param vocabulary_path: path to load vocabulary from
    :param tag: tag for vocabulary (only used for logging)
    :return: vocabulary or None if path is null
    """
    logger.info("Loading {} vocabulary from {}".format(tag, vocab_path))

    if not os.path.exists(vocab_path):
        raise RuntimeError(
            "{} vocabulary not found at {}".format(tag, vocab_path))
    else:
        with codecs.open(vocab_path, 'r', 'utf-8') as f:
            return [line.strip().split()[0] for line in f if line.strip()]


class OrderedIterator(torchtext.data.Iterator):

    def create_batches(self):
        """ Create batches """
        if self.train:
            def _pool(data, random_shuffler):
                for p in torchtext.data.batch(data, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = _pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


class DatasetLazyIter(object):
    """
    dataset_paths: a list containing the locations of datasets
    fields (dict): fields dict for the datasets.
    batch_size (int): batch size.
    batch_size_fn: custom batch process function.
    device: the GPU device.
    is_train (bool): train or valid?
    """

    def __init__(self, dataset_paths, fields, batch_size, batch_size_fn,
                 device, is_train):
        self._paths = dataset_paths
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.device = device
        self.is_train = is_train

    def __iter__(self):
        paths = cycle(self._paths) if self.is_train else self._paths
        for path in paths:
            cur_dataset = torch.load(path)
            logger.info('Loading dataset from %s, number of examples: %d' %
                        (path, len(cur_dataset)))
            cur_dataset.fields = self.fields

            # add by wchen
            if type(self.fields['src']) is NestedField:
                sort_within_batch = False
            else:
                sort_within_batch = True

            cur_iter = OrderedIterator(
                dataset=cur_dataset,
                batch_size=self.batch_size,
                batch_size_fn=self.batch_size_fn,
                device=self.device,
                train=self.is_train,
                sort=False,
                sort_within_batch=sort_within_batch,
                repeat=False
            )
            for batch in cur_iter:
                yield batch

            cur_dataset.examples = None
            gc.collect()
            del cur_dataset
            gc.collect()


def max_tok_len(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch, max_tgt_in_batch
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    # Src: <bos> w1 ... wN <eos>
    max_src_in_batch = max(max_src_in_batch, len(new.src) + 2)
    # Tgt: w1 ... wN <eos>
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 1)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def build_dataset_iter(corpus_type, fields, opt, is_train=True):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over. We implement simple ordered iterator strategy here,
    but more sophisticated strategy like curriculum learning is ok too.
    """
    dataset_paths = sorted(glob.glob(opt.data + '.' + corpus_type + '*.pt'))
    batch_size = opt.batch_size if is_train else opt.valid_batch_size
    batch_fn = max_tok_len if is_train and opt.batch_type == "tokens" else None

    device = "cuda" if opt.gpu_ranks else "cpu"

    # add by wchen: if the field is nested, check whether the vocab is shared with the nesting_field
    for side in ['src', 'tgt']:
        f = fields[side]
        if type(f) is NestedField:
            assert f.vocab is f.nesting_field.vocab, "The vocab is not shared in the {} NestedField.".format(side)

    return DatasetLazyIter(dataset_paths, fields, batch_size, batch_fn,
                           device, is_train)
