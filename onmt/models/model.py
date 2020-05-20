""" Onmt NMT Model base class definition """
import torch.nn as nn


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, sent_position_tuple=None, src_word_sent_ids=None, bptt=False, validation=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            bptt (:obj:`Boolean`):
                a flag indicating if truncated bptt is set. If reset then
                init_state

        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, lengths)
        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths)
        return dec_out, attns


# add by wchen
class HREDModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a hierarchical encoder + a flattened decoder model,
    or a flattened encoder + a hierarchical decoder,
    or a hierarchical encoder + a hierarchical decoder

    Args:
      encoder (:obj:`EncoderBase`): a flattened or hierarchical encoder object
      decoder (:obj:`RNNDecoderBase`): a flattened or hierarchical decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder):
        super(HREDModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, sent_position_tuple=None, src_word_sent_ids=None, bptt=False, validation=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[src_length x batch x features]` for seq enc.
                of size `[batch x sent_nums x sent_lens x features]` for hr enc.
            tgt (:obj:`LongTensor`):
                a target sequence of size `[tgt_len x batch]` for seq decoder.
                a target sequence of size `[batch x tgt_s_num x tgt_s_len x feat_num]` for hr decoder.
            lengths(:obj:`Tuple'):
                a tuple that consists of sent_nums and sent_lens, (sent_nums, sent_lens).
                Both sent_nums [batch] and sent_lens [batch, sent_nums] are `LongTensor``.
            bptt (:obj:`Boolean`):
                a flag indicating if truncated bptt is set. If reset then
                init_state

        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
        """
        # exclude last target from inputs
        if tgt.dim() == 3:
            # for seq decoder
            tgt = tgt[:-1]
        else:
            # for hr decoder
            assert tgt.dim() == 4
            tgt = tgt[:, :, :-1]
        enc_state, memory_banks, lengths = self.encoder(src, lengths)
        if bptt is False:
            # src and memory_banks are not used in this initialization function
            self.decoder.init_state(None, None, enc_state)
        # make the input format consistent
        if not isinstance(memory_banks, tuple):
            # ensure the memory banks are batch first
            memory_banks = (memory_banks.transpose(0, 1),)
            lengths = (lengths,)
        dec_out, attns_ex_outs = self.decoder(tgt, memory_banks, memory_lengths=lengths, validation=validation)
        return dec_out, attns_ex_outs


# add by wchen
class SeqHREDModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic sequential hierarchical encoder + flattened decoder model.

    Args:
      encoder (:obj:`EncoderBase`): a sequential hierarchical encoder object
      decoder (:obj:`RNNDecoderBase`): a flattened decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder):
        super(SeqHREDModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, sent_position_tuple, src_word_sent_ids, bptt=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[src_length x batch x features]`.
            tgt (:obj:`LongTensor`):
                a target sequence of size `[tgt_len x batch]` for seq decoder.
                a target sequence of size `[batch x tgt_s_num x tgt_s_len x feat_num]` for hr decoder.
            lengths(:obj:`LongTensor'):
                the src lengths with size `[batch]`.
            sent_position_tuple (:obj: `tuple`): (sent_p, sent_nums) with size `([batch_size, s_num, 2], [batch])`
            src_word_sent_ids (:obj: `tuple'): (word_sent_ids, src_lengths) with size `([batch, src_lengths], [batch])'
            bptt (:obj:`Boolean`):
                a flag indicating if truncated bptt is set. If reset then
                init_state

        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
        """
        # exclude last target from inputs
        if tgt.dim() == 3:
            # for seq decoder
            tgt = tgt[:-1]
        else:
            # for hr decoder
            assert tgt.dim() == 4
            tgt = tgt[:, :, :-1]
        enc_state, memory_banks, lengths = self.encoder(src, lengths, sent_position_tuple)
        if bptt is False:
            # src and memory_banks are not used in this initialization function
            self.decoder.init_state(None, None, enc_state)
        dec_out, attns = self.decoder(tgt, memory_banks, memory_lengths=lengths,
                                      sent_position_tuple=sent_position_tuple,
                                      src_word_sent_ids=src_word_sent_ids)
        return dec_out, attns


# add by wchen
class CatSeqDNMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + CatSeqD decoder encoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder):
        super(CatSeqDNMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, sent_position_tuple=None, src_word_sent_ids=None, bptt=False, validation=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            bptt (:obj:`Boolean`):
                a flag indicating if truncated bptt is set. If reset then
                init_state

        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder(src, lengths)
        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths)
        return dec_out, attns


class TGModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder):
        super(TGModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, query, tgt, lengths, query_lengths, bptt=False, validation=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            query (:obj:`Tensor`):
                a source query sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            query_lengths(:obj:`LongTensor`): the query lengths, pre-padding `[batch]`.
            bptt (:obj:`Boolean`):
                a flag indicating if truncated bptt is set. If reset then
                init_state

        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths = self.encoder((src, query), (lengths, query_lengths))
        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=lengths)
        return dec_out, attns