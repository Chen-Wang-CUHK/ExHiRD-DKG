""" Base Class and function for Decoders """

import torch
import torch.nn as nn
import numpy as np

import onmt.models.stacked_rnn
from onmt.utils.misc import aeq
from onmt.utils.rnn_factory import rnn_factory

from onmt.utils.source_representation_queue import SourceReprentationQueue


class RNNDecoderBase(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`models.NMTModel`.


    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[memory_bank]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general", attn_func="softmax",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False, use_catSeq_dp=False):
        super(RNNDecoderBase, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        self.use_catSeq_dp = use_catSeq_dp

        # Decoder state
        self.state = {}

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type, attn_func=attn_func
        )

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type, attn_func=attn_func
            )
        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn

    def init_state(self, src, memory_bank, encoder_final):
        """ Init decoder state with last state of the encoder """
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
                # add by wchen, only use the last "self.num_layers" encoder layers' final hidden
                enc_layers = hidden.size(0)
                if enc_layers >= self.num_layers:
                    hidden = hidden[enc_layers - self.num_layers:]
                else:
                    # broadcast the hidden of the last encoder layer to initialize every layer of the decoder
                    hidden = [hidden[-1]] * self.num_layers
                    hidden = torch.stack(hidden, dim=0)
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)
            return hidden

        if isinstance(encoder_final, tuple):  # LSTM
            self.state["hidden"] = tuple([_fix_enc_hidden(enc_hid)
                                          for enc_hid in encoder_final])
        else:  # GRU
            self.state["hidden"] = (_fix_enc_hidden(encoder_final), )

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)
        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = \
            self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
        self.state["coverage"] = None

    def map_state(self, fn):
        self.state["hidden"] = tuple(map(lambda x: fn(x, 1),
                                         self.state["hidden"]))
        self.state["input_feed"] = fn(self.state["input_feed"], 1)

    def detach_state(self):
        """ Need to document this """
        self.state["hidden"] = tuple([_.detach()
                                     for _ in self.state["hidden"]])
        self.state["input_feed"] = self.state["input_feed"].detach()

    def forward(self, tgt, memory_bank, memory_lengths=None,
                step=None, test=False):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                 `[tgt_len x batch x nfeats]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            memory_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * dec_outs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # Run the forward pass of the RNN.
        dec_state, dec_outs, attns = self._run_forward_pass(
            tgt, memory_bank, memory_lengths=memory_lengths, test=test)

        # Update the state with the result.
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state
        self.state["input_feed"] = dec_outs[-1].unsqueeze(0)
        self.state["coverage"] = None
        if "coverage" in attns:
            self.state["coverage"] = attns["coverage"][-1].unsqueeze(0)

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)

            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])
        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns


class StdRNNDecoder(RNNDecoderBase):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """

    def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                                 [len x batch x nfeats].
            memory_bank (FloatTensor): output(tensor sequence) from the
                          encoder RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            memory_lengths (LongTensor): the source memory_bank lengths.
        Returns:
            dec_state (Tensor): final hidden state from the decoder.
            dec_outs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """
        assert not self._copy  # TODO, no support yet.
        assert not self._coverage  # TODO, no support yet.

        # Initialize local and return variables.
        attns = {}
        emb = self.embeddings(tgt)

        # Run the forward pass of the RNN.
        if isinstance(self.rnn, nn.GRU):
            rnn_output, dec_state = self.rnn(emb, self.state["hidden"][0])
        else:
            rnn_output, dec_state = self.rnn(emb, self.state["hidden"])

        # Check
        tgt_len, tgt_batch, _ = tgt.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)
        # END

        # Calculate the attention.
        dec_outs, p_attn = self.attn(
            rnn_output.transpose(0, 1).contiguous(),
            memory_bank.transpose(0, 1),
            memory_lengths=memory_lengths
        )
        attns["std"] = p_attn

        # Calculate the context gate.
        if self.context_gate is not None:
            dec_outs = self.context_gate(
                emb.view(-1, emb.size(2)),
                rnn_output.view(-1, rnn_output.size(2)),
                dec_outs.view(-1, dec_outs.size(2))
            )
            dec_outs = \
                dec_outs.view(tgt_len, tgt_batch, self.hidden_size)

        dec_outs = self.dropout(dec_outs)
        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size


class InputFeedRNNDecoder(RNNDecoderBase):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[memory_bank n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None, test=False):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        input_feed = self.state["input_feed"].squeeze(0)
        input_feed_batch, _ = input_feed.size()
        _, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        # Initialize local and return variables.
        dec_outs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        # add for catSeq
        if self.use_catSeq_dp:
            emb = self.dropout(emb)

        dec_state = self.state["hidden"]
        coverage = self.state["coverage"].squeeze(0) \
            if self.state["coverage"] is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for _, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            decoder_input = torch.cat([emb_t, input_feed], 1)
            rnn_output, dec_state = self.rnn(decoder_input, dec_state)
            decoder_output, p_attn = self.attn(
                rnn_output,
                memory_bank.transpose(0, 1),
                memory_lengths=memory_lengths)
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                decoder_output = self.context_gate(
                    decoder_input, rnn_output, decoder_output
                )
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            dec_outs += [decoder_output]
            attns["std"] += [p_attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + p_attn \
                    if coverage is not None else p_attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy and not self._reuse_copy_attn:
                _, copy_attn = self.copy_attn(decoder_output,
                                              memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._copy:
                attns["copy"] = attns["std"]
        # Return result.
        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
            "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.models.stacked_rnn.StackedLSTM
        else:
            stacked_cell = onmt.models.stacked_rnn.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size


# add by wchen
class HREInputFeedRNNDecoder(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`models.NMTModel`.
    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general", attn_func="softmax",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False, hr_attn_type='sent_word_both'):
        super(HREInputFeedRNNDecoder, self).__init__()

        # Basic attributes.
        self.hr_attn_type = hr_attn_type
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        self.attn_type = attn_type

        # Decoder state
        self.state = {}

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        # self.attn = onmt.modules.GlobalAttention(
        #     hidden_size, coverage=coverage_attn,
        #     attn_type=attn_type, attn_func=attn_func
        # )
        self.sent_attn = onmt.modules.MyGlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type, attn_func=attn_func)

        # 'word_only', 'sent_word_both'
        self.word_attn = onmt.modules.WordGlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type, attn_func=attn_func)

        # add one more mlp layer to merge two outputs from sent-level and word-level attention
        # mlp attention wants it with bias
        out_bias = attn_type == "mlp"
        if self.hr_attn_type == 'sent_word_both':
            self.attn_merge_linear_out = nn.Linear(hidden_size * 3, hidden_size, bias=out_bias)
        else:
            # 'sent_only', 'word_only'
            self.attn_merge_linear_out = nn.Linear(hidden_size * 2, hidden_size, bias=out_bias)

        # TODO: implement copy mechanism
        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type, attn_func=attn_func
            )
        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn

    def init_state(self, src, memory_bank, encoder_final):
        """ Init decoder state with last state of the encoder """
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            assert self.bidirectional_encoder
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
                # add by wchen, only use the last "self.num_layers" encoder layers' final hidden
                enc_layers = hidden.size(0)
                if enc_layers >= self.num_layers:
                    hidden = hidden[enc_layers - self.num_layers:]
                else:
                    # broadcast the hidden of the last encoder layer to initialize every layer of the decoder
                    hidden = [hidden[-1]] * self.num_layers
                    hidden = torch.stack(hidden, dim=0)
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)
            return hidden

        if isinstance(encoder_final, tuple):  # LSTM
            self.state["hidden"] = tuple([_fix_enc_hidden(enc_hid)
                                          for enc_hid in encoder_final])
        else:  # GRU
            self.state["hidden"] = (_fix_enc_hidden(encoder_final), )

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)
        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = \
            self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
        self.state["coverage"] = None

    def map_state(self, fn):
        self.state["hidden"] = tuple(map(lambda x: fn(x, 1),
                                         self.state["hidden"]))
        self.state["input_feed"] = fn(self.state["input_feed"], 1)

    def detach_state(self):
        """ Need to document this """
        self.state["hidden"] = tuple([_.detach()
                                     for _ in self.state["hidden"]])
        self.state["input_feed"] = self.state["input_feed"].detach()

    def forward(self, tgt, memory_banks, memory_lengths=None,
                step=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                 `[tgt_len x batch x nfeats]`.
            memory_banks (`tuple`): (sent_memory_bank, word_memory_bank) from the encoder
                 `([batch x s_num x hidden], [batch x s_num x s_len x hidden])`.
            memory_lengths (`tuple`): the padded source lengths
                `([batch], [batch x s_num])`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * dec_outs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * attns: distribution over src words and sents at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # Run the forward pass of the RNN.
        dec_state, dec_outs, attns = self._run_forward_pass(
            tgt, memory_banks, memory_lengths=memory_lengths)

        # Update the state with the result.
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state
        self.state["input_feed"] = dec_outs[-1].unsqueeze(0)
        self.state["coverage"] = None
        if "coverage" in attns:
            self.state["coverage"] = attns["coverage"][-1].unsqueeze(0)

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)

            tgt_len, batch, _ = dec_outs.size()
            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])
                    # changed by wchen
                    if k == "copy":
                        attns[k] = attns[k].view(tgt_len, batch, -1)
        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns

    def _run_forward_pass(self, tgt, memory_banks, memory_lengths=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                 `[tgt_len x batch x nfeats]`.
            memory_banks (`tuple`): (sent_memory_bank, word_memory_bank) from the encoder
                 `([batch x s_num x hidden], [batch x s_num x s_len x hidden])`.
            memory_lengths (`tuple`): the padded source lengths
                `([batch], [batch x s_num])`.
        Returns:
            dec_state (Tensor): final hidden state from the decoder.
            dec_outs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """
        # Additional args check.
        input_feed = self.state["input_feed"].squeeze(0)
        input_feed_batch, _ = input_feed.size()
        _, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        assert isinstance(memory_banks, tuple)
        assert isinstance(memory_lengths, tuple)
        sent_memory_bank, word_memory_bank = memory_banks
        sent_lengths, word_lengths = memory_lengths

        # Initialize local and return variables.
        dec_outs = []
        attns = {"sent_std": [], "word_std": []}
        # attns = {"sent_std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        dec_state = self.state["hidden"]
        coverage = self.state["coverage"].squeeze(0) \
            if self.state["coverage"] is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for _, emb_t in enumerate(emb.split(1)):
            # [batch, emb_dim]
            emb_t = emb_t.squeeze(0)
            # [batch, emb_dim + h_size]
            decoder_input = torch.cat([emb_t, input_feed], 1)
            # rnn_output: [batch, h_size]
            # dec_state: LSTM (h, c) or GRU (h,)
            rnn_output, dec_state = self.rnn(decoder_input, dec_state)

            # sentence level attention
            # [batch, h_size], [batch, s_num]
            sent_attn_c, sent_p_attn = self.sent_attn(
                rnn_output,
                sent_memory_bank,
                memory_lengths=sent_lengths)

            # word level attention
            # [batch, h_size], [batch, s_num, s_len]
            word_attn_c, word_p_attn = self.word_attn(
                rnn_output,
                word_memory_bank,
                memory_lengths=word_lengths,
                sent_align_vectors=sent_p_attn,
                sent_nums=sent_lengths)

            # if self.context_gate is not None:
            #     # TODO: context gate should be employed
            #     # instead of second RNN transform.
            #     decoder_output = self.context_gate(
            #         decoder_input, rnn_output, decoder_output
            #     )

            if self.hr_attn_type == 'sent_word_both':
                # [batch, 3 * hidden_size]
                decoder_output = torch.cat([rnn_output, sent_attn_c, word_attn_c], dim=1)
            elif self.hr_attn_type == 'sent_only':
                # [batch, 2 * hidden_size]
                decoder_output = torch.cat([rnn_output, sent_attn_c], dim=1)
            else:
                # 'word_only'
                # [batch, 2 * hidden_size]
                decoder_output = torch.cat([rnn_output, word_attn_c], dim=1)

            # [batch, hidden_size]
            decoder_output = self.attn_merge_linear_out(decoder_output)
            if self.attn_type in ["general", "dot"]:
                decoder_output = torch.tanh(decoder_output)

            # decoder_output = sent_dec_output
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            dec_outs += [decoder_output]
            attns["sent_std"] += [sent_p_attn]
            attns["word_std"] += [word_p_attn]

            # TODO: coverage
            # # Update the coverage attention.
            # if self._coverage:
            #     coverage = coverage + p_attn \
            #         if coverage is not None else p_attn
            #     attns["coverage"] += [coverage]

            # TODO: copy mechanism of GU.
            # # Run the forward pass of the copy attention layer.
            # if self._copy and not self._reuse_copy_attn:
            #     _, copy_attn = self.copy_attn(decoder_output,
            #                                   memory_bank.transpose(0, 1))
            #     attns["copy"] += [copy_attn]
            # elif self._copy:
            if self._copy:
                assert self._reuse_copy_attn, "Only reuse copy attn is supported!"
                attns["copy"] = attns["word_std"]
        # Return result.
        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
            "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.models.stacked_rnn.StackedLSTM
        else:
            stacked_cell = onmt.models.stacked_rnn.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size

# add by wchen
class SeqHREInputFeedRNNDecoder(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`models.HREDNMTModel`.
    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general", attn_func="softmax",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False, hr_attn_type='sent_word_both', seqHRE_attn_rescale=False):
        super(SeqHREInputFeedRNNDecoder, self).__init__()

        # Basic attributes.
        self.hr_attn_type = hr_attn_type
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        self.attn_type = attn_type

        # Decoder state
        self.state = {}

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        # self.attn = onmt.modules.GlobalAttention(
        #     hidden_size, coverage=coverage_attn,
        #     attn_type=attn_type, attn_func=attn_func
        # )
        self.sent_attn = onmt.modules.MyGlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type, attn_func=attn_func)

        # 'word_only', 'sent_word_both'
        self.word_attn = onmt.modules.SeqHREWordGlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type, attn_func=attn_func, seqHRE_attn_rescale=seqHRE_attn_rescale)

        # add one more mlp layer to merge two outputs from sent-level and word-level attention
        # mlp attention wants it with bias
        out_bias = attn_type == "mlp"
        if self.hr_attn_type == 'sent_word_both':
            self.attn_merge_linear_out = nn.Linear(hidden_size * 3, hidden_size, bias=out_bias)
        else:
            # 'sent_only', 'word_only'
            self.attn_merge_linear_out = nn.Linear(hidden_size * 2, hidden_size, bias=out_bias)

        # TODO: implement copy mechanism
        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type, attn_func=attn_func
            )
        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn

    def init_state(self, src, memory_bank, encoder_final):
        """ Init decoder state with last state of the encoder """
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            assert self.bidirectional_encoder
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
                # add by wchen, only use the last "self.num_layers" encoder layers' final hidden
                enc_layers = hidden.size(0)
                if enc_layers >= self.num_layers:
                    hidden = hidden[enc_layers - self.num_layers:]
                else:
                    # broadcast the hidden of the last encoder layer to initialize every layer of the decoder
                    hidden = [hidden[-1]] * self.num_layers
                    hidden = torch.stack(hidden, dim=0)
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)
            return hidden

        if isinstance(encoder_final, tuple):  # LSTM
            self.state["hidden"] = tuple([_fix_enc_hidden(enc_hid)
                                          for enc_hid in encoder_final])
        else:  # GRU
            self.state["hidden"] = (_fix_enc_hidden(encoder_final), )

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)
        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = \
            self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
        self.state["coverage"] = None

    def map_state(self, fn):
        self.state["hidden"] = tuple(map(lambda x: fn(x, 1),
                                         self.state["hidden"]))
        self.state["input_feed"] = fn(self.state["input_feed"], 1)

    def detach_state(self):
        """ Need to document this """
        self.state["hidden"] = tuple([_.detach()
                                     for _ in self.state["hidden"]])
        self.state["input_feed"] = self.state["input_feed"].detach()

    def forward(self, tgt, memory_banks, memory_lengths=None,
                step=None, sent_position_tuple=None, src_word_sent_ids=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                 `[tgt_len x batch x nfeats]`.
            memory_banks (`tuple`): (sent_memory_bank, word_memory_bank) from the encoder
                 `([sent_num x batch x hidden], [src_len x batch x hidden])`.
            memory_lengths (`LongTensor`): the source lengths
                `[batch]`.
            sent_position_tuple (`tuple`): the sentence position infomation. (sent_position_info, sent_nums)
                `([sent_num, batch, 2], [batch])`
            src_word_sent_ids (:obj: `tuple'): (word_sent_ids, src_lengths) with size `([batch, src_lengths], [batch])'
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * dec_outs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * attns: distribution over src words and sents at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # Run the forward pass of the RNN.
        dec_state, dec_outs, attns = self._run_forward_pass(
            tgt, memory_banks, memory_lengths=memory_lengths, sent_position_tuple=sent_position_tuple, src_word_sent_ids=src_word_sent_ids)

        # Update the state with the result.
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state
        self.state["input_feed"] = dec_outs[-1].unsqueeze(0)
        self.state["coverage"] = None
        if "coverage" in attns:
            self.state["coverage"] = attns["coverage"][-1].unsqueeze(0)

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)

            tgt_len, batch, _ = dec_outs.size()
            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])
                    # changed by wchen
                    if k == "copy":
                        attns[k] = attns[k].view(tgt_len, batch, -1)
        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns

    def _run_forward_pass(self, tgt, memory_banks, memory_lengths=None, sent_position_tuple=None, src_word_sent_ids=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                 `[tgt_len x batch x nfeats]`.
            memory_banks (`tuple`): (sent_memory_bank, word_memory_bank) from the encoder
                 `([sent_num x batch x hidden], [src_len x batch x hidden])`.
            memory_lengths (`LongTensor`): the source lengths
                `[batch]`.
            sent_position_tuple (`tuple`): the sentence position infomation. (sent_position_info, sent_nums)
                `([sent_num, batch, 2], [batch])`
            src_word_sent_ids (:obj: `tuple'): (word_sent_ids, src_lengths) with size `([batch, src_lengths], [batch])'
        Returns:
            dec_state (Tensor): final hidden state from the decoder.
            dec_outs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """
        # Additional args check.
        input_feed = self.state["input_feed"].squeeze(0)
        input_feed_batch, _ = input_feed.size()
        _, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        assert isinstance(memory_banks, tuple)
        sent_memory_bank, word_memory_bank = memory_banks
        word_lengths = memory_lengths
        sent_position, sent_lengths = sent_position_tuple

        # Initialize local and return variables.
        dec_outs = []
        attns = {"sent_std": [], "word_std": []}
        # attns = {"sent_std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        dec_state = self.state["hidden"]
        coverage = self.state["coverage"].squeeze(0) \
            if self.state["coverage"] is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for _, emb_t in enumerate(emb.split(1)):
            # [batch, emb_dim]
            emb_t = emb_t.squeeze(0)
            # [batch, emb_dim + h_size]
            decoder_input = torch.cat([emb_t, input_feed], 1)
            # rnn_output: [batch, h_size]
            # dec_state: LSTM (h, c) or GRU (h,)
            rnn_output, dec_state = self.rnn(decoder_input, dec_state)

            # sentence level attention
            # [batch, h_size], [batch, sent_num]
            sent_attn_c, sent_p_attn = self.sent_attn(
                rnn_output,
                sent_memory_bank.transpose(0, 1),
                memory_lengths=sent_lengths)

            # word level attention
            # [batch, h_size], [batch, src_len]
            word_attn_c, word_p_attn = self.word_attn(
                rnn_output,
                word_memory_bank.transpose(0, 1),
                memory_lengths=word_lengths,
                sent_align_vectors=sent_p_attn,
                sent_position_tuple=sent_position_tuple,
                src_word_sent_ids=src_word_sent_ids)

            # if self.context_gate is not None:
            #     # TODO: context gate should be employed
            #     # instead of second RNN transform.
            #     decoder_output = self.context_gate(
            #         decoder_input, rnn_output, decoder_output
            #     )

            if self.hr_attn_type == 'sent_word_both':
                # [batch, 3 * hidden_size]
                decoder_output = torch.cat([rnn_output, sent_attn_c, word_attn_c], dim=1)
            elif self.hr_attn_type == 'sent_only':
                # [batch, 2 * hidden_size]
                decoder_output = torch.cat([rnn_output, sent_attn_c], dim=1)
            else:
                # 'word_only'
                # [batch, 2 * hidden_size]
                decoder_output = torch.cat([rnn_output, word_attn_c], dim=1)

            # [batch, hidden_size]
            decoder_output = self.attn_merge_linear_out(decoder_output)
            if self.attn_type in ["general", "dot"]:
                decoder_output = torch.tanh(decoder_output)

            # decoder_output = sent_dec_output
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            dec_outs += [decoder_output]
            attns["sent_std"] += [sent_p_attn]
            attns["word_std"] += [word_p_attn]

            # TODO: coverage
            # # Update the coverage attention.
            # if self._coverage:
            #     coverage = coverage + p_attn \
            #         if coverage is not None else p_attn
            #     attns["coverage"] += [coverage]

            # TODO: copy mechanism of GU.
            # # Run the forward pass of the copy attention layer.
            # if self._copy and not self._reuse_copy_attn:
            #     _, copy_attn = self.copy_attn(decoder_output,
            #                                   memory_bank.transpose(0, 1))
            #     attns["copy"] += [copy_attn]
            # elif self._copy:
            if self._copy:
                assert self._reuse_copy_attn, "Only reuse copy attn is supported!"
                attns["copy"] = attns["word_std"]
        # Return result.
        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
            "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.models.stacked_rnn.StackedLSTM
        else:
            stacked_cell = onmt.models.stacked_rnn.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size

# add by wchen
class HRDInputFeedRNNDecoder(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`models.HREDModel` or .
    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general", attn_func="softmax",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False, hr_attn_type='sent_word_both',
                 word_dec_init_type='attn_vec', remove_input_feed_w=False,
                 input_feed_w_type='attn_vec',
                 hr_enc=True, seqhr_enc=False, seqE_HRD_rescale_attn=False,
                 seqHRE_attn_rescale=False, use_zero_s_emb=False, not_detach_coverage=False,
                 eok_idx=None, eos_idx=None, pad_idx=None, sep_idx=None, p_end_idx=None, a_end_idx=None,
                 position_enc=None, position_enc_word_init=False,
                 position_enc_sent_feed_w=False, position_enc_first_word_feed=False,
                 position_enc_embsize=None, position_enc_start_token=False, position_enc_sent_state=False,
                 position_enc_all_first_valid_word_dec_inputs=False,
                 sent_dec_init_type='enc_first', remove_input_feed_h=False, detach_input_feed_w=False,
                 use_target_encoder=False, src_states_capacity=128, src_states_sample_size=32):
        super(HRDInputFeedRNNDecoder, self).__init__()

        assert rnn_type == "GRU", "Currently, only GRU is supported!"
        # Basic attributes.
        self.hr_attn_type = hr_attn_type
        self.word_dec_init_type = word_dec_init_type
        self.remove_input_feed_w = remove_input_feed_w
        self.detach_input_feed_w = detach_input_feed_w
        self.remove_input_feed_h = remove_input_feed_h
        self.input_feed_w_type = input_feed_w_type
        self.hr_enc = hr_enc
        self.seqhr_enc = seqhr_enc
        self.seqE_HRD_rescale_attn = seqE_HRD_rescale_attn
        self.use_zero_s_emb = use_zero_s_emb
        self.not_detach_coverage = not_detach_coverage
        self.eok_idx = eok_idx
        self.sep_idx = sep_idx
        self.p_end_idx = p_end_idx
        self.a_end_idx = a_end_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings

        self.sent_dec_init_type = sent_dec_init_type

        self.position_enc = position_enc
        self.position_enc_word_init = position_enc_word_init
        self.position_enc_sent_feed_w = position_enc_sent_feed_w
        self.position_enc_first_word_feed = position_enc_first_word_feed
        self.position_enc_sent_state = position_enc_sent_state
        self.position_enc_all_first_valid_word_dec_inputs = position_enc_all_first_valid_word_dec_inputs
        if self.position_enc is not None:
            assert position_enc_word_init or position_enc_sent_feed_w or \
                   position_enc_first_word_feed or position_enc_sent_state or \
                   position_enc_all_first_valid_word_dec_inputs
        else:
            assert (not position_enc_word_init) and (not position_enc_sent_feed_w) and \
                   (not position_enc_first_word_feed) and (not position_enc_sent_state) and \
                   (not position_enc_all_first_valid_word_dec_inputs)

        self.position_enc_embsize = position_enc_embsize
        self.position_enc_start_token = position_enc_start_token
        if position_enc_embsize is not None:
            assert position_enc_start_token
        else:
            assert not position_enc_start_token

        self.use_target_encoder = use_target_encoder
        self.src_states_capacity = src_states_capacity
        self.src_states_sample_size = src_states_sample_size

        self.dropout = nn.Dropout(dropout)
        self.attn_type = attn_type

        # Decoder state
        self.sent_state = {}
        self.word_state = {}

        # Build the sentence decoder RNN.
        self.sent_dec_rnn = self._build_rnn(rnn_type,
                                            input_size=hidden_size if self.remove_input_feed_w or self.remove_input_feed_h else 2 * hidden_size,
                                            hidden_size=hidden_size,
                                            num_layers=num_layers,
                                            dropout=dropout)

        # Build the word decoder RNN
        self.word_dec_rnn = self._build_rnn(rnn_type,
                                            input_size=self._input_size,
                                            hidden_size=hidden_size,
                                            num_layers=num_layers,
                                            dropout=dropout)
        # # Set up the context gate.
        # self.context_gate = None
        # if context_gate is not None:
        #     self.context_gate = onmt.modules.context_gate_factory(
        #         context_gate, self._input_size,
        #         hidden_size, hidden_size, hidden_size
        #     )

        # Set up the standard attention.
        self._coverage = coverage_attn
        # self.attn = onmt.modules.GlobalAttention(
        #     hidden_size, coverage=coverage_attn,
        #     attn_type=attn_type, attn_func=attn_func
        # )

        self.sent_attn = None
        if not self.remove_input_feed_h or self.seqE_HRD_rescale_attn or self._coverage or self.word_dec_init_type=='attn_vec':
            self.sent_attn = onmt.modules.MyGlobalAttention(
                hidden_size, coverage=coverage_attn,
                attn_type=attn_type, attn_func=attn_func, output_attn_h=True)

        # Set up the target enc attention
        self.src_states_queue = None
        self.target_attn = None
        if use_target_encoder:
            self.src_states_queue = SourceReprentationQueue(src_states_capacity)
            self.target_attn = onmt.modules.TargetEncGlobalAttention(tgt_enc_dim=hidden_size,
                                                                     src_enc_dim=hidden_size,
                                                                     attn_type=attn_type,
                                                                     attn_func=attn_func)
        if self.hr_enc:
            # 'word_only', 'sent_word_both'
            if self.seqhr_enc:
                self.word_attn = onmt.modules.SeqHREWordGlobalAttention(
                    hidden_size, coverage=False,
                    attn_type=attn_type, attn_func=attn_func, output_attn_h=True,
                    seqHRE_attn_rescale=seqHRE_attn_rescale)
            else:
                self.word_attn = onmt.modules.WordGlobalAttention(
                    hidden_size, coverage=False,
                    attn_type=attn_type, attn_func=attn_func, output_attn_h=True)
        elif self.seqE_HRD_rescale_attn:
            self.word_attn = onmt.modules.W2WordGlobalAttention(
                hidden_size, coverage=False,
                attn_type=attn_type, attn_func=attn_func, output_attn_h=True)
        else:
            self.word_attn = onmt.modules.MyGlobalAttention(
                hidden_size, coverage=False,
                attn_type=attn_type, attn_func=attn_func, output_attn_h=True)

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn:
            self._copy = True
            assert reuse_copy_attn, "Only reuse_copy_attn is supported."
        self._reuse_copy_attn = reuse_copy_attn

    def init_state(self, src, memory_bank, encoder_final):
        """ Init decoder state with last state of the encoder """
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)

            # last encoder state initialization
            if self.num_layers == 1 and self.sent_dec_init_type == "enc_last":
                hidden = hidden[-1].unsqueeze(0)

            # use the mean of the encoder layer's final state as the
            if self.sent_dec_init_type == "enc_mean":
                hidden = hidden.mean(dim=0, keepdim=True)

            return hidden

        batch_size = encoder_final.size(1)
        h_size = (batch_size, self.hidden_size)

        if isinstance(encoder_final, tuple):  # LSTM
            self.sent_state["hidden"] = [tuple([_fix_enc_hidden(enc_hid) for enc_hid in encoder_final])]
        else:  # GRU
            # currently it is using the final state of the first encoder layer
            init_hidden = _fix_enc_hidden(encoder_final)
            if self.sent_dec_init_type == 'zero':
                init_hidden.data.zero_()
            self.sent_state["hidden"] = [(init_hidden, )]

        # Init the sentence input feed.
        self.sent_state["input_feed_h"] = \
            [self.sent_state["hidden"][0][0].data.new(*h_size).zero_()]
        self.sent_state["input_feed_w"] = \
            [self.sent_state["hidden"][0][0].data.new(*h_size).zero_()]
        # for orthogonal loss
        self.sent_state["word_init_states"] = []
        # for sent_level coverage mechanism
        self.sent_state["sent_std_attn"] = []
        self.sent_state["sent_coverage"] = []
        # for orthogonal loss
        self.sent_state["orthog_states"] = []

        # for target enc loss
        # [b_size, self.hidden_size]
        self.src_encoder_final = self.sent_state["hidden"][0][0].squeeze(0)
        # init the word input feed
        self.init_word_state()

    def init_word_state(self, hidden=None):
        # Init the word input feed
        batch_size = self.sent_state["hidden"][0][0].size(1)
        h_size = (batch_size, self.hidden_size)

        self.word_state["input_feed"] = \
            self.sent_state["hidden"][0][0].data.new(*h_size).zero_()
        if hidden is not None:
            # (h,) or (h, c)
            # self.word_state["hidden"] = tuple([h.detach() for h in hidden])
            self.word_state["hidden"] = hidden
        else:
            # for GRU
            self.word_state["hidden"] = (self.sent_state["hidden"][0][0].data.new(*h_size).zero_().unsqueeze(0),)
            # TODO: for LSTM

        self.word_state["input_feed_list"] = [self.word_state["input_feed"]]
        self.word_state["hidden_list"] = [self.word_state["hidden"]]

        self.word_state["one_kp_not_finished"] = set(range(batch_size))
        self.word_state["finished_word_state"] = {}

    # def map_state(self, fn):
    #     self.state["hidden"] = tuple(map(lambda x: fn(x, 1),
    #                                      self.state["hidden"]))
    #     self.state["input_feed"] = fn(self.state["input_feed"], 1)
    #
    # def detach_state(self):
    #     """ Need to document this """
    #     self.state["hidden"] = tuple([_.detach()
    #                                  for _ in self.state["hidden"]])
    #     self.state["input_feed"] = self.state["input_feed"].detach()

    def forward(self, tgt, memory_banks, memory_lengths=None, sent_position_tuple=None, src_word_sent_ids=None, last_step=False, testing=False, validation=False):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                 `[batch x tgt_s_num x tgt_s_len x nfeats]`.
            memory_banks (`tuple`): the memory banks from encoder
                `(word_memory_bank,)` from a seq encoder with size `([batch x src_len x hidden],)`
                `(sent_memory_bank, word_memory_bank)` from a hr encoder with size
                `([batch x s_num x hidden], [batch x s_num x s_len x hidden])` for hr_enc.
                `([batch x s_num x hidden], [src_len x batch x hidden])` for seqhr_enc.
            memory_lengths (`tuple`): the source lengths
                `(src_lengths,)` for a sequence encoded memory bank with size `([batch],)`
                `(sent_nums, sent_lens)` for hr encoded memory banks with size `([batch], [batch x s_num])`.
            sent_position_tuple (:obj: `tuple`): Only used for seqhr_enc (sent_p, sent_nums) with size
                `([sent_num x batch x 2], [batch])`.
            src_word_sent_ids (:obj: `tuple'): (word_sent_ids, src_lengths) with size `([batch, src_lengths], [batch])'
            last_step ('bool'): Only used for translation. Whether current step is the last step of word level decoding.
            testing ('bool'): Indicate whether it is testing mode
            validation ('bool'): Indicate whether it is validation mode
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * dec_outs: output from the decoder (after attn)
                         `[batch x tgt_s_num x tgt_s_len x hidden]`.
                * attns: distribution over src words and sents at each tgt
                        `[batch x tgt_s_num x tgt_s_len x src_len]`.
        """
        # Run the forward pass of the RNN.
        # [tgt_s_num x tgt_s_len x batch x nfeats]
        tgt = tgt.transpose(0, 1).transpose(1, 2)
        if last_step:
            assert tgt.size(0) == 1 and tgt.size(1) == 1, "'last_step' is only used for translation."

        # sample src final states and update the src_states_queue, code is from ken's github
        _, _, b_size, _ = tgt.size()
        if self.use_target_encoder and not testing:
            if len(self.src_states_queue) < self.src_states_sample_size:
                src_states_samples_2dlist = None
                src_states_target_list = None
            else:
                src_states_samples_2dlist = []
                src_states_target_list = []
                for _ in range(b_size):
                    # N encoder states from the queue
                    src_states_samples_list, place_holder_idx = self.src_states_queue.sample(
                        self.src_states_sample_size)
                    # insert the smaple list of one batch to the 2d list
                    src_states_samples_2dlist.append(src_states_samples_list)
                    # store the idx of place-holder for the batch
                    src_states_target_list.append(place_holder_idx)
        else:
            src_states_samples_2dlist = None
            src_states_target_list = None

        dec_outs, attns, other_outs = self._run_forward_pass_one_step(tgt,
                                                                      memory_banks,
                                                                      memory_lengths=memory_lengths,
                                                                      sent_position_tuple=sent_position_tuple,
                                                                      src_word_sent_ids=src_word_sent_ids,
                                                                      src_states_samples_2dlist=src_states_samples_2dlist,
                                                                      src_states_target_list=src_states_target_list,
                                                                      last_step=last_step,
                                                                      testing=testing)

        if self.use_target_encoder and not testing and not validation:
            # put all the encoder final states to the queue, Need to call detach() first
            # self.src_encoder_final: [b_size, h_size]
            [self.src_states_queue.put(self.src_encoder_final[i, :].detach()) for i in range(b_size)]

        # # Update the state with the result.
        # if not isinstance(dec_state, tuple):
        #     dec_state = (dec_state,)
        # self.state["hidden"] = dec_state
        # self.state["input_feed"] = dec_outs[-1].unsqueeze(0)
        # self.state["coverage"] = None
        # if "coverage" in attns:
        #     self.state["coverage"] = attns["coverage"][-1].unsqueeze(0)

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(dec_outs) == list:
            # [tgt_s_num, tgt_s_len, batch, h_size]
            dec_outs = torch.stack(dec_outs)
            tgt_s_num, tgt_s_len, batch, h_size = dec_outs.size()
            # [tgt_s_num * tgt_s_len, batch, h_size]
            dec_outs = dec_outs.view(-1, batch, h_size)

            for k in attns:
                if type(attns[k]) == list:
                    # word_std: [tgt_s_num, tgt_s_len, batch, s_num, s_len]
                    # copy:     [tgt_s_num, tgt_s_len, batch, s_num, s_len]
                    attns[k] = torch.stack(attns[k])
                    if k == "copy" or k == "word_std":
                        # [tgt_s_num * tgt_s_len, batch, s_num * s_len]
                        attns[k] = attns[k].view(tgt_s_num * tgt_s_len, batch, -1)

            for k in other_outs:
                assert k not in attns
                if type(other_outs[k]) == list:
                    # "orthog_states": [batch, s_num, h_size]
                    # sent_std_attn: [batch, tgt_s_num, sent_memory_bank_len]
                    # sent_coverage: [batch, tgt_s_num, sent_memory_bank_len]
                    other_outs[k] = torch.stack(other_outs[k], dim=1)

        attns.update(other_outs)
        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns

    def _run_forward_pass(self, tgt, memory_banks, memory_lengths=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                 `[batch x tgt_s_num x tgt_s_len x nfeats]`.
            memory_banks (`tuple`): (sent_memory_bank, word_memory_bank) from the encoder
                 `([batch x s_num x hidden], [batch x s_num x s_len x hidden])`.
            memory_lengths (`tuple`): the padded source lengths
                `([batch], [batch x s_num])`.
        Returns:
            dec_state (Tensor): final hidden state from the decoder.
            dec_outs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """
        # Additional args check.
        # sent
        sent_input_feed_h = self.sent_state["input_feed_h"].squeeze(0)
        input_feed_h_batch, _ = sent_input_feed_h.size()
        sent_input_feed_w = self.sent_state["input_feed_w"].squeeze(0)
        input_feed_w_batch, _ = sent_input_feed_w.size()
        # word
        word_input_feed = self.word_state["input_feed"].squeeze(0)
        word_input_feed_batch, _ = word_input_feed.size()
        # [tgt_s_num x tgt_s_len x batch x nfeats]
        tgt = tgt.transpose(0, 1).transpose(1, 2)
        _, _, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_h_batch, input_feed_w_batch)
        # END Additional args check.

        if isinstance(memory_banks, tuple):
            sent_memory_bank, word_memory_bank = memory_banks
            sent_lengths, word_lengths = memory_lengths
        else:
            word_memory_bank = memory_banks
            word_lengths = memory_lengths

            sent_memory_bank = memory_banks
            sent_lengths = memory_lengths

        # Initialize local and return variables.
        dec_outs = []
        attns = {"sent_std": [], "word_std": []}
        # attns = {"sent_std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 4  # tgt_s_num x tgt_s_len  x batch x embedding_dim

        sent_dec_state = self.sent_state["hidden"]
        # TODO: sentence coverage mechanism
        # coverage = self.state["coverage"].squeeze(0) \
        #     if self.state["coverage"] is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for _, sent_emb_i in enumerate(emb.split(1, dim=0)):
            dec_outs += [[]]
            # [tgt_s_len, batch, emb_dim]
            sent_emb_i = sent_emb_i.squeeze(0)
            # [batch, 2 * h_size]
            sent_decoder_input = torch.cat([sent_input_feed_h, sent_input_feed_w], dim=1)
            # rnn_output: [batch, h_size]
            # dec_state: LSTM (h, c) or GRU (h,)
            sent_rnn_output, sent_dec_state = self.sent_dec_rnn(sent_decoder_input, sent_dec_state)
            # [batch, h_size], [batch, s_num]
            sent_attn_h, sent_p_attn = self.sent_attn(sent_rnn_output, sent_memory_bank, memory_lengths=sent_lengths)

            # sent_attn_h = self.dropout(sent_attn_h)

            attns["sent_std"] += [sent_p_attn]
            attns["word_std"] += [[]]

            # word level decoding
            # TODO: try sent h* or sent h
            word_dec_state = (sent_attn_h.unsuqeeze(0), )
            for _, word_emb_j in enumerate(sent_emb_i.split(1, dim=0)):
                # [batch, emb_dim]
                word_emb_j = word_emb_j.squeeze(0)
                # [batch, 2 * h_size]
                word_dec_input = torch.cat([word_emb_j, word_input_feed], 1)
                # rnn_output: [batch, h_size]
                # dec_state: LSTM (h, c) or GRU (h,)
                word_rnn_output, word_dec_state = self.word_dec_rnn(word_dec_input, word_dec_state)
                # word level attention
                # [batch, h_size], [batch, s_num, s_len]
                if self.hr_enc:
                    word_attn_h, word_p_attn = self.word_attn(
                        word_rnn_output,
                        word_memory_bank,
                        memory_lengths=word_lengths,
                        sent_align_vectors=sent_p_attn,
                        sent_nums=sent_lengths)
                elif self.seqE_HRD_rescale_attn:
                    word_attn_h, word_p_attn = self.word_attn(
                        word_rnn_output,
                        word_memory_bank,
                        memory_lengths=word_lengths,
                        sent_align_vectors=sent_p_attn)
                else:
                    word_attn_h, word_p_attn = self.word_attn(
                        word_rnn_output,
                        word_memory_bank,
                        memory_lengths=word_lengths)

                # [batch, h_size]
                decoder_output = self.dropout(word_attn_h)
                word_input_feed = decoder_output

                dec_outs[-1] += [decoder_output]
                attns["word_std"][-1] += [word_p_attn]

            # [tgt_s_len, batch, h_size]
            dec_outs[-1] = torch.stack(dec_outs[-1])
            # [tgt_s_len, batch, s_num, s_len]
            attns["word_std"][-1] = torch.stack(attns["word_std"][-1])

            sent_input_feed_h = sent_attn_h
            # TODO: collect the final hidden state of one keyphrase h* or h
            sent_input_feed_w = decoder_output

        if self._copy:
            assert self._reuse_copy_attn, "Only reuse copy attn is supported!"
            attns["copy"] = attns["word_std"]
        # Return result.
        return word_dec_state, dec_outs, attns

    def _run_forward_pass_one_step(self, tgt, memory_banks, memory_lengths=None,
                                   sent_position_tuple=None, src_word_sent_ids=None,
                                   src_states_samples_2dlist=None,
                                   src_states_target_list=None,
                                   last_step=False, testing=False):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                 `[tgt_s_num x batch x tgt_s_len x nfeats]`.
            memory_banks (`tuple`): the memory banks from encoder
                `(word_memory_bank,)` from a seq encoder with size `([batch x src_len x hidden],)`
                `(sent_memory_bank, word_memory_bank)` from a hr encoder with size
                `([batch x s_num x hidden], [batch x s_num x s_len x hidden])` for hr_enc.
                `([s_num x batch x hidden], [src_len x batch x hidden])` for seqhr_enc.
            memory_lengths (`tuple`): the source lengths
                `(src_lengths,)` for a sequence encoded memory bank with size `([batch],)`
                `(sent_nums, sent_lens)` for hr encoded memory banks with size `([batch], [batch x s_num])`.
            sent_position_tuple (:obj: `tuple`): Only used for seqhr_enc (sent_p, sent_nums) with size
                `([batch_size, s_num, 2], [batch])`.
            src_word_sent_ids (:obj: `tuple'): (word_sent_ids, src_lengths) with size `([batch, src_lengths], [batch])'
            last_step ('bool'): Only used for translation. Whether current step is the last step of word level decoding.
        Returns:
            dec_state (Tensor): final hidden state from the decoder.
            dec_outs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """
        # [1 x tgt_s_len x batch x nfeats]
        if last_step:
            assert tgt.size(0) == 1 and tgt.size(1) == 1, "'last_step' is only used for translation."

        tgt = tgt.contiguous()
        tgt_s_num, tgt_s_len, tgt_batch, nfeats = tgt.size()

        if len(memory_banks) == 2:
            sent_memory_bank, word_memory_bank = memory_banks
            if not self.seqhr_enc:
                sent_lengths, word_lengths = memory_lengths
            else:
                # [batch, s_num, 2*h_size]
                sent_memory_bank = sent_memory_bank.transpose(0, 1)
                # [batch, src_len, 2*h_size]
                word_memory_bank = word_memory_bank.transpose(0, 1)
                word_lengths = memory_lengths
                _, sent_lengths = sent_position_tuple
        else:
            word_memory_bank = memory_banks[0]
            word_lengths = memory_lengths[0]

            sent_memory_bank = memory_banks[0]
            sent_lengths = memory_lengths[0]
        # check whether the batch sizes are consistent
        aeq(tgt_batch, word_memory_bank.size(0), sent_memory_bank.size(0))
        # Initialize local and return variables.
        dec_outs = []
        attns = {"word_std": []}
        other_outs = {}
        if self._copy:
            attns["copy"] = []

        # [b_size, max_src_len]
        sent_coverage = self.sent_state["sent_coverage"][-1] \
            if len(self.sent_state["sent_coverage"]) != 0 else None

        # for collecting sep state
        target_sep_states_2dlist = [[] for _ in range(tgt_batch)]

        # word level decoding
        emb = self.embeddings(tgt.view(-1, tgt_batch, nfeats))
        emb = emb.view(tgt_s_num, tgt_s_len, tgt_batch, -1)
        assert emb.dim() == 4  # tgt_s_num x tgt_s_len x batch x embedding_dim
        # TODO: word level coverage mechanism
        # coverage = self.state["coverage"].squeeze(0) \
        #     if self.state["coverage"] is not None else None
        for i, sent_emb_i in enumerate(emb.split(1, dim=0)):
            # Since the word level decoding is based on sentence level decoding,
            # We first check whether to update sentence decoding state.
            # TODO: sentence coverage mechanism
            if len(self.sent_state["sent_std_attn"]) == 0 or len(self.word_state["finished_word_state"]) == tgt_batch:
                # The initial sentence decoding step or the middle sentence decoding state
                if len(self.word_state["finished_word_state"]) == tgt_batch:
                    assert len(self.word_state["one_kp_not_finished"]) == 0
                    finished_word_state_list = \
                        [self.word_state["finished_word_state"][tmp_idx] for tmp_idx in range(tgt_batch)]
                    # send_dec input 1
                    # [batch, h_size]
                    saved_input_feed_w = torch.cat(finished_word_state_list, dim=0)
                    if self.detach_input_feed_w:
                        saved_input_feed_w = saved_input_feed_w.detach()
                    self.sent_state["input_feed_w"] += [saved_input_feed_w]
                # check whether the step of the sent decoding hidden and inputs are consistent
                aeq(len(self.sent_state["hidden"]),
                    len(self.sent_state["input_feed_w"]),
                    len(self.sent_state["input_feed_h"]))
                # sent
                sent_dec_state = self.sent_state["hidden"][-1]
                _, hidden_batch, _ = sent_dec_state[0].size()
                sent_input_feed_w = self.sent_state["input_feed_w"][-1]
                input_feed_w_batch, _ = sent_input_feed_w.size()
                aeq(tgt_batch, hidden_batch, input_feed_w_batch)

                sent_input_feed_h = None
                if self.sent_attn is not None:
                    sent_input_feed_h = self.sent_state["input_feed_h"][-1]
                    input_feed_h_batch, _ = sent_input_feed_h.size()
                    aeq(input_feed_h_batch, input_feed_w_batch)

                sent_step_i = len(self.sent_state["input_feed_w"]) - 1
                sent_step_i = torch.cuda.LongTensor([sent_step_i])

                if self.position_enc is not None and self.position_enc_sent_feed_w:
                    # [batch, h_size]
                    sent_input_feed_w = sent_input_feed_w + self.position_enc(sent_step_i)

                if self.remove_input_feed_w:
                    # [batch, h_size]
                    sent_decoder_input = sent_input_feed_h
                elif self.remove_input_feed_h:
                    # [batch, h_size]
                    sent_decoder_input = sent_input_feed_w
                else:
                    # [batch, 2 * h_size]
                    sent_decoder_input = torch.cat([sent_input_feed_h, sent_input_feed_w], dim=1)
                # rnn_output: [batch, h_size]
                # dec_state: LSTM (h, c) or GRU (h,)
                if self.position_enc_sent_state:
                    sent_dec_state = (sent_dec_state[0] + self.position_enc(sent_step_i).unsqueeze(0),)
                sent_rnn_output, sent_dec_state = self.sent_dec_rnn(sent_decoder_input, sent_dec_state)
                # [batch, h_size], [batch, s_num]

                # add by wchen for target encoding
                if self.use_target_encoder:
                    if src_states_samples_2dlist is not None:
                        te_tgt_check = tgt[i, 1, :, 0]
                        saved_batch_idxs = self._get_unfinished_batch_idxes(te_tgt_check)
                        for te_b_idx in saved_batch_idxs:
                            # put the ground-truth target encoder state, need to call detach() first
                            src_states_samples_2dlist[te_b_idx][src_states_target_list[te_b_idx]] = \
                                self.src_encoder_final[te_b_idx, :].detach()
                            # the sep states of the target encoder are the sent_dec states
                            target_sep_states_2dlist[te_b_idx].append(sent_rnn_output[te_b_idx])

                if self.sent_attn is not None:
                    sent_attn_h, sent_p_attn = self.sent_attn(sent_rnn_output,
                                                              memory_bank=sent_memory_bank,
                                                              memory_lengths=sent_lengths,
                                                              coverage=sent_coverage)
                    # dropout sent_attn_h
                    # sent_attn_h = self.dropout(sent_attn_h)

                    # update the sent_coverage vector
                    if self.not_detach_coverage:
                        sent_coverage = sent_p_attn if sent_coverage is None else sent_coverage + sent_p_attn
                    else:
                        sent_coverage = sent_p_attn.detach() if sent_coverage is None else sent_coverage + sent_p_attn.detach()
                else:
                    sent_attn_h = None
                    sent_p_attn = None
                    sent_coverage = None

                # update the sent_states here for sent_state
                self.sent_state["hidden"] += [sent_dec_state]
                self.sent_state["input_feed_h"] += [sent_attn_h]
                self.sent_state["sent_std_attn"] += [sent_p_attn]
                self.sent_state["sent_coverage"] += [sent_coverage]
                # re-initialize the word decoding state
                if self.word_dec_init_type == 'attn_vec':
                    if self.position_enc is not None and self.position_enc_word_init:
                        final_word_init_state = sent_attn_h + self.position_enc(sent_step_i)
                    else:
                        final_word_init_state = sent_attn_h
                    self.sent_state["word_init_states"] += [final_word_init_state]
                    self.init_word_state(hidden=(final_word_init_state.unsqueeze(0), ))
                else:
                    # self.word_dec_init_type == 'hidden_vec'
                    # final_word_init_state = self.dropout(sent_dec_state[0].squeeze(0))
                    final_word_init_state = sent_dec_state[0].squeeze(0)
                    if self.position_enc is not None and self.position_enc_word_init:
                        final_word_init_state = final_word_init_state + self.position_enc(sent_step_i)
                    else:
                        final_word_init_state = final_word_init_state
                    self.sent_state["word_init_states"] += [final_word_init_state]
                    self.init_word_state(hidden=(final_word_init_state.unsqueeze(0), ))

            # word level decoding
            dec_outs_tmp = []
            word_attns_tmp = []
            # get the finished batch idx and tgt_s_len idx
            # [tgt_s_len x batch]
            tgt_check = tgt[i, :, :, 0]
            visited_end_tokens = [None]
            # get the finished index
            # [m, 2], [0, 2] for not found
            # 1. if last step output an sep token, we think the word level decoding is finished
            if self.sep_idx is not None:
                # 1.1 one sep_idx
                assert self.p_end_idx is None and self.a_end_idx is None
                word_dec_finished = (tgt_check == self.sep_idx).nonzero()
            else:
                # 1.2 two kinds of sep_idxs: p_end_idx, a_end_idx
                word_dec_finished_p = (tgt_check == self.p_end_idx).nonzero()
                word_dec_finished_a = (tgt_check == self.a_end_idx).nonzero()
                word_dec_finished = torch.cat([word_dec_finished_p, word_dec_finished_a], dim=0)

            visited_end_tokens.append(self.sep_idx)
            visited_end_tokens.append(self.p_end_idx)
            visited_end_tokens.append(self.a_end_idx)

            # 2. if last step output an eok token, we think the word level decoding is finished
            if self.eok_idx not in visited_end_tokens:
                word_dec_eok_finished = (tgt_check == self.eok_idx).nonzero()
                word_dec_finished = torch.cat([word_dec_finished, word_dec_eok_finished], dim=0)
            visited_end_tokens.append(self.eok_idx)

            # 3. if last step output an eos token, we think the word level decoding is finished
            if self.eos_idx not in visited_end_tokens and word_dec_finished.size(0) != tgt_batch:
                word_dec_eos_finished = (tgt_check == self.eos_idx).nonzero()
                word_dec_finished = torch.cat([word_dec_finished, word_dec_eos_finished], dim=0)
            if not testing:
                # 4. if the first word decoding step output a pad token, we think the word level decoding is finished
                # Only used for hr decoder training and validation
                word_dec_pad_finished = (tgt_check[0].unsqueeze(0) == self.pad_idx).nonzero()
                word_dec_finished = torch.cat([word_dec_finished, word_dec_pad_finished], dim=0)

            # Input feed concatenates hidden state with input at every time step.
            # [tgt_s_len, batch, emb_dim]
            sent_emb_i = sent_emb_i.squeeze(0)
            sent_p_attn = self.sent_state["sent_std_attn"][-1]
            # word level input_feed and hidden
            word_input_feed = self.word_state["input_feed"]
            word_input_feed_batch, _ = word_input_feed.size()
            word_dec_state = self.word_state["hidden"]
            _, word_dec_state_batch, _ = word_dec_state[0].size()
            aeq(tgt_batch, word_input_feed_batch, word_dec_state_batch)
            # word level decoding
            for j, word_emb_j in enumerate(sent_emb_i.split(1, dim=0)):
                # Word Level Decoding Finished Condition 1 (for both training and testing)
                # : 1.1 last step output an sep token
                # : 1.2 last step output an eok token
                # : 1.3 or an eos token
                # : 1.4 or first step output an pad token
                # check word dec finishing status at current j-th word level decoding step
                finished_num, _ = word_dec_finished.size()
                if finished_num != 0:
                    finished_num_idxs = (word_dec_finished[:, 0].view(-1) == j).nonzero()
                    finished_num_at_j, _ = finished_num_idxs.size()
                    if finished_num_at_j != 0:
                        for idx in range(finished_num_at_j):
                            b_idx = word_dec_finished[finished_num_idxs[idx, 0], 1].item()
                            assert b_idx not in self.word_state["finished_word_state"]

                            # finished_word_state: [1, h_size]
                            if self.input_feed_w_type == 'attn_vec':
                                self.word_state["finished_word_state"][b_idx] = word_input_feed[b_idx].unsqueeze(0)
                            elif self.input_feed_w_type == 'hidden_vec':
                                self.word_state["finished_word_state"][b_idx] = word_dec_state[0][:, b_idx, :]
                            elif self.input_feed_w_type == 'sec_attn_vec':
                                if len(self.word_state['input_feed_list']) < 2:
                                    saved_word_input_feed = self.word_state['input_feed_list'][-1]
                                else:
                                    saved_word_input_feed = self.word_state['input_feed_list'][-2]
                                self.word_state["finished_word_state"][b_idx] = \
                                    saved_word_input_feed[b_idx].unsqueeze(0)
                            elif self.input_feed_w_type == 'sec_hidden_vec':
                                if len(self.word_state['hidden_list']) < 2:
                                    saved_word_dec_state = self.word_state['hidden_list'][-1]
                                else:
                                    saved_word_dec_state = self.word_state['hidden_list'][-2]
                                self.word_state["finished_word_state"][b_idx] = \
                                    saved_word_dec_state[0][:, b_idx, :]

                            self.word_state["one_kp_not_finished"].remove(b_idx)

                # first_word_dec_step = (word_input_feed == 0).all()
                first_word_dec_step = len(self.word_state["input_feed_list"]) == 1
                first_valid_word_dec_step = len(self.word_state["input_feed_list"]) == 2

                # [batch, emb_dim]
                word_emb_j = word_emb_j.squeeze(0)
                # [batch, 2 * h_size]
                if self.use_zero_s_emb and first_word_dec_step:
                    word_emb_j = word_emb_j.zero_()

                # add position encoding if needed
                sent_step_i = len(self.sent_state["input_feed_w"]) - 1
                sent_step_i = torch.cuda.LongTensor([sent_step_i])

                if self.position_enc_start_token and first_word_dec_step:
                    word_emb_j = word_emb_j + self.position_enc_embsize(sent_step_i)
                if self.position_enc_first_word_feed and first_word_dec_step:
                    word_input_feed = word_input_feed + self.position_enc(sent_step_i)
                if self.position_enc_all_first_valid_word_dec_inputs and first_valid_word_dec_step:
                    word_emb_j = word_emb_j + self.position_enc(sent_step_i)[0, :self.embeddings.embedding_size].unsqueeze(0)
                    word_input_feed = word_input_feed + self.position_enc(sent_step_i)
                    word_dec_state = (word_dec_state[0] + self.position_enc(sent_step_i).unsqueeze(0), )

                word_dec_input = torch.cat([word_emb_j, word_input_feed], 1)
                # rnn_output: [batch, h_size]
                # dec_state: LSTM (h, c) or GRU (h,)
                word_rnn_output, word_dec_state = self.word_dec_rnn(word_dec_input, word_dec_state)
                # for orthogonal loss
                if first_word_dec_step:
                    assert j == 0
                    self.sent_state["orthog_states"].append(word_rnn_output)
                # word level attention
                if self.hr_enc:
                    if self.seqhr_enc:
                        # [batch, h_size], [batch, s_num, s_len]
                        word_attn_h, word_p_attn = self.word_attn(
                            word_rnn_output,
                            word_memory_bank,
                            memory_lengths=word_lengths,
                            sent_align_vectors=sent_p_attn,
                            sent_position_tuple=sent_position_tuple,
                            src_word_sent_ids=src_word_sent_ids)
                    else:
                        # [batch, h_size], [batch, s_num, s_len]
                        word_attn_h, word_p_attn = self.word_attn(
                            word_rnn_output,
                            word_memory_bank,
                            memory_lengths=word_lengths,
                            sent_align_vectors=sent_p_attn,
                            sent_nums=sent_lengths)
                elif self.seqE_HRD_rescale_attn:
                    # [batch, h_size], [batch, src_len]
                    word_attn_h, word_p_attn = self.word_attn(
                        word_rnn_output,
                        word_memory_bank,
                        memory_lengths=word_lengths,
                        sent_align_vectors=sent_p_attn)
                else:
                    # [batch, h_size], [batch, src_len]
                    word_attn_h, word_p_attn = self.word_attn(
                        word_rnn_output,
                        word_memory_bank,
                        memory_lengths=word_lengths)
                # [batch, h_size]
                decoder_output = self.dropout(word_attn_h)
                word_input_feed = decoder_output
                # update word level decoding state
                self.word_state["hidden"] = word_dec_state
                self.word_state["hidden_list"] += [word_dec_state]
                self.word_state["input_feed"] = word_input_feed
                self.word_state["input_feed_list"] += [word_input_feed]
                # # """Depreciate since one more pad token is added for each keyphrase"""
                # # Word Decoding Finished Condition 2 (only for training)
                # # : Meet the last step
                # if tgt_s_len != 1 and j == (tgt_s_len - 1):
                #     assert len(self.word_state["one_kp_not_finished"]) != 0
                #     for b_idx in self.word_state["one_kp_not_finished"]:
                #         assert b_idx not in self.word_state["finished_word_state"]
                #         # b_idx: [1, h_size]
                #         self.word_state["finished_word_state"][b_idx] = word_input_feed[b_idx].unsqueeze(0)
                #         self.word_state["one_kp_not_finished"].remove(b_idx)

                # save the word level decoding outputs and attentions
                dec_outs_tmp += [decoder_output]
                word_attns_tmp += [word_p_attn]

                if last_step:
                    # only used for translation
                    for b_idx in self.word_state["one_kp_not_finished"]:
                        # # finished_word_state: [1, h_size]
                        # if self.input_feed_w_type == 'attn_vec':
                        #     self.word_state["finished_word_state"][b_idx] = word_input_feed[b_idx].unsqueeze(0)
                        # else:
                        #     assert self.input_feed_w_type == 'hidden_vec'
                        #     self.word_state["finished_word_state"][b_idx] = word_dec_state[0][:, b_idx, :]

                        # finished_word_state: [1, h_size]
                        if self.input_feed_w_type == 'attn_vec':
                            self.word_state["finished_word_state"][b_idx] = word_input_feed[b_idx].unsqueeze(0)
                        elif self.input_feed_w_type == 'hidden_vec':
                            self.word_state["finished_word_state"][b_idx] = word_dec_state[0][:, b_idx, :]
                        elif self.input_feed_w_type == 'sec_attn_vec':
                            if len(self.word_state['input_feed_list']) < 2:
                                saved_word_input_feed = self.word_state['input_feed_list'][-1]
                            else:
                                saved_word_input_feed = self.word_state['input_feed_list'][-2]
                            self.word_state["finished_word_state"][b_idx] = \
                                saved_word_input_feed[b_idx].unsqueeze(0)
                        elif self.input_feed_w_type == 'sec_hidden_vec':
                            if len(self.word_state['hidden_list']) < 2:
                                saved_word_dec_state = self.word_state['hidden_list'][-1]
                            else:
                                saved_word_dec_state = self.word_state['hidden_list'][-2]
                            self.word_state["finished_word_state"][b_idx] = \
                                saved_word_dec_state[0][:, b_idx, :]

                    self.word_state["one_kp_not_finished"] = set()

            # [[tgt_s_len, batch, h_size], ]
            dec_outs += [torch.stack(dec_outs_tmp)]
            # For sequence encoding: [[tgt_s_len, batch, src_len], ]
            # For hr encoding: [[tgt_s_len, batch, s_num, s_len], ]
            attns["word_std"] += [torch.stack(word_attns_tmp)]

        #sent_input_feed_h = sent_attn_h
        # TODO: collect the final hidden state of one keyphrase h* or h
        #sent_input_feed_w = decoder_output

        if self._copy:
            assert self._reuse_copy_attn, "Only reuse copy attn is supported!"
            attns["copy"] = attns["word_std"]

        # add word_init_states for orthogonal regularization
        assert len(self.sent_state["orthog_states"]) == len(self.sent_state["word_init_states"])
        other_outs["orthog_states"] = self.sent_state["orthog_states"]
        # add sent_std_attn and sent_coverage for sentence level coverage
        if self.sent_attn is not None:
            other_outs["sent_std_attn"] = self.sent_state["sent_std_attn"]
            other_outs["sent_coverage"] = self.sent_state["sent_coverage"]

        # for target encoding like operation
        # ===================== calculate target attention distances ==================
        target_attn_dist = None
        target_sep_states_lens = None
        if src_states_samples_2dlist is not None:
            # [b_size, src_states_sample_size + 1, hidden_size]
            sampled_src_states = self._tensor_2dlist_to_tensor(
                src_states_samples_2dlist, tgt_batch, self.hidden_size,
                [self.src_states_sample_size + 1] * tgt_batch)
            # [b_size]
            target_sep_states_lens = [len(sep_state) for sep_state in target_sep_states_2dlist]
            # [b_size, max_target_sep_states_lens, target_h_size]
            target_sep_states = self._tensor_2dlist_to_tensor(target_sep_states_2dlist, tgt_batch,
                                                              self.hidden_size,
                                                              target_sep_states_lens)
            # compute attention distance, since the sampled_src_states are full, no need to input lengths
            # [b_size, max_target_sep_states_lens, src_states_sample_size + 1]
            target_attn_dist = self.target_attn(target_sep_states, sampled_src_states)
        other_outs["target_attns"] = (target_attn_dist, target_sep_states_lens, src_states_target_list)
        # Return result.
        return dec_outs, attns, other_outs

    def _get_unfinished_batch_idxes(self, tgt_check):
        """
        finde the batch idxs whose decoding process is not finished
        :param tgt_check: [batch_size]
        :return: a list of batch idxes whose decoding process is not finished
        """
        # collect the batch idxes whose decoding process is not finished
        # 1. the first valid word is not padding token
        # 2. the first valid word is not eos token
        b_size = tgt_check.size()[0]
        not_finished_batch_idxes = [i for i in range(b_size) if tgt_check[i] != self.pad_idx and tgt_check[i] != self.eos_idx]
        return not_finished_batch_idxes

    def _tensor_2dlist_to_tensor(self, tensor_2d_list, batch_size, hidden_size, seq_lens):
        """
        Code is from ken's github
        :param tensor_2d_list: a 2d list of tensor with size=[hidden_size], len(tensor_2d_list)=batch_size, len(tensor_2d_list[i])=seq_len[i]
        :param batch_size:
        :param hidden_size:
        :param seq_lens: a list that store the seq len of each batch, with len=batch_size
        :return: [batch_size, max_seq_len, hidden_size]
        """
        # assert tensor_2d_list[0][0].size() == torch.Size([hidden_size])
        assert len(tensor_2d_list) != 0
        assert len(tensor_2d_list[0]) != 0
        device = tensor_2d_list[0][0].device
        max_seq_len = max(seq_lens)
        for i in range(batch_size):
            for j in range(max_seq_len - seq_lens[i]):
                tensor_2d_list[i].append(torch.zeros(hidden_size).to(device))  # [hidden_size]
            tensor_2d_list[i] = torch.stack(tensor_2d_list[i], dim=0)  # [max_seq_len, hidden_size]
        tensor_3d = torch.stack(tensor_2d_list, dim=0)  # [batch_size, max_seq_len, hidden_size]
        return tensor_3d.contiguous()

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
            "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.models.stacked_rnn.StackedLSTM
        else:
            stacked_cell = onmt.models.stacked_rnn.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size


# add by wchen
class CatSeqDInputFeedRNNDecoder(nn.Module):
    """
    Base recurrent attention-based catseqD decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`models.NMTModel`.


    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[memory_bank]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general", attn_func="softmax",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False, sep_idx=None,
                 use_target_encoder=True, target_hidden_size=64,
                 src_states_capacity=128, src_states_sample_size=32, use_catSeq_dp=False):
        super(CatSeqDInputFeedRNNDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # add by wchen
        self.sep_idx = sep_idx
        self.use_target_encoder = use_target_encoder
        self.src_states_sample_size = src_states_sample_size
        self.target_hidden_size = target_hidden_size

        self.use_catSeq_dp = use_catSeq_dp

        # Decoder state
        self.state = {}

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        self.target_enc_rnn = None
        if use_target_encoder:
            self.target_enc_rnn = self._build_rnn(rnn_type,
                                                  input_size=self.embeddings.embedding_size,
                                                  hidden_size=target_hidden_size,
                                                  num_layers=1,
                                                  dropout=dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type, attn_func=attn_func
        )

        # Set up the target encoding attention
        self.src_states_queue = None
        self.target_attn = None
        if use_target_encoder:
            self.src_states_queue = SourceReprentationQueue(src_states_capacity)
            self.target_attn = \
                onmt.modules.TargetEncGlobalAttention(
                    tgt_enc_dim=target_hidden_size, src_enc_dim=hidden_size, attn_type=attn_type, attn_func=attn_func)

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type, attn_func=attn_func
            )
        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn

        # self.brigde_layer = nn.tanh(nn.Linear(hidden_size, hidden_size))

    def init_state(self, src, memory_bank, encoder_final):
        """ Init decoder state with last state of the encoder """
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
                # add by wchen, only use the last "self.num_layers" encoder layers' final hidden
                enc_layers = hidden.size(0)
                if enc_layers >= self.num_layers:
                    hidden = hidden[enc_layers - self.num_layers:]
                else:
                    # broadcast the hidden of the last encoder layer to initialize every layer of the decoder
                    hidden = [hidden[-1]] * self.num_layers
                    hidden = torch.stack(hidden, dim=0)
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)
            return hidden

        if isinstance(encoder_final, tuple):  # LSTM
            self.state["hidden"] = tuple([_fix_enc_hidden(enc_hid)
                                          for enc_hid in encoder_final])
        else:  # GRU
            self.state["hidden"] = (_fix_enc_hidden(encoder_final), )
        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)
        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = \
            self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
        self.state["coverage"] = None
        # add by wchen for target encoder
        # [b_size, h_size]
        self.src_encoder_final = self.state["hidden"][0].squeeze(0)
        self.state["target_enc_hidden"] =\
            (torch.zeros((1, batch_size, self.target_hidden_size)).to(self.state["hidden"][0].device),)

    def map_state(self, fn):
        self.state["hidden"] = tuple(map(lambda x: fn(x, 1),
                                         self.state["hidden"]))
        self.state["input_feed"] = fn(self.state["input_feed"], 1)

    def detach_state(self):
        """ Need to document this """
        self.state["hidden"] = tuple([_.detach()
                                     for _ in self.state["hidden"]])
        self.state["input_feed"] = self.state["input_feed"].detach()

    def forward(self, tgt, memory_bank, memory_lengths=None,
                step=None, test=False):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                 `[tgt_len x batch x nfeats]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            memory_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * dec_outs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # sample src final states and update the src_states_queue, code is from ken's github
        _, b_size, _ = tgt.size()
        if self.use_target_encoder and not test:
            if len(self.src_states_queue) < self.src_states_sample_size:
                src_states_samples_2dlist = None
                src_states_target_list = None
            else:
                src_states_samples_2dlist = []
                src_states_target_list = []
                for _ in range(b_size):
                    # N encoder states from the queue
                    src_states_samples_list, place_holder_idx = self.src_states_queue.sample(self.src_states_sample_size)
                    # insert the smaple list of one batch to the 2d list
                    src_states_samples_2dlist.append(src_states_samples_list)
                    # store the idx of place-holder for the batch
                    src_states_target_list.append(place_holder_idx)
        else:
            src_states_samples_2dlist = None
            src_states_target_list = None

        # Run the forward pass of the RNN.
        dec_state, dec_outs, attns, target_enc_hidden = self._run_forward_pass(
            tgt, memory_bank, memory_lengths=memory_lengths,
            src_states_samples_2dlist=src_states_samples_2dlist, src_states_target_list=src_states_target_list, test=test)

        if self.use_target_encoder and not test:
            # put all the encoder final states to the queue, Need to call detach() first
            # self.src_encoder_final: [b_size, h_size]
            [self.src_states_queue.put(self.src_encoder_final[i, :].detach()) for i in range(b_size)]

        # Update the state with the result.
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state
        self.state["input_feed"] = dec_outs[-1].unsqueeze(0)
        self.state["coverage"] = None
        if "coverage" in attns:
            self.state["coverage"] = attns["coverage"][-1].unsqueeze(0)
        self.state["target_enc_hidden"] = target_enc_hidden

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)
            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])
        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns

    def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None,
                          src_states_samples_2dlist=None, src_states_target_list=None, test=False):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        :param src_states_samples_2dlist: only effective when using target encoder, a 2dlist of tensor with dim=[memory_bank_size]
        :param src_states_target_list: a list that store the index of ground truth source representation for each batch, dim=[batch_size]
        """
        # Additional args check.
        input_feed = self.state["input_feed"].squeeze(0)
        input_feed_batch, _ = input_feed.size()
        _, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        # Initialize local and return variables.
        dec_outs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []
        # for collecting sep state
        sep_states_2dlist = [[] for _ in range(tgt_batch)]
        target_sep_states_2dlist = [[] for _ in range(tgt_batch)]

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        # add for catSeqD
        if self.use_catSeq_dp:
            emb = self.dropout(emb)

        dec_state = self.state["hidden"]
        # [layer, b_size, h_size]
        _, _, h_size = dec_state[0].size()
        coverage = self.state["coverage"].squeeze(0) \
            if self.state["coverage"] is not None else None

        target_enc_state = self.state["target_enc_hidden"]
        # Input feed concatenates hidden state with
        # input at every time step.
        rnn_output = None
        for pos_idx, emb_t in enumerate(emb.split(1)):

            # add by wchen for collect the sep state
            sep_batch_idxs = self._get_sep_batch_idxes(tgt[pos_idx, :, 0])

            if not test:
                for saved_batch_idx in sep_batch_idxs:
                    assert rnn_output is not None
                    sep_states_2dlist[saved_batch_idx].append(rnn_output[saved_batch_idx])

            emb_t = emb_t.squeeze(0)
            # add by wchen for target encoding
            if self.use_target_encoder:
                assert self.target_enc_rnn is not None
                # update the target encoder state
                target_enc_output, target_enc_state = self.target_enc_rnn(emb_t, target_enc_state)
                # update the decoder state
                decoder_input = torch.cat([emb_t, input_feed, target_enc_output.detach()], 1)

                if src_states_samples_2dlist is not None:
                    for saved_batch_idx in sep_batch_idxs:
                        # put the ground-truth target encoder state, need to call detach() first
                        src_states_samples_2dlist[saved_batch_idx][src_states_target_list[saved_batch_idx]] = \
                            self.src_encoder_final[saved_batch_idx, :].detach()
                        # collect the sep states of the target encoder
                        target_sep_states_2dlist[saved_batch_idx].append(target_enc_output[saved_batch_idx])
            else:
                decoder_input = torch.cat([emb_t, input_feed], 1)

            rnn_output, dec_state = self.rnn(decoder_input, dec_state)
            decoder_output, p_attn = self.attn(
                rnn_output,
                memory_bank.transpose(0, 1),
                memory_lengths=memory_lengths)
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                decoder_output = self.context_gate(
                    decoder_input, rnn_output, decoder_output
                )
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            dec_outs += [decoder_output]
            attns["std"] += [p_attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + p_attn \
                    if coverage is not None else p_attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy and not self._reuse_copy_attn:
                _, copy_attn = self.copy_attn(decoder_output,
                                              memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._copy:
                attns["copy"] = attns["std"]

        # ===================== get original sep_states ==================
        sep_states_lens = None
        sep_states = None
        if not test:
            sep_states_lens = [len(sep_state) for sep_state in sep_states_2dlist]
            sep_states = self._tensor_2dlist_to_tensor(sep_states_2dlist, tgt_batch, self.hidden_size, sep_states_lens)
        # ===================== calculate target attention distances ==================
        target_attn_dist = None
        target_sep_states_lens = None
        if src_states_samples_2dlist is not None:
            # [b_size, src_states_sample_size + 1, h_size]
            sampled_src_states = self._tensor_2dlist_to_tensor(
                src_states_samples_2dlist, tgt_batch, h_size, [self.src_states_sample_size + 1] * tgt_batch)
            # [b_size]
            target_sep_states_lens = [len(sep_state) for sep_state in target_sep_states_2dlist]
            # [b_size, max_target_sep_states_lens, target_h_size]
            target_sep_states = self._tensor_2dlist_to_tensor(target_sep_states_2dlist, tgt_batch, self.target_hidden_size, target_sep_states_lens)
            # [b_size, max_target_sep_states_lens, src_states_sample_size + 1]
            target_attn_dist = self.target_attn(target_sep_states, sampled_src_states)

        attns["sep_states"] = (sep_states, sep_states_lens)
        attns["target_attns"] = (target_attn_dist, target_sep_states_lens, src_states_target_list)
        # Return result.
        return dec_state, dec_outs, attns, target_enc_state

    def _tensor_2dlist_to_tensor(self, tensor_2d_list, batch_size, hidden_size, seq_lens):
        """
        Code is from ken's github
        :param tensor_2d_list: a 2d list of tensor with size=[hidden_size], len(tensor_2d_list)=batch_size, len(tensor_2d_list[i])=seq_len[i]
        :param batch_size:
        :param hidden_size:
        :param seq_lens: a list that store the seq len of each batch, with len=batch_size
        :return: [batch_size, max_seq_len, hidden_size]
        """
        # assert tensor_2d_list[0][0].size() == torch.Size([hidden_size])
        assert len(tensor_2d_list) != 0
        assert len(tensor_2d_list[0]) != 0
        device = tensor_2d_list[0][0].device
        max_seq_len = max(seq_lens)
        for i in range(batch_size):
            for j in range(max_seq_len - seq_lens[i]):
                tensor_2d_list[i].append(torch.zeros(hidden_size).to(device))  # [hidden_size]
            tensor_2d_list[i] = torch.stack(tensor_2d_list[i], dim=0)  # [max_seq_len, hidden_size]
        tensor_3d = torch.stack(tensor_2d_list, dim=0)  # [batch_size, max_seq_len, hidden_size]
        return tensor_3d.contiguous()

    def _get_sep_batch_idxes(self, tgt_check):
        """
        :param tgt_check: [batch_size]
        :return: list of batch idxes that is sep
        """
        sep_batch_idxes = []
        # collect the sep states of the target encoder
        sep_input_indicators = (tgt_check == self.sep_idx).nonzero()
        sep_num, _ = sep_input_indicators.size()
        if sep_num != 0:
            for indi_idx in range(sep_num):
                saved_batch_idx = sep_input_indicators[indi_idx, 0]
                sep_batch_idxes.append(saved_batch_idx)
        return sep_batch_idxes

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
                                      "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.models.stacked_rnn.StackedLSTM
        else:
            stacked_cell = onmt.models.stacked_rnn.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size + self.target_hidden_size


class CatSeqCorrInputFeedRNNDecoder(nn.Module):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[memory_bank n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general", attn_func="softmax",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False):
        super(CatSeqCorrInputFeedRNNDecoder, self).__init__()

        # Basic attributes.
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Decoder state
        self.state = {}

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(
            hidden_size, coverage=coverage_attn,
            attn_type=attn_type, attn_func=attn_func
        )

        # Set up the review attention
        self.review_attn = onmt.modules.GlobalAttention(
            hidden_size, coverage=False,
            attn_type=attn_type, attn_func=attn_func
        )

        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type, attn_func=attn_func
            )
        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn

    def init_state(self, src, memory_bank, encoder_final):
        """ Init decoder state with last state of the encoder """
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
                # add by wchen, only use the last "self.num_layers" encoder layers' final hidden
                enc_layers = hidden.size(0)
                if enc_layers >= self.num_layers:
                    hidden = hidden[enc_layers - self.num_layers:]
                else:
                    # broadcast the hidden of the last encoder layer to initialize every layer of the decoder
                    hidden = [hidden[-1]] * self.num_layers
                    hidden = torch.stack(hidden, dim=0)
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)
            return hidden

        if isinstance(encoder_final, tuple):  # LSTM
            self.state["hidden"] = tuple([_fix_enc_hidden(enc_hid)
                                          for enc_hid in encoder_final])
        else:  # GRU
            self.state["hidden"] = (_fix_enc_hidden(encoder_final), )

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)
        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = \
            self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
        self.state["coverage"] = None
        # add for review mechanism
        self.state["previous_hiddens"] = []
        self.state["zero_vec"] = self.state["hidden"][0].data.new(*h_size).zero_()

    def map_state(self, fn):
        self.state["hidden"] = tuple(map(lambda x: fn(x, 1),
                                         self.state["hidden"]))
        self.state["input_feed"] = fn(self.state["input_feed"], 1)

    def detach_state(self):
        """ Need to document this """
        self.state["hidden"] = tuple([_.detach()
                                     for _ in self.state["hidden"]])
        self.state["input_feed"] = self.state["input_feed"].detach()

    def forward(self, tgt, memory_bank, memory_lengths=None,
                step=None, test=False):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                 `[tgt_len x batch x nfeats]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            memory_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * dec_outs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        # Run the forward pass of the RNN.
        dec_state, dec_outs, attns = self._run_forward_pass(
            tgt, memory_bank, memory_lengths=memory_lengths, test=test)

        # Update the state with the result.
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state
        self.state["input_feed"] = dec_outs[-1].unsqueeze(0)
        self.state["coverage"] = None
        if "coverage" in attns:
            self.state["coverage"] = attns["coverage"][-1].unsqueeze(0)

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)

            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])
        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns

    def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None, test=False):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        input_feed = self.state["input_feed"].squeeze(0)
        input_feed_batch, _ = input_feed.size()
        _, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        # Initialize local and return variables.
        dec_outs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        dec_state = self.state["hidden"]
        coverage = self.state["coverage"].squeeze(0) \
            if self.state["coverage"] is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for _, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)

            # Review mechanism
            if len(self.state["previous_hiddens"]) != 0:
                # [b_size, t-1, h_size]
                review_memory_bank = torch.stack(self.state['previous_hiddens'], dim=1)
                # [b_size, h_size]
                review_query = self.state['previous_hiddens'][-1]
                # [b_size, 1, h_size]
                review_attn_h, review_attn_scores = self.review_attn(review_query, review_memory_bank)
            else:
                review_attn_h = self.state["zero_vec"]

            decoder_input = torch.cat([emb_t, input_feed, review_attn_h], 1)
            rnn_output, dec_state = self.rnn(decoder_input, dec_state)

            # For Review mechanism
            self.state["previous_hiddens"].append(rnn_output)

            decoder_output, p_attn = self.attn(
                rnn_output,
                memory_bank.transpose(0, 1),
                memory_lengths=memory_lengths,
                coverage=coverage)
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                decoder_output = self.context_gate(
                    decoder_input, rnn_output, decoder_output
                )
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            dec_outs += [decoder_output]
            attns["std"] += [p_attn]

            # Update the coverage attention.
            if self._coverage:
                coverage = coverage + p_attn \
                    if coverage is not None else p_attn
                attns["coverage"] += [coverage]

            # Run the forward pass of the copy attention layer.
            if self._copy and not self._reuse_copy_attn:
                _, copy_attn = self.copy_attn(decoder_output,
                                              memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._copy:
                attns["copy"] = attns["std"]
        # Return result.
        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
            "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = onmt.models.stacked_rnn.StackedLSTM
        else:
            stacked_cell = onmt.models.stacked_rnn.StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + 2 * self.hidden_size
