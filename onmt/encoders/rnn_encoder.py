"""Define RNN-based encoders."""
import torch.nn as nn
import torch.nn.functional as F

# add by wchen
import torch

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory

# add by wchen
from onmt.utils.misc import aeq
from onmt.utils.invalid_sent_processor import valid_src_compress, recover_src


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False, use_catSeq_dp=False):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        # add by wchen
        self.use_catSeq_dp = use_catSeq_dp
        self.num_layers = num_layers
        self.catSeq_dp = None
        if use_catSeq_dp:
            self.catSeq_dp = nn.Dropout(dropout)
        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    def forward(self, src, lengths=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(src, lengths)

        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()

        # add by wchen
        if self.use_catSeq_dp:
            emb = self.catSeq_dp(emb)

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths_list)

        memory_bank, encoder_final = self.rnn(packed_emb)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        # add by wchen
        if self.use_catSeq_dp and self.num_layers == 1:
            memory_bank = self.catSeq_dp(memory_bank)

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)
        return encoder_final, memory_bank, lengths

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            # return F.relu(result).view(size)
            # changed from relu to tanh for catseqD bridge, the original is the above
            return F.tanh(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs


class HREncoder(EncoderBase):
    """ A generic hierarchical recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super(HREncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.word_rnn, self.word_no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)
        assert not self.word_no_pack_padded_seq

        self.sent_rnn, self.sent_no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=hidden_size * num_directions,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)
        assert not self.sent_no_pack_padded_seq

        # add by wchen
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        # self.dropout = nn.Dropout(dropout)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    def forward(self, src, lengths=None):
        "See :obj:`EncoderBase.forward()`"
        # self._check_args(src, lengths)
        assert type(lengths) is tuple, "The src lengths for HREncoder should be a tuple."
        # get the sent number and the sent lens
        sent_nums, sent_lens = lengths
        assert sent_nums is not None
        assert sent_lens is not None

        batch, s_num, s_len, f_num = src.size()
        # we do args check here
        batch_1 = sent_nums.size(0)
        batch_2, s_num_ = sent_lens.size()
        aeq(batch, batch_1, batch_2)
        aeq(s_num, s_num_)

        # [s_total, s_len, f_num], [s_total]
        valid_src, valid_sent_lens = valid_src_compress(src, sent_nums, sent_lens)

        # sort each sentence w.r.t the sentence length
        sorted_valid_sent_lens, idx_sort = torch.sort(valid_sent_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        # sort
        # [s_total, s_len, f_num]
        valid_src = valid_src.index_select(0, idx_sort)

        # -> [s_total, s_len, emb_dim]
        emb = self.embeddings(valid_src)

        packed_emb = emb
        # Lengths data is wrapped inside a Tensor.
        sorted_valid_sent_lens_list = sorted_valid_sent_lens.view(-1).tolist()
        packed_emb = pack(emb, sorted_valid_sent_lens_list, batch_first=True)
        word_memory_bank, word_encoder_final = self.word_rnn(packed_emb)
        # word_encoder_final: [num_layers * num_directions, s_total, hidden_size]
        # adjust the memory bank
        # -> [s_total, s_len, 2*hidden_size]
        word_memory_bank = unpack(word_memory_bank, batch_first=True)[0]
        # recover the memory_bank order
        word_memory_bank = word_memory_bank.index_select(0, idx_unsort)

        # adjust the final hidden state
        # only select the final state of the last layer [num_layers * num_directions, s_total, hidden_size]
        # [num_layers * num_directions, s_total, hidden_size]
        word_encoder_final = word_encoder_final.index_select(1, idx_unsort)

        if self.num_directions == 2:
            # -> [num_layers, s_total, 2*hidden_size]
            l_mul_d = word_encoder_final.size(0)
            word_encoder_final = torch.cat([word_encoder_final[0:l_mul_d:2],
                                            word_encoder_final[1:l_mul_d:2]], dim=2)

        # only use the final state of the last layer
        # -> [s_total, 2*hidden_size] if brnn or [s_total, hidden_size] if unidirectional rnn
        word_encoder_final = word_encoder_final[-1]

        # recover the word memory bank
        word_memory_bank = recover_src(word_memory_bank, sent_nums)
        # recover the word enc final
        word_encoder_final = recover_src(word_encoder_final, sent_nums)
        # # dropout the word_encoder_final before input the sentence decoder
        # word_encoder_final = self.dropout(word_encoder_final)

        # sentence level encoding, based on sent_nums
        # 1. sort w.r.t sent_nums
        sorted_sent_nums, idx_sort = torch.sort(sent_nums, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        # 2. encode the sorted sent hidden state
        packed_input = word_encoder_final.index_select(0, idx_sort)
        sorted_sent_nums_list = sorted_sent_nums.view(-1).tolist()
        packed_input = pack(packed_input, sorted_sent_nums_list, batch_first=True)
        sent_memory_bank, sent_encoder_final = self.sent_rnn(packed_input)

        # -> [batch, s_num, 2 * hidden_size]
        sent_memory_bank = unpack(sent_memory_bank, batch_first=True)[0]
        # recover the order
        sent_memory_bank = sent_memory_bank.index_select(0, idx_unsort)

        # sent_encoder_final: [num_layers * num_directions, batch, hidden_size]
        # recover the orger
        sent_encoder_final = sent_encoder_final.index_select(1, idx_unsort)

        if self.use_bridge:
            sent_encoder_final = self._bridge(sent_encoder_final)

        return sent_encoder_final, (sent_memory_bank, word_memory_bank), lengths

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs


class SeqHREncoder(EncoderBase):
    """ A generic sequentially hierarchical recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False, output_word_final=False):
        super(SeqHREncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.word_rnn, self.word_no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)
        assert not self.word_no_pack_padded_seq

        self.sent_rnn, self.sent_no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=hidden_size * num_directions,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)
        assert not self.sent_no_pack_padded_seq

        # add by wchen
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.output_word_final = output_word_final
        self.dropout = nn.Dropout(dropout)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    def forward(self, src, src_lengths=None, sent_position_tuple=None):
        "See :obj:`EncoderBase.forward()`"
        # self._check_args(src, src_lengths)
        assert src_lengths is not None
        assert isinstance(sent_position_tuple, tuple), "The sent_position for seqHREncoder should be a tuple."

        # sent_p: [batch_size, s_num, 2], sent_nums: [batch_size]
        sent_p, sent_nums = sent_position_tuple

        # we do args check here
        src_len, batch, f_num = src.size()
        batch_1 = src_lengths.size(0)
        batch_2, s_num, _ = sent_p.size()
        batch_3 = sent_nums.size(0)
        aeq(batch, batch_1, batch_2, batch_3)

        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()

        # Lengths data is wrapped inside a Tensor.
        lengths_list = src_lengths.view(-1).tolist()
        packed_emb = pack(emb, lengths_list)

        # word_memory_bank: [s_len, batch_size, 2*h_size]
        # word_encoder_final: [2, batch_size, h_size]
        word_memory_bank, word_encoder_final = self.word_rnn(packed_emb)
        word_memory_bank = unpack(word_memory_bank)[0]

        # sentence level encoding
        # 1. get the sent representation form word_memory_bank using sent_position
        # [sent_num, batch_size, 2]
        # sent_p = sent_p.transpose(0, 1)
        # method 1, 35s for 500 examples
        # # [src_len, batch_size, h_size]
        # f_word_memory_bank = word_memory_bank[:, :, :self.hidden_size]
        # b_word_memory_bank = word_memory_bank[:, :, self.hidden_size:]
        # dim==0: out[i][j][k] = input[index[i][j][k]][j][k]
        # [sent_num, batch_size, h_size]
        # f_index = sent_p[:, :, 0].unsqueeze(-1).expand(-1, -1, self.hidden_size)
        # f_sent_vector = f_word_memory_bank.gather(dim=0, index=f_index)
        # b_index = sent_p[:, :, 1].unsqueeze(-1).expand(-1, -1, self.hidden_size)
        # b_sent_vector = b_word_memory_bank.gather(dim=0, index=b_index)
        # [sent_num, batch_size, 2 * h_size]
        # sent_vector = torch.cat([f_sent_vector, b_sent_vector], dim=-1)

        # method 2, 36s for 500 examples
        sent_p = sent_p.transpose(0, 1)
        # [sent_num, batch_size, 2 * h_size]
        f_index = sent_p[:, :, 0].unsqueeze(-1).expand(-1, -1, self.hidden_size)
        b_index = sent_p[:, :, 1].unsqueeze(-1).expand(-1, -1, self.hidden_size)
        gather_index = torch.cat([f_index, b_index], dim=-1)
        sent_vector = word_memory_bank.gather(dim=0, index=gather_index)

        # # method 3, 36s for 500 examples
        # sent_vector = []
        # for b_idx in range(batch):
        #     sent_vector_b = []
        #     for sent_idx in range(s_num):
        #         if sent_p[b_idx, sent_idx].sum() != 0:
        #             f_sent_vector_i = word_memory_bank[sent_p[b_idx, sent_idx, 0], b_idx, :self.hidden_size]
        #             b_sent_vector_i = word_memory_bank[sent_p[b_idx, sent_idx, 1], b_idx, self.hidden_size:]
        #             sent_vector_b.append(torch.cat([f_sent_vector_i, b_sent_vector_i]))
        #         else:
        #             sent_vector_b.append(torch.zeros([2 * self.hidden_size],
        #                                              dtype=word_memory_bank.dtype,
        #                                              device=word_memory_bank.device))
        #     # [s_num, 2 * h_size]
        #     sent_vector_b = torch.stack(sent_vector_b, dim=0)
        #     sent_vector.append(sent_vector_b)
        # # [s_num, batch, 2 * h_size]
        # sent_vector = torch.stack(sent_vector, dim=1)

        # dropout the sentence vector
        sent_vector = self.dropout(sent_vector)

        # 2. use sent_rnn to encode the sentence representations
        sorted_sent_nums, idx_sort = torch.sort(sent_nums, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        # use the sorted order
        sent_vector = sent_vector.index_select(1, idx_sort)
        sorted_sent_nums_list = sorted_sent_nums.view(-1).tolist()
        packed_emb = pack(sent_vector, sorted_sent_nums_list)
        sent_memory_bank, sent_encoder_final = self.sent_rnn(packed_emb)
        sent_memory_bank = unpack(sent_memory_bank)[0]
        # recover the original order
        sent_memory_bank = sent_memory_bank.index_select(1, idx_unsort)
        sent_encoder_final = sent_encoder_final.index_select(1, idx_unsort)

        out_final = sent_encoder_final
        if self.output_word_final:
            out_final = word_encoder_final
        if self.use_bridge:
            out_final = self._bridge(out_final)

        return out_final, (sent_memory_bank, word_memory_bank), src_lengths

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs


class TGEncoder(EncoderBase):
    """
    Title-guided encoder
    (src -> BiRNN1, title -> BiRNN3) -> match layer -> BiRNN2->Decoder
    """
    def __init__(self, rnn_type, bidirectional, num_layers, hidden_size,
                 dropout, embeddings):
        super(TGEncoder, self).__init__()

        assert bidirectional
        # In current version, only one BiGRU layer for context encoding, one BiGRU layer for title encoding
        # one BiGRU layer for merging.
        # We regard num_layers as 2 since context encoding and title encoding are at the same level.
        assert num_layers == 2
        self.rnn_type = rnn_type
        hidden_size = hidden_size // 2
        self.real_hidden_size = hidden_size
        # self.no_pack_padded_seq = False
        self.bidirectional = bidirectional
        # TODO: set res_ratio as an argument
        self.res_ratio = 0.5

        self.embeddings = embeddings

        # One BiGRU layer for context encoding
        self.src_rnn, self.src_no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=1,
                        bidirectional=bidirectional)
        # self.src_rnn = getattr(nn, rnn_type)(
        #     input_size=embeddings.embedding_size,
        #     hidden_size=hidden_size,
        #     num_layers=1,
        #     bidirectional=self.bidirectional)

        # One BiGRU layer for title encoding
        self.query_rnn, self.query_no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=1,
                        bidirectional=bidirectional)
        # self.query_rnn = getattr(nn, rnn_type)(
        #     input_size=embeddings.embedding_size,
        #     hidden_size=hidden_size,
        #     num_layers=1,
        #     bidirectional=self.bidirectional)

        # One BiGRU layer for merging
        self.merge_rnn, self.merge_no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=4 * hidden_size,
                        hidden_size=hidden_size,
                        num_layers=1,
                        bidirectional=bidirectional)
        # self.merge_rnn = getattr(nn, rnn_type)(
        #     input_size=4 * hidden_size,
        #     hidden_size=hidden_size,
        #     num_layers=1,
        #     bidirectional=self.bidirectional)

        self.match_fc = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def sequence_mask(self, lengths, max_len=None):
        """
        Creates a boolean mask from sequence lengths.
        """
        batch_size = lengths.numel()
        max_len = max_len or lengths.max()
        return (torch.arange(0, max_len)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))

    def query_match(self, src_seq, query_seq, query_lengths):
        """
        Attentive Matching Layer
        :param src_seq: ``FloatTensor'', the encoded context vectors, [src_seq_len, batch, 2*hidden_size]
        :param query_seq: ``FloatTensor'', the encoded title vectors, [query_seq_len, batch, 2*hidden_size]
        :param query_lengths: ``LongTensor'', the title lengths, [batch]
        :return: matched_seq, the aggregated vectors of each src word from the title
        """
        BF_query_mask = self.sequence_mask(query_lengths)  # [batch, query_seq_len]
        BF_src_outputs = src_seq.transpose(0, 1)  # [batch, src_seq_len, 2*hidden_size]
        BF_query_outputs_orig = query_seq.transpose(0, 1)  # [batch, query_seq_len, 2*hidden_size]
        BF_query_outputs = self.match_fc(BF_query_outputs_orig)  # [batch, query_seq_len, 2*hidden_size]

        # compute attention scores
        scores = BF_src_outputs.bmm(BF_query_outputs.transpose(2, 1))  # [batch, src_seq_len, query_seq_len]

        # mask padding
        Expand_BF_query_mask = BF_query_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_((1 - Expand_BF_query_mask).byte(), -float('inf'))

        # normalize with softmax
        alpha = F.softmax(scores, dim=2)  # [batch, src_seq_len, query_seq_len]

        # take the weighted average and transpose to length first
        # 1. the corrected version, which is consistent with the description of the paper
        BF_matched_seq = alpha.bmm(BF_query_outputs_orig)  # [batch, src_seq_len, 2*hidden_size]
        # # 2. the experimented version, which is used to run the experiments in the paper
        # # Note: Sorry for the inconsistency mistake! But these two versions get similar performance!
        # BF_matched_seq = alpha.bmm(BF_query_outputs)  # [batch, src_seq_len, 2*hidden_size]

        matched_seq = BF_matched_seq.transpose(0, 1)  # [src_seq_len, batch, 2*hidden_size]

        return matched_seq

    def forward(self, input, lengths=None, encoder_state=None):
        """
        Title-Guided Encoding
        :param input: a ``Tuple'', (src_input, query_input), src_input: [src_seq_len, batch, feat_num], query_input: [query_seq_len, batch, feat_num]
        :param lengths: a ``Tuple'', (src_lengths, query_lengths), src_lengths: [batch], query_lengths: [batch]
        :param encoder_state: the encoder initial state
        :return: (src_hidden_2, src_outputs)
        """
        assert isinstance(input, tuple)
        assert isinstance(lengths, tuple)
        src_input, query_input = input
        src_lengths, query_lengths = lengths

        # check whether the batch sizes are consistent
        self._check_args(src_input, src_lengths, encoder_state)
        self._check_args(query_input, query_lengths, encoder_state)

        src_input = self.embeddings(src_input)    # [src_seq_len, batch_size, emb_dim]
        query_input = self.embeddings(query_input)  # [query_seq_len, batch_size, emb_dim]

        # sort query w.r.t lengths
        sorted_query_lengths, idx_sort = torch.sort(query_lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        # context encoding
        packed_input = src_input
        if src_lengths is not None and not self.src_no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            packed_input = pack(src_input, src_lengths.view(-1).tolist())
        src_outputs, src_hidden_1 = self.src_rnn(packed_input, encoder_state)
        if src_lengths is not None and not self.src_no_pack_padded_seq:
            # src encoded outputs
            src_outputs = unpack(src_outputs)[0]

        # add for residual link
        res_src_outputs = src_outputs

        # query encoding
        packed_query_input = query_input.index_select(1, idx_sort)
        if sorted_query_lengths is not None and not self.query_no_pack_padded_seq:
            packed_query_input = pack(packed_query_input, sorted_query_lengths.view(-1).tolist())
        # separate BiRNN
        query_outputs, query_hidden = self.query_rnn(packed_query_input, encoder_state)
        if sorted_query_lengths is not None and not self.query_no_pack_padded_seq:
            query_outputs = unpack(query_outputs)[0]
            # recover the original order
            query_outputs = query_outputs.index_select(1, idx_unsort)
            # h, c
            if self.rnn_type == 'LSTM':
                query_hidden = tuple([query_hidden[i].index_select(1, idx_unsort) for i in range(2)])
            elif self.rnn_type == 'GRU':
                query_hidden = query_hidden.index_select(1, idx_unsort) # [2, batch, hidden_size]

        # src-query matching
        attn_matched_seq = self.query_match(src_outputs, query_outputs, query_lengths)  # [src_seq_len, batch, 2*hidden_size]

        src_outputs = torch.cat([src_outputs, attn_matched_seq], dim=-1)    # [src_seq_len, batch, 4*hidden_size]

        # dropout
        src_outputs = self.dropout(src_outputs)

        # final merge layer
        packed_input = src_outputs
        if src_lengths is not None and not self.merge_no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            packed_input = pack(src_outputs, src_lengths.view(-1).tolist())
        # merging
        src_outputs, src_hidden_2 = self.merge_rnn(packed_input, encoder_state)
        if src_lengths is not None and not self.merge_no_pack_padded_seq:
            # merged encoding output
            src_outputs = unpack(src_outputs)[0]
        src_outputs = self.res_ratio * res_src_outputs + (1 - self.res_ratio) * src_outputs
        return src_hidden_2, src_outputs, src_lengths