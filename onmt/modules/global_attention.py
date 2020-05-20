""" Global attention modules (Luong / Bahdanau) """
import torch
import torch.nn as nn
import torch.nn.functional as F

from onmt.modules.sparse_activations import sparsemax
from onmt.utils.misc import aeq, sequence_mask

# add by wchen
from onmt.utils.invalid_sent_processor import valid_src_compress, recover_src

# This class is mainly used by decoder.py for RNNs but also
# by the CNN / transformer decoder when copy attention is used
# CNN has its own attention mechanism ConvMultiStepAttention
# Transformer has its own MultiHeadedAttention


class GlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """

    def __init__(self, dim, coverage=False, attn_type="dot",
                 attn_func="softmax"):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type.")
        self.attn_type = attn_type
        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function.")
        self.attn_func = attn_func

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, source, memory_bank, memory_lengths=None, coverage=None):
        """

        Args:
          source (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """

        # one step input
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        else:
            one_step = False

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)
        if coverage is not None:
            batch_, source_l_ = coverage.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            memory_bank = memory_bank + self.linear_cover(cover).view_as(memory_bank)
            memory_bank = torch.tanh(memory_bank)

        # compute attention scores, as in Luong et al.
        align = self.score(source, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(1 - mask, -float('inf'))

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            align_vectors = F.softmax(align.view(batch*target_l, source_l), -1)
        else:
            align_vectors = sparsemax(align.view(batch*target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, memory_bank)

        # concatenate
        concat_c = torch.cat([c, source], 2).view(batch*target_l, dim*2)
        attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = torch.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            aeq(dim, dim_)
            batch_, source_l_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()
            # Check output sizes
            target_l_, batch_, dim_ = attn_h.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            aeq(dim, dim_)
            target_l_, batch_, source_l_ = align_vectors.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        return attn_h, align_vectors


# add by wchen
# sentence to word global attention
class WordGlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    All models compute the output as
    :math:`c = sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """

    def __init__(self, dim, coverage=False, attn_type="dot",
                 attn_func="softmax", output_attn_h=False):
        super(WordGlobalAttention, self).__init__()

        self.dim = dim
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type.")
        self.attn_type = attn_type
        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function.")
        self.attn_func = attn_func

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)

        self.output_attn_h = output_attn_h
        if output_attn_h:
            # mlp wants it with bias
            out_bias = self.attn_type == "mlp"
            self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`
          attn_level ('str'): the string indicator of current attention level

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """
        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, tgt_len, d) x (batch, d, src_len) --> (batch, tgt_len, src_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, source, memory_bank, memory_lengths=None, coverage=None,
                sent_align_vectors=None, sent_nums=None):
        """
        Only one-step attention is supported now.
        Args:
          source (`FloatTensor`): query vectors `[batch x dim]`
          memory_bank (`FloatTensor`): word_memory_bank is `FloatTensor` with shape `[batch x s_num x s_len x dim]`
          sent_lens (`LongTensor`): for word_memory_bank, `[batch x s_num]`
          coverage (`FloatTensor`): None (not supported yet)
          sent_align_vectors (`FloatTensor`): the computed sentence align distribution, `[batch x s_num]`
          sent_nums (`LongTensor`): the sentence numbers of inputs, `[batch]`
          use_tanh (`bool`): True, whether use tanh activation function for `general` and 'dot' attention

        Returns:
          (`FloatTensor`, `FloatTensor`):
            * Computed word attentional vector `[batch x dim]`
            * Word Attention distribtutions for the query of word `[batch x s_num x s_len]`
        """

        # only one step input is supported
        assert source.dim() == 2, "Only one step input is supported for current attention."
        one_step = True
        # [batch, 1, dim]
        source = source.unsqueeze(1)
        batch, tgt_l, dim = source.size()

        # check the specification for word level attention
        assert sent_align_vectors is not None, "For word level attention, the 'sent_align' must be specified."
        assert sent_nums is not None, "For word level attention, the 'sent_nums' must be specified."
        assert memory_lengths is not None, "The lengths for the word memory bank are required."
        sent_lens = memory_lengths

        batch_1, s_num, s_len, dim_ = memory_bank.size()
        batch_2, s_num_ = sent_align_vectors.size()
        batch_3 = sent_nums.size(0)

        aeq(batch, batch_1, batch_2, batch_3)
        aeq(dim, dim_, self.dim)
        aeq(s_num, s_num_)

        # if coverage is not None:
        #     batch_, source_l_ = coverage.size()
        #     aeq(batch, batch_)
        #     aeq(source_l, source_l_)
        #
        # if coverage is not None:
        #     cover = coverage.view(-1).unsqueeze(1)
        #     memory_bank += self.linear_cover(cover).view_as(memory_bank)
        #     memory_bank = torch.tanh(memory_bank)

        # compute word attention scores, as in Luong et al.
        # [batch, s_num, s_len, dim] -> [batch, s_num * s_len, dim]
        memory_bank = memory_bank.view(batch, s_num * s_len, dim)
        # [batch, 1, s_num * s_len]
        word_align = self.score(source, memory_bank)
        # [batch, s_num * s_len]
        word_align = word_align.squeeze(1)
        # [batch, s_num, s_len]
        word_align = word_align.view(batch, s_num, s_len)

        # remove the empty sentences
        # [s_toal, s_len], [s_total]
        valid_word_align, valid_sent_lens = valid_src_compress(word_align, sent_nums=sent_nums, sent_lens=sent_lens)

        # [s_toal, s_len]
        word_mask = sequence_mask(valid_sent_lens, max_len=valid_word_align.size(-1))

        # word_mask = word_mask.view(batch, s_num, s_len)

        # # [batch, s_num]
        # sent_mask = sequence_mask(sent_nums, max_len=s_num)
        # # [batch, s_num, 1]
        # sent_mask = sent_mask.unsqueeze(2)
        # # [batch, s_num, s_len]
        # align_vectors.masked_fill_(1 - sent_mask, 0.0)

        # [s_total, s_len]
        valid_word_align.masked_fill_(1 - word_mask, -float('inf'))

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            align_vectors = F.softmax(valid_word_align, -1)
        else:
            align_vectors = sparsemax(valid_word_align, -1)

        # Recover the original shape by pad 0.s for empty sentence
        # [batch, s_num, s_len]
        align_vectors = recover_src(align_vectors, sent_nums)

        # # For the whole invalid sentence, we set all the word aligns to 0s.
        # # Since
        # # [batch, s_num]
        # sent_mask = sequence_mask(sent_nums, max_len=s_num)
        # # [batch, s_num, 1]
        # sent_mask = sent_mask.unsqueeze(2)
        # # [batch, s_num, s_len]
        # align_vectors.masked_fill_(1 - sent_mask, 0.0)

        # [batch, s_num, 1]
        sent_align_vectors = sent_align_vectors.unsqueeze(-1)
        # [batch, s_num, s_len]
        align_vectors = align_vectors * sent_align_vectors

        # each context vector c_t is the weighted average
        # over all the source hidden states
        # [batch, 1, s_num * s_len]
        align_vectors = align_vectors.view(batch, -1).unsqueeze(1)
        # [batch, 1, s_num * s_len] x [batch, s_num * s_len, dim] -> [batch, 1, dim]
        c = torch.bmm(align_vectors, memory_bank)
        # [batch, dim]
        c = c.squeeze(1)
        returned_vec = c

        # If output_attn_h == False, we put linear out layer into decoder part
        if self.output_attn_h:
            # concatenate
            # [batch, dim]
            source = source.squeeze(1)
            # [batch, 2*dim]
            concat_c = torch.cat([c, source], 1)
            # [batch, dim]
            attn_h = self.linear_out(concat_c)
            if self.attn_type in ["general", "dot"]:
                attn_h = torch.tanh(attn_h)
            returned_vec = attn_h

        align_vectors = align_vectors.squeeze(1).view(batch, s_num, s_len)
        # Check output sizes
        batch_, dim_ = returned_vec.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        # check
        batch_, s_num_, s_len_ = align_vectors.size()
        aeq(batch, batch_)
        aeq(s_num, s_num_)

        return returned_vec, align_vectors


class MyGlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.

    All models compute the output as
    :math:`c = sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """

    def __init__(self, dim, coverage=False, attn_type="dot",
                 attn_func="softmax", output_attn_h=False):
        super(MyGlobalAttention, self).__init__()

        self.dim = dim
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type.")
        self.attn_type = attn_type
        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function.")
        self.attn_func = attn_func

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)

        # # wchen: we put the linear_out layer into the decoder module
        # mlp wants it with bias
        self.output_attn_h = output_attn_h
        if output_attn_h:
            out_bias = self.attn_type == "mlp"
            self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        self.coverage_attn = coverage
        # if coverage:
        #     self.linear_cover = nn.Linear(1, 1, bias=False)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, source, memory_bank, memory_lengths=None, coverage=None):
        """

        Args:
          source (`FloatTensor`): query vectors `[batch x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[batch x dim]`
          * Attention distribtutions for each query
             `[batch x src_len]`
        """

        # one step input
        assert source.dim() == 2, "Only one step input is supported"
        #one_step = True
        source = source.unsqueeze(1)

        batch, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)

        # compute attention scores, as in Luong et al.
        # [batch x 1 x src_len]
        align = self.score(source, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(1 - mask, -float('inf'))

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            align_vectors = F.softmax(align.view(batch*target_l, source_l), -1)
        else:
            align_vectors = sparsemax(align.view(batch*target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        if coverage is not None:
            batch_, source_l_ = coverage.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)
        if self.coverage_attn and coverage is not None:
            # [batch, src_len]
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            # [batch, src_len]
            coverage_reversed = -1 * coverage
            coverage_reversed.masked_fill_(1 - mask, -float('inf'))
            coverage_reversed = F.softmax(coverage_reversed, -1)
            coverage_reversed = coverage_reversed.unsqueeze(1)
            # we only use the coverage_reversed to rescale the current sent attention and do not backward the gradient
            coverage_reversed = coverage_reversed.detach()

            align_vectors = align_vectors * coverage_reversed
            norm_term = align_vectors.sum(dim=2, keepdim=True)
            align_vectors = align_vectors / norm_term

        # each context vector c_t is the weighted average
        # over all the source hidden states
        # [batch, target_l, dim]
        c = torch.bmm(align_vectors, memory_bank)
        # [batch, dim]
        returned_vec = c.squeeze(1)

        # # concatenate
        if self.output_attn_h:
            concat_c = torch.cat([c, source], 2).view(batch*target_l, dim*2)
            attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
            if self.attn_type in ["general", "dot"]:
                attn_h = torch.tanh(attn_h)
            attn_h = attn_h.squeeze(1)
            returned_vec = attn_h

        align_vectors = align_vectors.squeeze(1)
        # Check output sizes
        batch_, dim_ = returned_vec.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        # Check output sizes
        batch_, source_l_ = align_vectors.size()
        aeq(batch, batch_)
        aeq(source_l, source_l_)

        return returned_vec, align_vectors


class SeqHREWordGlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    All models compute the output as
    :math:`c = sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """

    def __init__(self, dim, coverage=False, attn_type="dot",
                 attn_func="softmax", output_attn_h=False, seqHRE_attn_rescale=False):
        super(SeqHREWordGlobalAttention, self).__init__()

        self.seqHRE_attn_rescale = seqHRE_attn_rescale
        self.dim = dim
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type.")
        self.attn_type = attn_type
        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function.")
        self.attn_func = attn_func

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)

        self.output_attn_h = output_attn_h
        if output_attn_h:
            # mlp wants it with bias
            out_bias = self.attn_type == "mlp"
            self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`
          attn_level ('str'): the string indicator of current attention level

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """
        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, tgt_len, d) x (batch, d, src_len) --> (batch, tgt_len, src_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, source, memory_bank, memory_lengths=None, coverage=None,
                sent_align_vectors=None, sent_position_tuple=None, src_word_sent_ids=None):
        """
        Only one-step attention is supported now.
        Args:
          source (`FloatTensor`): query vectors `[batch x dim]`
          memory_bank (`FloatTensor`): word_memory_bank is `FloatTensor` with shape `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): for word_memory_bank, `[batch]`
          coverage (`FloatTensor`): None (not supported yet)
          sent_align_vectors (`FloatTensor`): the computed sentence align distribution, `[batch x s_num]`
          sent_position_tuple (:obj: `tuple`): Only used for seqhr_enc (sent_p, sent_nums) with size
                `([batch_size, s_num, 2], [batch])`.
          src_word_sent_ids (:obj: `tuple'): (word_sent_ids, src_lengths) with size `([batch, src_len], [batch])'
          use_tanh (`bool`): True, whether use tanh activation function for `general` and 'dot' attention

        Returns:
          (`FloatTensor`, `FloatTensor`):
            * Computed word attentional vector `[batch x dim]`
            * Word Attention distribtutions for the query of word `[batch x src_len]`
        """

        # only one step input is supported
        assert source.dim() == 2, "Only one step input is supported for current attention."
        assert isinstance(sent_position_tuple, tuple)
        sent_position, sent_nums = sent_position_tuple
        one_step = True
        # [batch, 1, dim]
        source = source.unsqueeze(1)
        batch, tgt_l, dim = source.size()

        # check the specification for word level attention
        assert sent_align_vectors is not None, "For word level attention, the 'sent_align' must be specified."
        assert sent_position is not None, "For word level attention, the 'sent_position' must be specified."
        assert sent_nums is not None, "For word level attention, the 'sent_nums' must be specified."
        assert memory_lengths is not None, "The lengths for the word memory bank are required."
        sent_lens = memory_lengths

        batch_1, src_len, dim_ = memory_bank.size()
        batch_2, sent_num = sent_align_vectors.size()
        batch_3 = sent_nums.size(0)

        aeq(batch, batch_1, batch_2, batch_3)
        aeq(dim, dim_, self.dim)

        # if coverage is not None:
        #     batch_, source_l_ = coverage.size()
        #     aeq(batch, batch_)
        #     aeq(source_l, source_l_)
        #
        # if coverage is not None:
        #     cover = coverage.view(-1).unsqueeze(1)
        #     memory_bank += self.linear_cover(cover).view_as(memory_bank)
        #     memory_bank = torch.tanh(memory_bank)

        # compute word attention scores, as in Luong et al.
        # [batch, 1, src_len]
        word_align = self.score(source, memory_bank)
        # [batch, src_len]
        word_align = word_align.squeeze(1)

        # [batch, src_len]
        word_mask = sequence_mask(memory_lengths, max_len=word_align.size(-1))

        # [batch, src_len]
        word_align.masked_fill_(1 - word_mask, -float('inf'))

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            align_vectors = F.softmax(word_align, -1)
        else:
            align_vectors = sparsemax(word_align, -1)

        if self.seqHRE_attn_rescale:
            word_sent_ids, memory_lengths_ = src_word_sent_ids
            assert memory_lengths.eq(memory_lengths_).all(), \
                "The src lengths in src_word_sent_ids should be the same as the memory_lengths"

            # # attention score reweighting method 1
            # # broadcast the sent_align_vectors from [batch, sent_num] to [batch, src_len]
            # # according to sent_position [batch, sent_num, 2] and sent_nums [batch]
            # expand_sent_align_vectors = []
            # for b_idx in range(batch):
            #     one_ex_expand = []
            #     for sent_idx in range(sent_num):
            #         sent_token_num = sent_position[b_idx][sent_idx][0] - sent_position[b_idx][sent_idx][1] + 1
            #         if sent_token_num != 1:
            #             one_ex_expand.append(sent_align_vectors[b_idx][sent_idx].expand(sent_token_num))
            #         else:
            #             break
            #     one_ex_expand = torch.cat(one_ex_expand, dim=0)
            #     if one_ex_expand.size(0) < src_len:
            #         pad_vector = torch.zeros([src_len - one_ex_expand.size(0)],
            #                                  dtype=one_ex_expand.dtype, device=one_ex_expand.device)
            #         one_ex_expand = torch.cat([one_ex_expand, pad_vector], dim=0).contiguous()
            #     expand_sent_align_vectors.append(one_ex_expand)
            #
            # # [batch, src_len]
            # expand_sent_align_vectors = torch.stack(expand_sent_align_vectors, dim=0).contiguous()
            # # reweight and renormalize the word align_vectors
            # align_vectors = align_vectors * expand_sent_align_vectors
            # norm_term = align_vectors.sum(dim=1, keepdim=True)
            # align_vectors = align_vectors / norm_term

            # attention score reweighting method 2
            # word_sent_ids: [batch, src_len]
            # sent_align_vectors: [batch, sent_num]
            # expand_sent_align_vectors: [batch, src_len]
            expand_sent_align_vectors = sent_align_vectors.gather(dim=1, index=word_sent_ids)
            # # reweight and renormalize the word align_vectors
            # Although word_sent_ids are padded with 0s which will gather the attention score of the sentence 0
            # align_vectors are 0.0000 on these padded places.
            align_vectors = align_vectors * expand_sent_align_vectors
            norm_term = align_vectors.sum(dim=1, keepdim=True)
            align_vectors = align_vectors / norm_term


        # each context vector c_t is the weighted average
        # over all the source hidden states
        # [batch, 1, src_len]
        align_vectors = align_vectors.unsqueeze(1)
        # [batch, 1, src_len] x [batch, src_len, dim] -> [batch, 1, dim]
        c = torch.bmm(align_vectors, memory_bank)
        # [batch, dim]
        c = c.squeeze(1)
        returned_vec = c
        # If output_attn_h == False, we put linear out layer on decoder part
        if self.output_attn_h:
            # concatenate
            # [batch, dim]
            source = source.squeeze(1)
            # [batch, 2*dim]
            concat_c = torch.cat([c, source], 1)
            # [batch, dim]
            attn_h = self.linear_out(concat_c)
            if self.attn_type in ["general", "dot"]:
                attn_h = torch.tanh(attn_h)
            returned_vec = attn_h

        # [batch, src_len]
        align_vectors = align_vectors.squeeze(1)
        # Check output sizes
        batch_1, dim_ = returned_vec.size()
        batch_2, _ = align_vectors.size()
        aeq(batch, batch_1, batch_2)
        aeq(dim, dim_)

        return returned_vec, align_vectors


class W2WordGlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.

    All models compute the output as
    :math:`c = sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """

    def __init__(self, dim, coverage=False, attn_type="dot",
                 attn_func="softmax", output_attn_h=True):
        super(W2WordGlobalAttention, self).__init__()

        self.dim = dim
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type.")
        self.attn_type = attn_type
        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function.")
        self.attn_func = attn_func

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)

        # # wchen: we put the linear_out layer into the decoder module
        # mlp wants it with bias
        self.output_attn_h = output_attn_h
        if output_attn_h:
            out_bias = self.attn_type == "mlp"
            self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, source, memory_bank, memory_lengths=None, coverage=None, sent_align_vectors=None):
        """

        Args:
          source (`FloatTensor`): query vectors `[batch x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)
          sent_align_vectors (`FloatTensor`): sentence level attention cores `[batch x src_len]`

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[batch x dim]`
          * Attention distribtutions for each query
             `[batch x src_len]`
        """

        # one step input
        assert source.dim() == 2, "Only one step input is supported"
        #one_step = True
        source = source.unsqueeze(1)
        sent_align_vectors = sent_align_vectors.unsqueeze(1)

        batch, src_len, dim = memory_bank.size()
        batch1, tgt_len, dim1 = source.size()
        batch2, tgt_len2, src_len2 = sent_align_vectors.size()
        aeq(batch, batch1, batch2)
        aeq(self.dim, dim, dim1)
        aeq(src_len, src_len2)
        aeq(tgt_len, tgt_len2)

        # if coverage is not None:
        #     batch_, source_l_ = coverage.size()
        #     aeq(batch, batch_)
        #     aeq(source_l, source_l_)
        # if coverage is not None:
        #     cover = coverage.view(-1).unsqueeze(1)
        #     memory_bank += self.linear_cover(cover).view_as(memory_bank)
        #     memory_bank = torch.tanh(memory_bank)

        # compute attention scores, as in Luong et al.
        # [batch, tgt_len, src_len]
        align = self.score(source, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(1 - mask, -float('inf'))

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            align_vectors = F.softmax(align.view(batch*tgt_len, src_len), -1)
        else:
            align_vectors = sparsemax(align.view(batch*tgt_len, src_len), -1)
        align_vectors = align_vectors.view(batch, tgt_len, src_len)

        # rescale the word attention scores using the sent_align_vec
        align_vectors = align_vectors * sent_align_vectors
        norm_vec = align_vectors.sum(dim=-1, keepdim=True)
        align_vectors = align_vectors / norm_vec

        # each context vector c_t is the weighted average
        # over all the source hidden states
        # [batch, tgt_len, dim]
        c = torch.bmm(align_vectors, memory_bank)
        # [batch, dim]
        returned_vec = c.squeeze(1)

        # # concatenate
        if self.output_attn_h:
            concat_c = torch.cat([c, source], 2).view(batch*tgt_len, dim*2)
            attn_h = self.linear_out(concat_c).view(batch, tgt_len, dim)
            if self.attn_type in ["general", "dot"]:
                attn_h = torch.tanh(attn_h)
            attn_h = attn_h.squeeze(1)
            returned_vec = attn_h

        align_vectors = align_vectors.squeeze(1)
        # Check output sizes
        batch_, dim_ = returned_vec.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        # Check output sizes
        batch_, src_len_ = align_vectors.size()
        aeq(batch, batch_)
        aeq(src_len, src_len_)

        return returned_vec, align_vectors


class TargetEncGlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """

    def __init__(self, tgt_enc_dim, src_enc_dim, coverage=False, attn_type="general", attn_func="softmax"):
        super(TargetEncGlobalAttention, self).__init__()
        self.tgt_enc_dim = tgt_enc_dim
        self.src_enc_dim = src_enc_dim
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type.")
        self.attn_type = attn_type
        assert attn_func in ["softmax", "sparsemax"], (
            "Please select a valid attention function.")
        self.attn_func = attn_func

        if self.attn_type == "general":
            self.linear_in = nn.Linear(tgt_enc_dim, src_enc_dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(src_enc_dim, src_enc_dim, bias=False)
            self.linear_query = nn.Linear(tgt_enc_dim, src_enc_dim, bias=True)
            self.v = nn.Linear(src_enc_dim, 1, bias=False)
        # # mlp wants it with bias
        # out_bias = self.attn_type == "mlp"
        # self.linear_out = nn.Linear(tgt_enc_dim + src_enc_dim, dim, bias=out_bias)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x tgt_enc_dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x src_enc_dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, self.src_enc_dim)
        aeq(self.tgt_enc_dim, tgt_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, src_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, source, memory_bank, memory_lengths=None):
        """

        Args:
          source (`FloatTensor`): query vectors `[batch x tgt_len x tgt_enc_dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x src_enc_dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`

        Returns:
          (`FloatTensor`):

          * Attention distribtutions for each query
             `[batch x tgt_len x src_len]`
        """

        # one step input
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        else:
            one_step = False

        batch, source_l, src_enc_dim = memory_bank.size()
        batch_, target_l, tgt_enc_dim = source.size()
        aeq(batch, batch_)
        aeq(self.src_enc_dim, src_enc_dim)
        aeq(self.tgt_enc_dim, tgt_enc_dim)

        # compute attention scores, as in Luong et al.
        # (batch, t_len, s_len)
        align = self.score(source, memory_bank)

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths, max_len=align.size(-1))
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.masked_fill_(1 - mask, -float('inf'))

        # Softmax or sparsemax to normalize attention weights
        if self.attn_func == "softmax":
            align_vectors = F.softmax(align.view(batch*target_l, source_l), -1)
        else:
            align_vectors = sparsemax(align.view(batch*target_l, source_l), -1)
        align_vectors = align_vectors.view(batch, target_l, source_l)

        # # each context vector c_t is the weighted average
        # # over all the source hidden states
        # c = torch.bmm(align_vectors, memory_bank)
        #
        # # concatenate
        # concat_c = torch.cat([c, source], 2).view(batch*target_l, dim*2)
        # attn_h = self.linear_out(concat_c).view(batch, target_l, dim)
        # if self.attn_type in ["general", "dot"]:
        #     attn_h = torch.tanh(attn_h)

        if one_step:
            # attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            # batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            # aeq(dim, dim_)
            batch_, source_l_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        else:
            # attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.contiguous()
            # Check output sizes
            # target_l_, batch_, dim_ = attn_h.size()
            # aeq(target_l, target_l_)
            aeq(batch, batch_)
            # aeq(dim, dim_)
            batch_, target_l_, source_l_ = align_vectors.size()
            aeq(target_l, target_l_)
            aeq(batch, batch_)
            aeq(source_l, source_l_)

        return align_vectors