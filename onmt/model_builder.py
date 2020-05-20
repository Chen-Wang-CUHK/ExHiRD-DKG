"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""
import re
import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import xavier_uniform_

import onmt.inputters as inputters
import onmt.modules
from onmt.encoders.rnn_encoder import RNNEncoder, HREncoder, SeqHREncoder, TGEncoder
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder
from onmt.encoders.audio_encoder import AudioEncoder
from onmt.encoders.image_encoder import ImageEncoder

from onmt.decoders.decoder import InputFeedRNNDecoder, StdRNNDecoder, CatSeqDInputFeedRNNDecoder, CatSeqCorrInputFeedRNNDecoder
from onmt.decoders.decoder import HREInputFeedRNNDecoder, HRDInputFeedRNNDecoder, SeqHREInputFeedRNNDecoder
from onmt.decoders.transformer import TransformerDecoder
from onmt.decoders.cnn_decoder import CNNDecoder

from onmt.modules import Embeddings, CopyGenerator
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger

# add by wchen
from data_utils import KEY_SEPERATOR, EOKP_TOKEN, P_END, A_END

def build_embeddings(opt, word_field, feat_fields, for_encoder=True):
    """
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    emb_dim = opt.src_word_vec_size if for_encoder else opt.tgt_word_vec_size

    word_padding_idx = word_field.vocab.stoi[word_field.pad_token]
    num_word_embeddings = len(word_field.vocab)

    feat_pad_indices = [ff.vocab.stoi[ff.pad_token] for ff in feat_fields]
    num_feat_embeddings = [len(ff.vocab) for ff in feat_fields]

    emb = Embeddings(
        word_vec_size=emb_dim,
        position_encoding=opt.position_encoding,
        feat_merge=opt.feat_merge,
        feat_vec_exponent=opt.feat_vec_exponent,
        feat_vec_size=opt.feat_vec_size,
        dropout=opt.dropout,
        word_padding_idx=word_padding_idx,
        feat_padding_idx=feat_pad_indices,
        word_vocab_size=num_word_embeddings,
        feat_vocab_sizes=num_feat_embeddings,
        sparse=opt.optim == "sparseadam"
    )
    return emb


def build_position_encoding(dim, max_length=20, learned_position_enc=False):
    """
    Init the sinusoid position encoding table
    Some codes are borrowed from https://github.com/dmlc/gluon-nlp/blob/master/src/gluonnlp/model/transformer.py#L44
    """
    if learned_position_enc:
        position_enc = nn.Embedding(max_length, dim)
    else:
        position_enc = np.arange(max_length).reshape((-1, 1)) / (np.power(10000, (2. / dim) * np.arange(dim).reshape((1, -1))))
        # Apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        position_enc = torch.from_numpy(position_enc).float().to(device)

        position_enc = nn.Embedding.from_pretrained(position_enc, freeze=True)
    return position_enc


def build_encoder(opt, embeddings):
    """
    Various encoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this encoder.
    """
    if opt.encoder_type == "transformer":
        encoder = TransformerEncoder(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout,
            embeddings)
    elif opt.encoder_type == "cnn":
        encoder = CNNEncoder(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.cnn_kernel_width,
            opt.dropout,
            embeddings)
    elif opt.encoder_type == "mean":
        encoder = MeanEncoder(opt.enc_layers, embeddings)
    elif opt.encoder_type == "hr_brnn":
        bi_enc = True
        encoder = HREncoder(
            opt.rnn_type,
            bi_enc,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout,
            embeddings,
            opt.bridge)
    elif opt.encoder_type == "seq_hr_brnn":
        bi_enc = True
        encoder = SeqHREncoder(
            opt.rnn_type,
            bi_enc,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout,
            embeddings,
            opt.bridge)
    elif opt.encoder_type == "tg_brnn":
        bi_enc = True
        encoder = TGEncoder(opt.rnn_type,
                            bi_enc,
                            opt.enc_layers,
                            opt.enc_rnn_size,
                            opt.dropout,
                            embeddings)
    else:
        bi_enc = 'brnn' in opt.encoder_type
        encoder = RNNEncoder(
            opt.rnn_type,
            bi_enc,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout,
            embeddings,
            opt.bridge,
            use_catSeq_dp=opt.use_catSeq_dp)
    return encoder


def build_decoder(opt, embeddings,
                  eok_idx=None, eos_idx=None, pad_idx=None, sep_idx=None, p_end_idx=None, a_end_idx=None,
                  position_enc=None, position_enc_embsize=None):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    if opt.decoder_type == "transformer":
        decoder = TransformerDecoder(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.global_attention,
            opt.copy_attn,
            opt.self_attn_type,
            opt.dropout,
            embeddings
        )
    elif opt.decoder_type == "cnn":
        decoder = CNNDecoder(
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.global_attention,
            opt.copy_attn,
            opt.cnn_kernel_width,
            opt.dropout,
            embeddings
        )
    elif opt.decoder_type == "hre_rnn":
        assert opt.input_feed
        bi_enc = 'brnn' in opt.encoder_type
        dec_class = HREInputFeedRNNDecoder
        decoder = dec_class(
            opt.rnn_type,
            bi_enc,
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.global_attention,
            opt.global_attention_function,
            opt.coverage_attn,
            opt.context_gate,
            opt.copy_attn,
            opt.dropout,
            embeddings,
            opt.reuse_copy_attn,
            opt.hr_attn_type
        )
    elif opt.decoder_type == "seq_hre_rnn":
        assert opt.input_feed
        bi_enc = 'brnn' in opt.encoder_type
        dec_class = SeqHREInputFeedRNNDecoder
        decoder = dec_class(
            opt.rnn_type,
            bi_enc,
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.global_attention,
            opt.global_attention_function,
            opt.coverage_attn,
            opt.context_gate,
            opt.copy_attn,
            opt.dropout,
            embeddings,
            opt.reuse_copy_attn,
            hr_attn_type=opt.hr_attn_type,
            seqHRE_attn_rescale=opt.seqHRE_attn_rescale
        )
    elif opt.decoder_type == "hrd_rnn" or opt.decoder_type == "seq_hre_hrd_rnn":
        assert opt.input_feed
        # assert eok_idx is not None
        assert eos_idx is not None
        assert pad_idx is not None
        bi_enc = 'brnn' in opt.encoder_type
        hr_enc = 'hr' in opt.encoder_type
        seqhr_enc = opt.encoder_type == "seq_hr_brnn"
        dec_class = HRDInputFeedRNNDecoder
        decoder = dec_class(
            opt.rnn_type,
            bi_enc,
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.global_attention,
            opt.global_attention_function,
            opt.coverage_attn,
            opt.context_gate,
            opt.copy_attn,
            opt.dropout,
            embeddings,
            opt.reuse_copy_attn,
            hr_attn_type=opt.hr_attn_type,
            word_dec_init_type=opt.word_dec_init_type,
            remove_input_feed_w=opt.remove_input_feed_w,
            input_feed_w_type=opt.sent_dec_input_feed_w_type,
            hr_enc=hr_enc,
            seqhr_enc=seqhr_enc,
            seqE_HRD_rescale_attn=opt.seqE_HRD_rescale_attn,
            seqHRE_attn_rescale=opt.seqHRE_attn_rescale,
            use_zero_s_emb=opt.use_zero_s_emb,
            not_detach_coverage=opt.not_detach_coverage,
            eok_idx=eok_idx,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            sep_idx=sep_idx,
            p_end_idx=p_end_idx,
            a_end_idx=a_end_idx,
            position_enc=position_enc,
            position_enc_word_init=opt.use_position_enc_word_init_state,
            position_enc_sent_feed_w=opt.use_position_enc_sent_input_feed_w,
            position_enc_first_word_feed=opt.use_position_enc_first_word_feed,
            position_enc_embsize=position_enc_embsize,
            position_enc_start_token=opt.use_opsition_enc_start_token,
            position_enc_sent_state=opt.use_position_enc_sent_state,
            position_enc_all_first_valid_word_dec_inputs=opt.use_position_enc_first_valid_word_dec_inputs,
            sent_dec_init_type=opt.sent_dec_init_type,
            remove_input_feed_h=opt.remove_input_feed_h,
            detach_input_feed_w=opt.detach_input_feed_w,
            use_target_encoder=opt.use_target_encoder,
            src_states_capacity=opt.src_states_capacity,
            src_states_sample_size=opt.src_states_sample_size
        )
    elif opt.decoder_type == "CatSeqD_rnn":
        dec_class = CatSeqDInputFeedRNNDecoder
        bi_enc = 'brnn' in opt.encoder_type
        decoder = dec_class(
            opt.rnn_type,
            bi_enc,
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.global_attention,
            opt.global_attention_function,
            opt.coverage_attn,
            opt.context_gate,
            opt.copy_attn,
            opt.dropout,
            embeddings,
            opt.reuse_copy_attn,
            sep_idx=sep_idx,
            use_target_encoder=opt.use_target_encoder,
            target_hidden_size=opt.target_hidden_size,
            src_states_capacity=opt.src_states_capacity,
            src_states_sample_size=opt.src_states_sample_size,
            use_catSeq_dp=opt.use_catSeq_dp
        )
    elif opt.decoder_type == "CatSeqCorr_rnn":
        assert opt.input_feed
        dec_class = CatSeqCorrInputFeedRNNDecoder
        bi_enc = 'brnn' in opt.encoder_type
        decoder = dec_class(
            opt.rnn_type,
            bi_enc,
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.global_attention,
            opt.global_attention_function,
            opt.coverage_attn,
            opt.context_gate,
            opt.copy_attn,
            opt.dropout,
            embeddings,
            opt.reuse_copy_attn
        )
    else:
        assert opt.input_feed
        dec_class = InputFeedRNNDecoder if opt.input_feed else StdRNNDecoder
        bi_enc = 'brnn' in opt.encoder_type
        decoder = dec_class(
            opt.rnn_type,
            bi_enc,
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.global_attention,
            opt.global_attention_function,
            opt.coverage_attn,
            opt.context_gate,
            opt.copy_attn,
            opt.dropout,
            embeddings,
            opt.reuse_copy_attn,
            use_catSeq_dp=opt.use_catSeq_dp
        )

    return decoder


def load_test_model(opt, dummy_opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)

    vocab = checkpoint['vocab']
    if inputters.old_style_vocab(vocab):
        fields = inputters.load_fields_from_vocab(vocab, opt.data_type)
    else:
        fields = vocab

    model_opt = checkpoint['opt']

    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]
    # changed to my_build_base_model by wchen
    if 'hr' in model_opt.encoder_type or 'hr' in model_opt.decoder_type or 'CatSeqD' in model_opt.decoder_type:
        model = my_build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    else:
        model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt


def build_base_model(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    assert model_opt.model_type in ["text", "img", "audio"], \
        "Unsupported model type %s" % model_opt.model_type

    # for backward compatibility
    if model_opt.rnn_size != -1:
        model_opt.enc_rnn_size = model_opt.rnn_size
        model_opt.dec_rnn_size = model_opt.rnn_size

    # Build encoder.
    if model_opt.model_type == "text":
        src_fields = [f for n, f in fields['src']]
        src_emb = build_embeddings(model_opt, src_fields[0], src_fields[1:])
        encoder = build_encoder(model_opt, src_emb)
    elif model_opt.model_type == "img":
        # why is build_encoder not used here?
        # why is the model_opt.__dict__ check necessary?
        if "image_channel_size" not in model_opt.__dict__:
            image_channel_size = 3
        else:
            image_channel_size = model_opt.image_channel_size

        encoder = ImageEncoder(
            model_opt.enc_layers,
            model_opt.brnn,
            model_opt.enc_rnn_size,
            model_opt.dropout,
            image_channel_size
        )
    elif model_opt.model_type == "audio":
        encoder = AudioEncoder(
            model_opt.rnn_type,
            model_opt.enc_layers,
            model_opt.dec_layers,
            model_opt.brnn,
            model_opt.enc_rnn_size,
            model_opt.dec_rnn_size,
            model_opt.audio_enc_pooling,
            model_opt.dropout,
            model_opt.sample_rate,
            model_opt.window_size
        )

    # Build decoder.
    tgt_fields = [f for n, f in fields['tgt']]
    tgt_emb = build_embeddings(
        model_opt, tgt_fields[0], tgt_fields[1:], for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        assert src_fields[0].vocab == tgt_fields[0].vocab, \
            "preprocess with -share_vocab if you use share_embeddings"

        tgt_emb.word_lut.weight = src_emb.word_lut.weight

    decoder = build_decoder(model_opt, tgt_emb)

    # Build NMTModel(= encoder + decoder).
    device = torch.device("cuda" if gpu else "cpu")
    model = onmt.models.NMTModel(encoder, decoder)

    # Build Generator.
    if not model_opt.copy_attn:
        if model_opt.generator_function == "sparsemax":
            gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
        else:
            gen_func = nn.LogSoftmax(dim=-1)
        generator = nn.Sequential(
            nn.Linear(model_opt.dec_rnn_size, len(fields["tgt"][0][1].vocab)),
            gen_func
        )
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        vocab_size = len(fields["tgt"][0][1].vocab)
        pad_idx = fields["tgt"][0][1].vocab.stoi[fields["tgt"][0][1].pad_token]
        generator = CopyGenerator(model_opt.dec_rnn_size, vocab_size, pad_idx)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        # This preserves backward-compat for models using customed layernorm
        def fix_key(s):
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                       r'\1.layer_norm\2.bias', s)
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                       r'\1.layer_norm\2.weight', s)
            return s

        checkpoint['model'] = {fix_key(k): v
                               for k, v in checkpoint['model'].items()}
        # end of patch for backward compatibility

        model.load_state_dict(checkpoint['model'], strict=False)
        generator.load_state_dict(checkpoint['generator'], strict=False)
    else:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    model.generator = generator
    model.to(device)

    return model


def my_build_base_model(model_opt, fields, gpu, checkpoint=None):
    """
    Revised from build_base_model by wchen
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the NMTModel.
    """
    assert model_opt.model_type in ["text"], \
        "Unsupported model type %s" % model_opt.model_type

    # for backward compatibility
    if model_opt.rnn_size != -1:
        model_opt.enc_rnn_size = model_opt.rnn_size
        model_opt.dec_rnn_size = model_opt.rnn_size

    # Build encoder.
    src_fields = [f for n, f in fields['src']]
    src_emb = build_embeddings(model_opt, src_fields[0], src_fields[1:])
    encoder = build_encoder(model_opt, src_emb)

    # Build decoder.
    tgt_fields = [f for n, f in fields['tgt']]
    tgt_emb = build_embeddings(
        model_opt, tgt_fields[0], tgt_fields[1:], for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        assert src_fields[0].vocab == tgt_fields[0].vocab, \
            "preprocess with -share_vocab if you use share_embeddings"

        tgt_emb.word_lut.weight = src_emb.word_lut.weight
    # obtain the end of keyphrase token index
    eok_idx = None
    if EOKP_TOKEN in tgt_fields[0].vocab.stoi:
        eok_idx = tgt_fields[0].vocab.stoi[EOKP_TOKEN]
    # else:
    #     eok_idx = tgt_fields[0].vocab.stoi[KEY_SEPERATOR]
    eos_idx = tgt_fields[0].vocab.stoi[tgt_fields[0].eos_token]
    pad_idx = tgt_fields[0].vocab.stoi[tgt_fields[0].pad_token]

    sep_idx = None
    p_end_idx = None
    a_end_idx = None
    if KEY_SEPERATOR in tgt_fields[0].vocab.stoi:
        sep_idx = tgt_fields[0].vocab.stoi[KEY_SEPERATOR]
    else:
        p_end_idx = tgt_fields[0].vocab.stoi[P_END]
        a_end_idx = tgt_fields[0].vocab.stoi[A_END]

    if model_opt.use_position_enc_sent_input_feed_w or \
            model_opt.use_position_enc_word_init_state or \
            model_opt.use_position_enc_first_word_feed or \
            model_opt.use_position_enc_sent_state or \
            model_opt.use_position_enc_first_valid_word_dec_inputs:
        position_enc = build_position_encoding(dim=model_opt.dec_rnn_size,
                                               learned_position_enc=model_opt.learned_position_enc)
    else:
        position_enc = None

    if model_opt.use_opsition_enc_start_token:
        position_enc_embsize = build_position_encoding(dim=model_opt.word_vec_size,
                                                       learned_position_enc=model_opt.learned_position_enc)
    else:
        position_enc_embsize = None

    decoder = build_decoder(model_opt, tgt_emb,
                            eok_idx, eos_idx, pad_idx, sep_idx, p_end_idx, a_end_idx,
                            position_enc, position_enc_embsize)

    # Build NMTModel(= encoder + decoder).
    device = torch.device("cuda" if gpu else "cpu")
    if 'hr' not in model_opt.encoder_type and 'hr' not in model_opt.decoder_type:
        if model_opt.encoder_type != 'tg_brnn':
            model = onmt.models.NMTModel(encoder, decoder)
        else:
            model = onmt.models.TGModel(encoder, decoder)
    elif model_opt.encoder_type == 'seq_hr_brnn':
        model = onmt.models.SeqHREDModel(encoder, decoder)
    else:
        model = onmt.models.HREDModel(encoder, decoder)

    # Build Generator.
    if not model_opt.copy_attn:
        if model_opt.generator_function == "sparsemax":
            gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
        else:
            gen_func = nn.LogSoftmax(dim=-1)
        generator = nn.Sequential(
            nn.Linear(model_opt.dec_rnn_size, len(fields["tgt"][0][1].vocab)),
            gen_func
        )
        if model_opt.share_decoder_embeddings:
            generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        vocab_size = len(fields["tgt"][0][1].vocab)
        pad_idx = fields["tgt"][0][1].vocab.stoi[fields["tgt"][0][1].pad_token]
        generator = CopyGenerator(model_opt.dec_rnn_size, vocab_size, pad_idx)

    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        # This preserves backward-compat for models using customed layernorm
        def fix_key(s):
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                       r'\1.layer_norm\2.bias', s)
            s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                       r'\1.layer_norm\2.weight', s)
            return s

        checkpoint['model'] = {fix_key(k): v
                               for k, v in checkpoint['model'].items()}
        # end of patch for backward compatibility

        model.load_state_dict(checkpoint['model'], strict=False)
        generator.load_state_dict(checkpoint['generator'], strict=False)
    else:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                if p.requires_grad:
                    p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    model.generator = generator
    model.to(device)

    return model


def build_model(model_opt, opt, fields, checkpoint):
    logger.info('Building model...')
    # original
    # model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    model = my_build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    logger.info(model)
    return model
