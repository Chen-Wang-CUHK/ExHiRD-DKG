""" Implementation of all available options """
from __future__ import print_function

import configargparse
from onmt.models.sru import CheckSRU


def config_opts(parser):
    parser.add('-config', '--config', required=False,
               is_config_file_arg=True, help='config file path')
    parser.add('-save_config', '--save_config', required=False,
               is_write_out_config_file_arg=True,
               help='config file save path')


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """

    # Embedding Options
    group = parser.add_argument_group('Model-Embeddings')
    group.add('--src_word_vec_size', '-src_word_vec_size',
              type=int, default=500,
              help='Word embedding size for src.')
    group.add('--tgt_word_vec_size', '-tgt_word_vec_size',
              type=int, default=500,
              help='Word embedding size for tgt.')
    group.add('--word_vec_size', '-word_vec_size', type=int, default=-1,
              help='Word embedding size for src and tgt.')

    group.add('--share_decoder_embeddings', '-share_decoder_embeddings',
              action='store_true',
              help="""Use a shared weight matrix for the input and
                       output word  embeddings in the decoder.""")
    group.add('--share_embeddings', '-share_embeddings', action='store_true',
              help="""Share the word embeddings between encoder
                       and decoder. Need to use shared dictionary for this
                       option.""")
    group.add('--position_encoding', '-position_encoding', action='store_true',
              help="""Use a sin to mark relative words positions.
                       Necessary for non-RNN style models.
                       """)

    group = parser.add_argument_group('Model-Embedding Features')
    group.add('--feat_merge', '-feat_merge', type=str, default='concat',
              choices=['concat', 'sum', 'mlp'],
              help="""Merge action for incorporating features embeddings.
                       Options [concat|sum|mlp].""")
    group.add('--feat_vec_size', '-feat_vec_size', type=int, default=-1,
              help="""If specified, feature embedding sizes
                       will be set to this. Otherwise, feat_vec_exponent
                       will be used.""")
    group.add('--feat_vec_exponent', '-feat_vec_exponent',
              type=float, default=0.7,
              help="""If -feat_merge_size is not set, feature
                       embedding sizes will be set to N^feat_vec_exponent
                       where N is the number of values the feature takes.""")

    # Encoder-Decoder Options
    group = parser.add_argument_group('Model- Encoder-Decoder')
    group.add('--model_type', '-model_type', default='text',
              help="""Type of source model to use. Allows
                       the system to incorporate non-text inputs.
                       Options are [text|img|audio].""")
    # add hr_brnn by wchen
    group.add('--encoder_type', '-encoder_type', type=str, default='rnn',
              choices=['rnn', 'brnn', 'mean', 'transformer', 'cnn', 'hr_brnn', 'seq_hr_brnn', 'tg_brnn'],
              help="""Type of encoder layer to use. Non-RNN layers
                       are experimental. Options are
                       [rnn|brnn|mean|transformer|cnn|hr_brnn|seq_hr_brnn|tg_brnn].""")
    group.add('--decoder_type', '-decoder_type', type=str, default='rnn',
              choices=['rnn', 'transformer', 'cnn', 'hre_rnn', 'seq_hre_rnn', 'hrd_rnn', 'seq_hre_hrd_rnn', 'CatSeqD_rnn', 'CatSeqCorr_rnn'],
              help="""Type of decoder layer to use. Non-RNN layers
                       are experimental. Options are
                       [rnn|transformer|cnn|hre_rnn|hrd_rnn|seq_hre_rnn|seq_hre_hrd_rnn|CatSeqD_rnn|CatSeqCorr_rnn].""")

    group.add('--use_catSeq_dp', '-use_catSeq_dp', action="store_true",
              help='Use dropout for the embedding vector and the output of the encoder when enc_layer==1')

    # add by wchen for target encoding of CatSeqD decoder
    group.add('--use_target_encoder', '-use_target_encoder', action="store_true",
              help='Use a target encoder layer for CatSeqD decoder.')
    group.add('--target_hidden_size', '-target_hidden_size', type=int, default=64,
              help='The hidden size of the target encoder')
    group.add('--src_states_capacity', '-src_states_capacity', type=int, default=128,
              help='The src states capacity of the target encoder')
    group.add('--src_states_sample_size', '-src_states_sample_size', type=int, default=32,
              help='The src states sample size of the target encoder')

    group.add('--hr_attn_type', '-hr_attn_type', type=str, default='sent_word_both',
              choices=['sent_only', 'word_only', 'sent_word_both'],
              help="""Type of hierarchical attention. Options are [sent_only|word_only|sent_word_both]""")
    group.add('--seqHRE_attn_rescale', '-seqHRE_attn_rescale', action="store_true",
              help='Whether to recale the word level attention using the sentence level attention for the seqHRE type models.')
    group.add('--seqE_HRD_rescale_attn', '-seqE_HRD_rescale_attn', action="store_true",
              help='Whether to recale the word level attention using the sentence level attention for the seqE_HRD type models.')
    group.add('--word_dec_init_type', '-word_dec_init_type', type=str, default='attn_vec',
              choices=['attn_vec', 'hidden_vec'],
              help="""Type of word decoder initialization type. Options are [attn_vec|hidden_vec]""")
    group.add('--sent_dec_input_feed_w_type', '-sent_dec_input_feed_w_type', type=str, default='attn_vec',
              choices=['attn_vec', 'hidden_vec', 'sec_attn_vec', 'sec_hidden_vec'],
              help="""Type of sent decoder input_feed_w type. Options are [attn_vec|hidden_vec]""")
    group.add('--sent_dec_init_type', '-sent_dec_init_type', type=str, default='enc_first',
              choices=['enc_first', 'enc_last', 'zero', 'enc_mean'],
              help="""Type of the initialization of the sent decoder. Options are [enc_first|enc_last|zero]""")
    # without_input_feed_w
    group.add('--detach_input_feed_w', '-detach_input_feed_w', action="store_true",
              help='Whether to detach the input_feed_w for the sent level decoder.')
    group.add('--remove_input_feed_w', '-remove_input_feed_w', action="store_true",
              help='Whether to remove the input_feed_w for the sent level decoder.')
    group.add('--remove_input_feed_h', '-remove_input_feed_h', action="store_true",
              help='Whether to remove the input_feed_h for the sent level decoder.')
    group.add('--use_zero_s_emb', '-use_zero_s_emb', action="store_true",
              help='Whether to use zero vector as the start token embedding.')
    # for positional encoding
    group.add('--learned_position_enc', '-learned_position_enc', action="store_true",
              help='Whether to use a learned position embedding.')
    group.add('--use_position_enc_sent_input_feed_w', '-use_position_enc_sent_input_feed_w', action="store_true",
              help='Whether to use position encoding in the ending state of the last word-level decoding process.')
    group.add('--use_position_enc_word_init_state', '-use_position_enc_word_init_state', action="store_true",
              help='Whether to use position encoding in the initial state of the word-level decoding process.')
    group.add('--use_position_enc_sent_state', '-use_position_enc_sent_state', action="store_true",
              help='Whether to use position encoding in the states of the sent-level decoding process.')
    group.add('--use_position_enc_first_word_feed', '-use_position_enc_first_word_feed', action="store_true",
              help='Whether to use position encoding in the first word input feed of the word-level decoding process.')
    group.add('--use_opsition_enc_start_token', '-use_opsition_enc_start_token', action="store_true",
              help='Whether to use position encoding in the start token embedding of the word-level decoding process.')
    group.add('--use_position_enc_first_valid_word_dec_inputs', '-use_position_enc_first_valid_word_dec_inputs',
              action="store_true",
              help='Whether to use position encoding in all the inputs of the first valid word decoding step.')


    # add by wchen
    # hre_attn_type
    # group.add('--hre_aggregation_type', '-hre_aggregation_type', type=str, default='sent_word',
    #           choices=['sent_word', 'sent', 'word'],
    #           help="""Type of decoder layer to aggregate attentional vector. Options are
    #                 [sent_word|sent|word].""")

    group.add('--layers', '-layers', type=int, default=-1,
              help='Number of layers in enc/dec.')
    group.add('--enc_layers', '-enc_layers', type=int, default=2,
              help='Number of layers in the encoder')
    group.add('--dec_layers', '-dec_layers', type=int, default=2,
              help='Number of layers in the decoder')
    group.add('--rnn_size', '-rnn_size', type=int, default=-1,
              help="""Size of rnn hidden states. Overwrites
                       enc_rnn_size and dec_rnn_size""")
    group.add('--enc_rnn_size', '-enc_rnn_size', type=int, default=500,
              help="""Size of encoder rnn hidden states.
                       Must be equal to dec_rnn_size except for
                       speech-to-text.""")
    group.add('--dec_rnn_size', '-dec_rnn_size', type=int, default=500,
              help="""Size of decoder rnn hidden states.
                       Must be equal to enc_rnn_size except for
                       speech-to-text.""")
    group.add('--audio_enc_pooling', '-audio_enc_pooling',
              type=str, default='1',
              help="""The amount of pooling of audio encoder,
                       either the same amount of pooling across all layers
                       indicated by a single number, or different amounts of
                       pooling per layer separated by comma.""")
    group.add('--cnn_kernel_width', '-cnn_kernel_width', type=int, default=3,
              help="""Size of windows in the cnn, the kernel_size is
                       (cnn_kernel_width, 1) in conv layer""")

    group.add('--input_feed', '-input_feed', type=int, default=1,
              help="""Feed the context vector at each time step as
                       additional input (via concatenation with the word
                       embeddings) to the decoder.""")
    group.add('--bridge', '-bridge', action="store_true",
              help="""Have an additional layer between the last encoder
                       state and the first decoder state""")
    group.add('--rnn_type', '-rnn_type', type=str, default='LSTM',
              choices=['LSTM', 'GRU', 'SRU'],
              action=CheckSRU,
              help="""The gate type to use in the RNNs""")
    # group.add('--residual', '-residual',   action="store_true",
    #                     help="Add residual connections between RNN layers.")

    group.add('--brnn', '-brnn', action=DeprecateAction,
              help="Deprecated, use `encoder_type`.")

    group.add('--context_gate', '-context_gate', type=str, default=None,
              choices=['source', 'target', 'both'],
              help="""Type of context gate to use.
                       Do not select for no context gate.""")

    # Attention options
    group = parser.add_argument_group('Model- Attention')
    group.add('--global_attention', '-global_attention',
              type=str, default='general', choices=['dot', 'general', 'mlp'],
              help="""The attention type to use:
                       dotprod or general (Luong) or MLP (Bahdanau)""")
    group.add('--global_attention_function', '-global_attention_function',
              type=str, default="softmax", choices=["softmax", "sparsemax"])
    group.add('--self_attn_type', '-self_attn_type',
              type=str, default="scaled-dot",
              help="""Self attention type in Transformer decoder
                       layer -- currently "scaled-dot" or "average" """)
    group.add('--heads', '-heads', type=int, default=8,
              help='Number of heads for transformer self-attention')
    group.add('--transformer_ff', '-transformer_ff', type=int, default=2048,
              help='Size of hidden transformer feed-forward')

    # Generator and loss options.
    group.add('--copy_attn', '-copy_attn', action="store_true",
              help='Train copy attention layer.')
    group.add('--generator_function', '-generator_function', default="softmax",
              choices=["softmax", "sparsemax"],
              help="""Which function to use for generating
              probabilities over the target vocabulary (choices:
              softmax, sparsemax)""")
    group.add('--copy_attn_force', '-copy_attn_force', action="store_true",
              help='When available, train to copy.')
    group.add('--reuse_copy_attn', '-reuse_copy_attn', action="store_true",
              help="Reuse standard attention for copy")
    group.add('--copy_loss_by_seqlength', '-copy_loss_by_seqlength',
              action="store_true",
              help="Divide copy loss by length of sequence")
    group.add('--coverage_attn', '-coverage_attn', action="store_true",
              help='Train a coverage attention layer.')
    group.add('--not_detach_coverage', '-not_detach_coverage', action="store_true",
              help='Do not detach the coverage vector when training with coverage loss.')


def preprocess_opts(parser):
    """ Pre-procesing options """
    # Data options add by wchen, For hierarchical encoder and decoder
    group = parser.add_argument_group('HRED')
    group.add('--seq_hr_src', '-seq_hr_src', action='store_true',
              help="Whether to preprocess the src input into a "
                   "sequential context + sentence forward position + sentence backward position format.")
    group.add('--hr_src', '-hr_src', action='store_true',
              help="Whether to preprocess the src input into a hierarchical format.")
    group.add('--hr_tgt', '-hr_tgt', action='store_true',
              help="Whether to preprocess the tgt input into a hierarchical format.")

    group.add('--contain_title', '-contain_title', action='store_true',
              help="Whether to contain the title as an extra query input.")

    group.add('--use_bi_end', '-use_bi_end', action='store_true',
              help="Whether to preprocess the tgt input into a format with two ending tokens.")
    group.add('--use_existing_vocab', '-use_existing_vocab', action='store_true',
              help="Whether to vocab building process and use the existing vocab.")

    # Data options
    group = parser.add_argument_group('Data')
    group.add('--data_type', '-data_type', default="text",
              help="""Type of the source input.
                       Options are [text|img].""")

    group.add('--train_src', '-train_src', required=True,
              help="Path to the training source data")
    group.add('--train_tgt', '-train_tgt', required=True,
              help="Path to the training target data")
    group.add('--valid_src', '-valid_src', required=True,
              help="Path to the validation source data")
    group.add('--valid_tgt', '-valid_tgt', required=True,
              help="Path to the validation target data")

    # add by wchen
    group.add('--train_title', '-train_title', default='',
              help="Path to the training source data's title")
    group.add('--valid_title', '-valid_title', default='',
              help="Path to the validation source data's title")

    group.add('--src_dir', '-src_dir', default="",
              help="Source directory for image or audio files.")

    group.add('--save_data', '-save_data', required=True,
              help="Output file for the prepared data")

    group.add('--max_shard_size', '-max_shard_size', type=int, default=0,
              help="""Deprecated use shard_size instead""")

    group.add('--shard_size', '-shard_size', type=int, default=1000000,
              help="""Divide src_corpus and tgt_corpus into
                       smaller multiple src_copus and tgt corpus files, then
                       build shards, each shard will have
                       opt.shard_size samples except last shard.
                       shard_size=0 means no segmentation
                       shard_size>0 means segment dataset into multiple shards,
                       each shard has shard_size samples""")

    # Dictionary options, for text corpus

    group = parser.add_argument_group('Vocab')
    group.add('--src_vocab', '-src_vocab', default="",
              help="""Path to an existing source vocabulary. Format:
                       one word per line.""")
    group.add('--tgt_vocab', '-tgt_vocab', default="",
              help="""Path to an existing target vocabulary. Format:
                       one word per line.""")
    group.add('--features_vocabs_prefix', '-features_vocabs_prefix',
              type=str, default='',
              help="Path prefix to existing features vocabularies")
    group.add('--src_vocab_size', '-src_vocab_size', type=int, default=50000,
              help="Size of the source vocabulary")
    group.add('--tgt_vocab_size', '-tgt_vocab_size', type=int, default=50000,
              help="Size of the target vocabulary")

    group.add('--src_words_min_frequency',
              '-src_words_min_frequency', type=int, default=0)
    group.add('--tgt_words_min_frequency',
              '-tgt_words_min_frequency', type=int, default=0)

    group.add('--dynamic_dict', '-dynamic_dict', action='store_true',
              help="Create dynamic dictionaries")
    group.add('--share_vocab', '-share_vocab', action='store_true',
              help="Share source and target vocabulary")

    # Truncation options, for text corpus
    group = parser.add_argument_group('Pruning')
    # add by wchen
    group.add('--src_sent_num_trunc', '-src_sent_num_trunc',
              type=int, default=None,
              help="Truncate the number of src sentence. None means no truncating. 15 is for HRE.")
    group.add('--src_sent_length_trunc', '-src_sent_length_trunc',
              type=int, default=None,
              help="Truncate a source sentence. None means no truncating. 55 is for HRE.")
    group.add('--src_seq_min_length', '-src_seq_min_length', type=int, default=1,
              help="Minimum source sequence length")
    group.add('--tgt_seq_min_length', '-tgt_seq_min_length', type=int, default=1,
              help="Minimum target sequence length")
    #
    group.add('--src_seq_length', '-src_seq_length', type=int, default=50,
              help="Maximum source sequence length")
    group.add('--src_seq_length_trunc', '-src_seq_length_trunc',
              type=int, default=None,
              help="Truncate source sequence length.")
    group.add('--tgt_seq_length', '-tgt_seq_length', type=int, default=50,
              help="Maximum target sequence length to keep.")
    group.add('--tgt_seq_length_trunc', '-tgt_seq_length_trunc',
              type=int, default=None,
              help="Truncate target sequence length.")
    group.add('--lower', '-lower', action='store_true', help='lowercase data')
    group.add('--filter_valid', '-filter_valid', action='store_true',
              help='Filter validation data by src and/or tgt length')
    group.add('--used_eokp', '-used_eokp', action='store_true',
              help="Indicate whether the raw tgt used eokp token at the last position.")

    # Data processing options
    group = parser.add_argument_group('Random')
    group.add('--shuffle', '-shuffle', type=int, default=0,
              help="Shuffle data")
    group.add('--seed', '-seed', type=int, default=3435,
              help="Random seed")

    group = parser.add_argument_group('Logging')
    group.add('--report_every', '-report_every', type=int, default=100000,
              help="Report status every this many sentences")
    group.add('--log_file', '-log_file', type=str, default="",
              help="Output logs to a file under this path.")
    group.add('--log_file_level', '-log_file_level', type=str,
              action=StoreLoggingLevelAction,
              choices=StoreLoggingLevelAction.CHOICES,
              default="0")

    # Options most relevant to speech
    group = parser.add_argument_group('Speech')
    group.add('--sample_rate', '-sample_rate', type=int, default=16000,
              help="Sample rate.")
    group.add('--window_size', '-window_size', type=float, default=.02,
              help="Window size for spectrogram in seconds.")
    group.add('--window_stride', '-window_stride', type=float, default=.01,
              help="Window stride for spectrogram in seconds.")
    group.add('--window', '-window', default='hamming',
              help="Window type for spectrogram generation.")

    # Option most relevant to image input
    group.add('--image_channel_size', '-image_channel_size',
              type=int, default=3,
              choices=[3, 1],
              help="""Using grayscale image can training
                       model faster and smaller""")


def train_opts(parser):
    """ Training and saving options """

    group = parser.add_argument_group('General')
    group.add('--data', '-data', required=True,
              help="""Path prefix to the ".train.pt" and
                       ".valid.pt" file path from preprocess.py""")

    group.add('--save_model', '-save_model', default='model',
              help="""Model filename (the model will be saved as
                       <save_model>_N.pt where N is the number
                       of steps""")

    group.add('--save_checkpoint_steps', '-save_checkpoint_steps',
              type=int, default=5000,
              help="""Save a checkpoint every X steps""")
    group.add('--keep_checkpoint', '-keep_checkpoint', type=int, default=-1,
              help="""Keep X checkpoints (negative: keep all)""")

    # GPU
    group.add('--gpuid', '-gpuid', default=[], nargs='*', type=int,
              help="Deprecated see world_size and gpu_ranks.")
    group.add('--gpu_ranks', '-gpu_ranks', default=[], nargs='*', type=int,
              help="list of ranks of each process.")
    group.add('--world_size', '-world_size', default=1, type=int,
              help="total number of distributed processes.")
    group.add('--gpu_backend', '-gpu_backend',
              default="nccl", type=str,
              help="Type of torch distributed backend")
    group.add('--gpu_verbose_level', '-gpu_verbose_level', default=0, type=int,
              help="Gives more info on each process per GPU.")
    group.add('--master_ip', '-master_ip', default="localhost", type=str,
              help="IP of master for torch.distributed training.")
    group.add('--master_port', '-master_port', default=10000, type=int,
              help="Port of master for torch.distributed training.")

    group.add('--seed', '-seed', type=int, default=-1,
              help="""Random seed used for the experiments
                       reproducibility.""")

    # Init options
    group = parser.add_argument_group('Initialization')
    group.add('--param_init', '-param_init', type=float, default=0.1,
              help="""Parameters are initialized over uniform distribution
                       with support (-param_init, param_init).
                       Use 0 to not use initialization""")
    group.add('--param_init_glorot', '-param_init_glorot', action='store_true',
              help="""Init parameters with xavier_uniform.
                       Required for transfomer.""")

    group.add('--train_from', '-train_from', default='', type=str,
              help="""If training from a checkpoint then this is the
                       path to the pretrained model's state_dict.""")
    group.add('--reset_optim', '-reset_optim', default='none',
              choices=['none', 'all', 'states', 'keep_states'],
              help="""Optimization resetter when train_from.""")

    # Pretrained word vectors
    group.add('--pre_word_vecs_enc', '-pre_word_vecs_enc',
              help="""If a valid path is specified, then this will load
                       pretrained word embeddings on the encoder side.
                       See README for specific formatting instructions.""")
    group.add('--pre_word_vecs_dec', '-pre_word_vecs_dec',
              help="""If a valid path is specified, then this will load
                       pretrained word embeddings on the decoder side.
                       See README for specific formatting instructions.""")
    # Fixed word vectors
    group.add('--fix_word_vecs_enc', '-fix_word_vecs_enc',
              action='store_true',
              help="Fix word embeddings on the encoder side.")
    group.add('--fix_word_vecs_dec', '-fix_word_vecs_dec',
              action='store_true',
              help="Fix word embeddings on the decoder side.")

    # Optimization options
    group = parser.add_argument_group('Optimization- Type')
    group.add('--batch_size', '-batch_size', type=int, default=64,
              help='Maximum batch size for training')
    group.add('--batch_type', '-batch_type', default='sents',
              choices=["sents", "tokens"],
              help="""Batch grouping for batch_size. Standard
                               is sents. Tokens will do dynamic batching""")
    group.add('--normalization', '-normalization', default='sents',
              choices=["sents", "tokens"],
              help='Normalization method of the gradient.')
    group.add('--accum_count', '-accum_count', type=int, default=1,
              help="""Accumulate gradient this many times.
                       Approximately equivalent to updating
                       batch_size * accum_count batches at once.
                       Recommended for Transformer.""")
    group.add('--valid_steps', '-valid_steps', type=int, default=10000,
              help='Perfom validation every X steps')
    group.add('--valid_batch_size', '-valid_batch_size', type=int, default=32,
              help='Maximum batch size for validation')
    group.add('--max_generator_batches', '-max_generator_batches',
              type=int, default=32,
              help="""Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but
                        uses more memory. Set to 0 to disable""")
    group.add('--train_steps', '-train_steps', type=int, default=100000,
              help='Number of training steps')
    group.add('--epochs', '-epochs', type=int, default=0,
              help='Deprecated epochs see train_steps')
    group.add('--optim', '-optim', default='sgd',
              choices=['sgd', 'adagrad', 'adadelta', 'adam',
                       'sparseadam', 'adafactor'],
              help="""Optimization method.""")
    group.add('--adagrad_accumulator_init', '-adagrad_accumulator_init',
              type=float, default=0,
              help="""Initializes the accumulator values in adagrad.
                       Mirrors the initial_accumulator_value option
                       in the tensorflow adagrad (use 0.1 for their default).
                       """)
    group.add('--max_grad_norm', '-max_grad_norm', type=float, default=5,
              help="""If the norm of the gradient vector exceeds this,
                       renormalize it to have the norm equal to
                       max_grad_norm""")
    group.add('--dropout', '-dropout', type=float, default=0.3,
              help="Dropout probability; applied in LSTM stacks.")
    group.add('--truncated_decoder', '-truncated_decoder', type=int, default=0,
              help="""Truncated bptt.""")
    group.add('--adam_beta1', '-adam_beta1', type=float, default=0.9,
              help="""The beta1 parameter used by Adam.
                       Almost without exception a value of 0.9 is used in
                       the literature, seemingly giving good results,
                       so we would discourage changing this value from
                       the default without due consideration.""")
    group.add('--adam_beta2', '-adam_beta2', type=float, default=0.999,
              help="""The beta2 parameter used by Adam.
                       Typically a value of 0.999 is recommended, as this is
                       the value suggested by the original paper describing
                       Adam, and is also the value adopted in other frameworks
                       such as Tensorflow and Kerras, i.e. see:
                       https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
                       https://keras.io/optimizers/ .
                       Whereas recently the paper "Attention is All You Need"
                       suggested a value of 0.98 for beta2, this parameter may
                       not work well for normal models / default
                       baselines.""")
    group.add('--label_smoothing', '-label_smoothing', type=float, default=0.0,
              help="""Label smoothing value epsilon.
                       Probabilities of all non-true labels
                       will be smoothed by epsilon / (vocab_size - 1).
                       Set to zero to turn off label smoothing.
                       For more detailed information, see:
                       https://arxiv.org/abs/1512.00567""")
    # learning rate
    group = parser.add_argument_group('Optimization- Rate')
    group.add('--learning_rate', '-learning_rate', type=float, default=1.0,
              help="""Starting learning rate.
                       Recommended settings: sgd = 1, adagrad = 0.1,
                       adadelta = 1, adam = 0.001""")
    group.add('--learning_rate_decay', '-learning_rate_decay',
              type=float, default=0.5,
              help="""If update_learning_rate, decay learning rate by
                       this much if steps have gone past
                       start_decay_steps""")
    # add by wchen
    group.add('--lambda_valid_words_nll', '-lambda_valid_words_nll', type=float, default=1.0,
              help="""The factor of the negative log-liklihood loss of all valid words.""")
    group.add('--lambda_first_word_nll', '-lambda_first_word_nll', type=float, default=1.0,
              help="""The factor of the negative log-liklihood loss of  the first valid word.""")
    group.add('--exclusive_loss', '-exclusive_loss', action="store_true", default=False,
              help="""Whether to add the exclusive loss for the first valid word decoding step.""")
    group.add('--lambda_ex', '-lambda_ex', type=float, default=1,
              help="""The factor of the exclusive loss.""")
    group.add('--ex_loss_win_size', '-ex_loss_win_size', type=int, default=1,
              help="""The window size used when calculating the exclusive loss.""")
    group.add('--orthogonal_loss', '-orthogonal_loss', action="store_true", default=False,
              help="""Whether add the orthogonal regularization for the word_init_states 
              from the sentence level decoder""")
    group.add('--lambda_orthogonal', '-lambda_orthogonal', type=float, default=0.03,
              help="""The factor of the orthogonal loss. Refer to Yuan et al.""")
    group.add('--coverage_loss', '-coverage_loss', action="store_true", default=False,
              help="""Whether add the coverage loss for the sentence level decoder when training""")
    group.add('--lambda_coverage', '-lambda_coverage', type=float, default=1,
              help='Lambda value for coverage.')
    group.add('--target_enc_loss', '-target_enc_loss', action="store_true", default=False,
              help="""Whether add the target_enc_loss for the CatSeqD decoder when training""")
    group.add('--lambda_te', '-lambda_te', type=float, default=0.03,
              help='Lambda value for target_enc_loss.')
    group.add('--exclusive_dec_loss', '-exclusive_dec_loss', action="store_true", default=False,
              help="""Whether add the exclusive_dec_loss for the HRD decoder when training""")
    group.add('--lambda_ed', '-lambda_ed', type=float, default=0.03,
              help='Lambda value for exclusive_dec_loss.')
    group.add('--early_stop_cnt', '-early_stop_cnt', type=int, default=3,
              help="""Stop the training when the validation ppl stop decreasing for early_stop_cnt times""")
    group.add('--start_decay_steps', '-start_decay_steps',
              type=int, default=50000,
              help="""Start decaying every decay_steps after
                       start_decay_steps""")
    group.add('--decay_steps', '-decay_steps', type=int, default=10000,
              help="""Decay every decay_steps""")

    group.add('--decay_method', '-decay_method', type=str, default="none",
              choices=['noam', 'none'], help="Use a custom decay rate.")
    group.add('--warmup_steps', '-warmup_steps', type=int, default=4000,
              help="""Number of warmup steps for custom decay.""")

    group = parser.add_argument_group('Logging')
    group.add('--report_every', '-report_every', type=int, default=50,
              help="Print stats at this interval.")
    group.add('--log_file', '-log_file', type=str, default="",
              help="Output logs to a file under this path.")
    group.add('--log_file_level', '-log_file_level', type=str,
              action=StoreLoggingLevelAction,
              choices=StoreLoggingLevelAction.CHOICES,
              default="0")
    group.add('--exp_host', '-exp_host', type=str, default="",
              help="Send logs to this crayon server.")
    group.add('--exp', '-exp', type=str, default="",
                       help="Name of the experiment for logging.")
    # Use TensorboardX for visualization during training
    group.add('--tensorboard', '-tensorboard', action="store_true",
              help="""Use tensorboardX for visualization during training.
                       Must have the library tensorboardX.""")
    group.add_argument("-tensorboard_log_dir", type=str,
                       default="runs/onmt",
                       help="""Log directory for Tensorboard.
                       This is also the name of the run.
                       """)

    group = parser.add_argument_group('Speech')
    # Options most relevant to speech
    group.add('--sample_rate', '-sample_rate', type=int, default=16000,
              help="Sample rate.")
    group.add('--window_size', '-window_size', type=float, default=.02,
              help="Window size for spectrogram in seconds.")

    # Option most relevant to image input
    group.add('--image_channel_size', '-image_channel_size',
              type=int, default=3, choices=[3, 1],
              help="""Using grayscale image can training
                       model faster and smaller""")


def translate_opts(parser):
    """ Translation / inference options """
    group = parser.add_argument_group('Model')
    group.add('--model', '-model', dest='models', metavar='MODEL',
              nargs='+', type=str, default=[], required=True,
              help='Path to model .pt file(s). '
              'Multiple models can be specified, '
              'for ensemble decoding.')
    group.add('--avg_raw_probs', '-avg_raw_probs', action='store_true',
              help="""If this is set, during ensembling scores from
              different models will be combined by averaging their
              raw probabilities and then taking the log. Otherwise,
              the log probabilities will be averaged directly.
              Necessary for models whose output layers can assign
              zero probability.""")

    group = parser.add_argument_group('Data')
    group.add('--data_type', '-data_type', default="text",
              help="Type of the source input. Options: [text|img].")

    group.add('--src', '-src', required=True,
                       help="""Source sequence to decode (one line per
                       sequence)""")
    group.add('--src_title', '-src_title',
              help="""The title of source papers to decode (one line per title sequence)""")
    group.add('--src_dir', '-src_dir', default="",
              help='Source directory for image or audio files')
    group.add('--tgt', '-tgt',
                       help='True target sequence (optional)')
    group.add('--output', '-output', default='pred.txt',
              help="""Path to output the predictions (each line will
                       be the decoded sequence""")
    group.add('--report_bleu', '-report_bleu', action='store_true',
              help="""Report bleu score after translation,
                       call tools/multi-bleu.perl on command line""")
    group.add('--report_rouge', '-report_rouge', action='store_true',
              help="""Report rouge 1/2/3/L/SU4 score after translation
                       call tools/test_rouge.py on command line""")

    # Options for evaluation, add by wchen
    # parser.add_argument('--tgt_keys', '-tgt_keys', type=str,
    #                     default='data\\full_data\\kp20k_testing_keyword.txt')
    group.add('--single_word_maxnum', '-single_word_maxnum', type=int, default=-1,
              help=""""The maximum number of preserved single word predictions.
              The default is -1 which means preserve all the single word predictions.""")
    group.add('--filter_dot_comma_unk', '-filter_dot_comma_unk', type=bool, default=True,
              help="""Whether to filter out the predictions with dot, comma and unk symbol. The default is true.""")
    group.add('--match_method', '-match_method', type=str, default='word_match', choices=['word_match'])
    group.add('--ap_splitter', '-ap_splitter', type=str, default='<absent_end>')
    group.add('--stopwords_file', '-stopwords_file', type=str, default='data\\stfd_stopwords\\corenlp_stopwords.json')

    # Options most relevant to summarization.
    group.add('--dynamic_dict', '-dynamic_dict', action='store_true',
              help="Create dynamic dictionaries")
    group.add('--share_vocab', '-share_vocab', action='store_true',
              help="Share source and target vocabulary")

    group = parser.add_argument_group('Random Sampling')
    group.add('--random_sampling_topk', '-random_sampling_topk',
              default=1, type=int,
              help="""Set this to -1 to do random sampling from full
                      distribution. Set this to value k>1 to do random
                      sampling restricted to the k most likely next tokens.
                      Set this to 1 to use argmax or for doing beam
                      search.""")
    group.add('--random_sampling_temp', '-random_sampling_temp',
              default=1., type=float,
              help="""If doing random sampling, divide the logits by
                       this before computing softmax during decoding.""")

    group = parser.add_argument_group('Beam')
    group.add('--fast', '-fast', action="store_true",
              help="""Use fast beam search (some features may not be
                       supported!)""")
    group.add('--beam_size', '-beam_size', type=int, default=5,
              help='Beam size')
    group.add('--min_length', '-min_length', type=int, default=0,
              help='Minimum prediction length')
    group.add('--max_length', '-max_length', type=int, default=100,
              help='Maximum prediction length.')
    group.add('--max_sent_length', '-max_sent_length', action=DeprecateAction,
              help="Deprecated, use `-max_length` instead")

    # add by wchen for hierarchical keyphrase genenration model
    group.add('--preds_cutoff_num', '-preds_cutoff_num', type=int, default=-1)
    group.add('--min_kp_num', '-min_kp_num', type=int, default=1,
              help='The minimum number of predicting keyphrase.')
    group.add('--min_kp_length', '-min_kp_length', type=int, default=1,
              help='The minimum length of predicting keyphrase.')
    group.add('--max_kp_num', '-max_kp_num', type=int, default=10,
              help='The maximum number of predicting keyphrase.')
    group.add('--max_kp_length', '-max_kp_length', type=int, default=6,
              help='The maximum length of predicting keyphrase.')

    group.add('--translate_report_num', '-translate_report_num', type=int, default=500,
              help='Report after finishing the translation of every translate_report_num test examples.')

    # Alpha and Beta values for Google Length + Coverage penalty
    # Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
    group.add('--stepwise_penalty', '-stepwise_penalty', action='store_true',
              help="""Apply penalty at every decoding step.
                       Helpful for summary penalty.""")
    group.add('--length_penalty', '-length_penalty', default='none',
              choices=['none', 'wu', 'avg'],
              help="""Length Penalty to use.""")
    group.add('--coverage_penalty', '-coverage_penalty', default='none',
              choices=['none', 'wu', 'summary'],
              help="""Coverage Penalty to use.""")
    group.add('--alpha', '-alpha', type=float, default=0.,
              help="""Google NMT length penalty parameter
                        (higher = longer generation)""")
    group.add('--beta', '-beta', type=float, default=-0.,
              help="""Coverage penalty parameter""")
    group.add('--block_ngram_repeat', '-block_ngram_repeat',
              type=int, default=0,
              help='Block repetition of ngrams during decoding.')
    group.add('--ignore_when_blocking', '-ignore_when_blocking',
              nargs='+', type=str, default=[],
              help="""Ignore these strings when blocking repeats.
                       You want to block sentence delimiters.""")
    group.add('--replace_unk', '-replace_unk', action="store_true",
              help="""Replace the generated UNK tokens with the
                       source token that had highest attention weight. If
                       phrase_table is provided, it will lookup the
                       identified source token and give the corresponding
                       target token. If it is not provided(or the identified
                       source token does not exist in the table) then it
                       will copy the source token""")
    # add by wchen to forbid unk when translate
    group.add('--forbid_unk', '-forbid_unk', action="store_true",
              help="""Whether to set the generation probability of unk as 0.0.""")
    group.add('--first_valid_word_exclusive_search', '-first_valid_word_exclusive_search', action="store_true",
              help="""Whether to force to generate different first valid word with the previously generated keyphrases.""")
    group.add('--exclusive_window_size', '-exclusive_window_size', type=int, default=0,
              help="""The window size for first_valid_word_exclusive_search.
              0 means no exclusive search. -1 means using the whole decoding history.""")

    group = parser.add_argument_group('Logging')
    group.add('--verbose', '-verbose', action="store_true",
              help='Print scores and predictions for each sentence')
    group.add('--log_file', '-log_file', type=str, default="",
              help="Output logs to a file under this path.")
    group.add('--log_file_level', '-log_file_level', type=str,
              action=StoreLoggingLevelAction,
              choices=StoreLoggingLevelAction.CHOICES,
              default="0")
    group.add('--attn_debug', '-attn_debug', action="store_true",
              help='Print best attn for each word')
    group.add('--dump_beam', '-dump_beam', type=str, default="",
              help='File to dump beam information to.')
    group.add('--n_best', '-n_best', type=int, default=1,
              help="""If verbose is set, will output the n_best
                       decoded sentences""")

    group = parser.add_argument_group('Efficiency')
    group.add('--batch_size', '-batch_size', type=int, default=30,
              help='Batch size')
    group.add('--gpu', '-gpu', type=int, default=-1,
                       help="Device to run on")

    # Options most relevant to speech.
    group = parser.add_argument_group('Speech')
    group.add('--sample_rate', '-sample_rate', type=int, default=16000,
              help="Sample rate.")
    group.add('--window_size', '-window_size', type=float, default=.02,
              help='Window size for spectrogram in seconds')
    group.add('--window_stride', '-window_stride', type=float, default=.01,
              help='Window stride for spectrogram in seconds')
    group.add('--window', '-window', default='hamming',
              help='Window type for spectrogram generation')

    # Option most relevant to image input
    group.add('--image_channel_size', '-image_channel_size',
              type=int, default=3, choices=[3, 1],
              help="""Using grayscale image can training
                       model faster and smaller""")


def add_md_help_argument(parser):
    """ md help parser """
    parser.add('--md', '-md', action=MarkdownHelpAction,
               help='print Markdown-formatted help text and exit.')


# MARKDOWN boilerplate

# Copyright 2016 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
class MarkdownHelpFormatter(configargparse.HelpFormatter):
    """A really bare-bones configargparse help formatter that generates valid
       markdown.

       This will generate something like:
       usage
       # **section heading**:
       ## **--argument-one**
       ```
       argument-one help text
       ```
    """

    def _format_usage(self, usage, actions, groups, prefix):
        return ""

    def format_help(self):
        print(self._prog)
        self._root_section.heading = '# Options: %s' % self._prog
        return super(MarkdownHelpFormatter, self).format_help()

    def start_section(self, heading):
        super(MarkdownHelpFormatter, self) \
            .start_section('### **%s**' % heading)

    def _format_action(self, action):
        if action.dest == "help" or action.dest == "md":
            return ""
        lines = []
        lines.append('* **-%s %s** ' % (action.dest,
                                        "[%s]" % action.default
                                        if action.default else "[]"))
        if action.help:
            help_text = self._expand_help(action)
            lines.extend(self._split_lines(help_text, 80))
        lines.extend(['', ''])
        return '\n'.join(lines)


class MarkdownHelpAction(configargparse.Action):
    """ MD help action """

    def __init__(self, option_strings,
                 dest=configargparse.SUPPRESS, default=configargparse.SUPPRESS,
                 **kwargs):
        super(MarkdownHelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.formatter_class = MarkdownHelpFormatter
        parser.print_help()
        parser.exit()


class StoreLoggingLevelAction(configargparse.Action):
    """ Convert string to logging level """
    import logging
    LEVELS = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET
    }

    CHOICES = list(LEVELS.keys()) + [str(_) for _ in LEVELS.values()]

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(StoreLoggingLevelAction, self).__init__(
            option_strings, dest, help=help, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        # Get the key 'value' in the dict, or just use 'value'
        level = StoreLoggingLevelAction.LEVELS.get(value, value)
        setattr(namespace, self.dest, level)


class DeprecateAction(configargparse.Action):
    """ Deprecate action """

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(DeprecateAction, self).__init__(option_strings, dest, nargs=0,
                                              help=help, **kwargs)

    def __call__(self, parser, namespace, values, flag_name):
        help = self.help if self.mdhelp is not None else ""
        msg = "Flag '%s' is deprecated. %s" % (flag_name, help)
        raise configargparse.ArgumentTypeError(msg)
