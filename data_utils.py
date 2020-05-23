import re
import string
import nltk
import numpy as np

# from stanfordcorenlp import StanfordCoreNLP
# CoreNLP = StanfordCoreNLP(r'D:\D_software\stanford-corenlp-full-2017-06-09')

EOKP_TOKEN = '<eokp>'
P_START = '<p_start>'
A_START = '<a_start>'
P_END = '<p_end>'
A_END = '<a_end>'
KEY_SEPERATOR = ';'
TITLE_SEPERATOR = '<eot>'
DIGIT = '<digit>'
KEYWORDS_TUNCATE = 10
MAX_KEYWORD_LEN = 6
PRINTABLE = set(string.printable)


def get_tokens(text, fine_grad=True, replace_digit=True, fine_grad_digit_matching=True):
    """
    Need use the same word tokenizer between keywords and source context
    keep [_<>,\(\)\.\'%], tokenize by nltk and split by [^a-zA-Z0-9_<>,\(\)\.\'%], replace digits to <digit>
    """
    # lowercase
    text = text.strip().lower()
    # remove \r \n \t
    text = re.sub(r'[\r\n\t]', '', text)
    # remove the content in [] and {}
    text = re.sub(r'\[.*?\]', '', text) # '[]' is usually used for reference
    text = re.sub(r'\{.*?\}', '', text) # '{}' is usually used for math expression
    text = re.sub(r'[\{\}\[\]]', '', text)
    # remove non-printable chars
    text = ''.join(list(filter(lambda x: x in PRINTABLE, text)))

    if fine_grad:
        # tokenize by non-letters
        # we still use the following tokenizer for fine granularity
        tokens = list(filter(lambda w: len(w) > 0, re.split(r"[^a-zA-Z0-9_<>,\(\)\.\']", text)))
    else:
        tokens = text.split()

    # if the text is empty, we return an empty string
    if len(tokens) == 0:
        return []

    # tokenize by a tokenizer
    tokens = CoreNLP.word_tokenize(' '.join(tokens))

    if replace_digit:
        # replace the digit terms with <digit>
        if fine_grad_digit_matching:
            # # ------------fine_grad_digit_matching 1---------------
            # # ['123', '12.3', '12d', 'd12', '2w2', '.12', '123.'] ->
            # # ['<digit>', '<digit>', '12d', 'd12', '2w2', '<digit>', '<digit>']
            # tokens = [w if not re.match('^[+-]?((\d+(\.\d*)?)|(\.\d+))$', w) else DIGIT for w in tokens]

            # ------------fine_grad_digit_matching 2---------------
            # ['123', '12.3', '12d', 'd12', '2w2', '.12', '123.'] ->
            # ['<digit>', '<digit>', '<digit>', 'd12', '<digit>', '<digit>', '<digit>']
            tokens = [w if not (re.match('^[+-]?((\d+(\.\d*)?)|(\.\d+))$', w) or w[0].isdigit()) else DIGIT for w in tokens]
        else:
            # "123 a123 123a" --> ["<digit>", "a123", "123a"]
            # TG-Net use '^\d+': "123 a123 123a" --> ["<digit>", "a123", "<digit>"]
            # KG_KE_KR_M also uses "^\d+$"
            tokens = [w if not re.match('^\d+$', w) else DIGIT for w in tokens]

        # remove repeated DIGIT tokens
        dup_digit_indicators = [False] + [True if (tokens[i-1] == DIGIT and tokens[i] == DIGIT) else False for i in range(1, len(tokens))]
        if len(dup_digit_indicators) != len(tokens):
            print('here')
        assert len(dup_digit_indicators) == len(tokens)
        tokens = [w for w, dup_flag in zip(tokens, dup_digit_indicators) if not dup_flag]

    return tokens


def process_keyphrase(keyword_str, limit_num=True, fine_grad=True, replace_digit=True, truncate_key_num=False):
    # lowercasing
    keyword_str = keyword_str.strip().lower()
    # replace some noise characters
    keyphrases = keyword_str.replace('?', '')
    # replace abbreviations
    keyphrases = re.sub(r'\(.*?\)', '', keyphrases)
    # Note: keyword should be applied the same tokenizer as the source did
    keyphrases = [get_tokens(keyword.strip(), fine_grad, replace_digit=replace_digit) for keyword in keyphrases.split(KEY_SEPERATOR) if len(keyword.strip()) != 0]

    # ['key1a key1b', 'key2a key2b']
    if limit_num:
        keyphrases = [' '.join(key) for key in keyphrases if 0 < len(key) <= MAX_KEYWORD_LEN]
    else:
        keyphrases = [' '.join(key) for key in keyphrases if 0 < len(key)]

    # constrain the maximum keyphrase number of each instance
    if truncate_key_num:
        keyphrases = keyphrases[:KEYWORDS_TUNCATE]

    return keyphrases


def in_context(context_list, tgt_list):
    match = False
    for c_idx in range(len(context_list) - len(tgt_list) + 1):
        context_piece = ' '.join(context_list[c_idx: c_idx + len(tgt_list)])
        tgt_piece = ' '.join(tgt_list)
        if context_piece == tgt_piece:
            match = True
            break
    return match


def ken_in_context(src_str, keyphrase_str_list, match_by_str=False):
    """
    From Ken's one to many code
    :param src_str: stemmed word list of source text
    :param keyphrase_str_list: stemmed list of word list
    :return:
    """
    num_keyphrases = len(keyphrase_str_list)
    is_present = np.zeros(num_keyphrases, dtype=bool)

    for i, keyphrase_word_list in enumerate(keyphrase_str_list):
        joined_keyphrase_str = ' '.join(keyphrase_word_list)

        if joined_keyphrase_str.strip() == "":  # if the keyphrase is an empty string
            is_present[i] = False
        else:
            if not match_by_str:  # match by word
                # check if it appears in source text
                match = False
                for src_start_idx in range(len(src_str) - len(keyphrase_word_list) + 1):
                    match = True
                    for keyphrase_i, keyphrase_w in enumerate(keyphrase_word_list):
                        src_w = src_str[src_start_idx + keyphrase_i]
                        if src_w != keyphrase_w:
                            match = False
                            break
                    if match:
                        break
                if match:
                    is_present[i] = True
                else:
                    is_present[i] = False
            else:  # match by str
                if joined_keyphrase_str in ' '.join(src_str):
                    is_present[i] = True
                else:
                    is_present[i] = False
    return is_present[0]
