"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""
from onmt.inputters.inputter import make_features, \
    load_fields_from_vocab, get_fields, get_nested_fields, OrderedIterator, \
    build_dataset, build_vocab, old_style_vocab
from onmt.inputters.dataset_base import DatasetBase
from onmt.inputters.text_dataset import TextDataset
from onmt.inputters.image_dataset import ImageDataset
from onmt.inputters.audio_dataset import AudioDataset

# add by wchen
from onmt.inputters.my_fields import SentPosiField


__all__ = ['DatasetBase', 'make_features',
           'load_fields_from_vocab', 'get_fields', 'get_nested_fields',
           'build_dataset', 'old_style_vocab',
           'build_vocab', 'OrderedIterator',
           'TextDataset', 'ImageDataset', 'AudioDataset', 'SentPosiField']
