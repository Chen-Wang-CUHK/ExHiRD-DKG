#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import configargparse

from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator
from evaluation_utils import evaluate_func

import onmt.opts as opts


def main(opt):
    logger = init_logger(opt.log_file)
    translator = build_translator(opt, report_score=True, logger=logger)
    translator.translate(
        src=opt.src,
        src_title=opt.src_title,
        tgt=opt.tgt,
        src_dir=opt.src_dir,
        batch_size=opt.batch_size,
        attn_debug=opt.attn_debug
    )
    # add by wchen
    evaluate_func(opt)


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description='translate.py',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    opts.config_opts(parser)
    opts.add_md_help_argument(parser)
    opts.translate_opts(parser)

    opt = parser.parse_args()
    main(opt)
