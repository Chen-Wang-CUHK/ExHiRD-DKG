# ExHiRD-DKG
code for ACL 2020 paper [Exclusive Hierarchical Decoding for Deep Keyphrase Generation](https://arxiv.org/pdf/2004.08511.pdf)
# The code will be released soon.

# Citation
You can cite our paper by:
```
@article{DBLP:journals/corr/abs-2004-08511,
  author    = {Wang Chen and
               Hou Pong Chan and
               Piji Li and
               Irwin King},
  title     = {Exclusive Hierarchical Decoding for Deep Keyphrase Generation},
  journal   = {CoRR},
  volume    = {abs/2004.08511},
  year      = {2020},
  url       = {https://arxiv.org/abs/2004.08511},
  archivePrefix = {arXiv},
  eprint    = {2004.08511},
  timestamp = {Wed, 22 Apr 2020 12:57:53 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2004-08511.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
# Table of contents
   * [Dependencies](#dependencies)
   * [Get the processed train/val/test datasets](#get-the-processed-train/val/test-datasets)
   * How to use the code? (these parts are still under updating.)
   * [Evaluate the predictions](#evaluate-the-predictions)
   * [Download our final predictions](#download-our-final-predictions)
   * [Citation](#citation)

# Dependencies
- python 3.6.8
- pytorch 1.0 (CUDA9.0)
- torchtext 0.4.0

The full dependencies are listed in `Requirements.txt`.

# Get the processed train/val/test datasets
Download the [processed testing datasets](https://www.dropbox.com/s/tavebz23va1hvrd/ExHiRD_test_datasets.zip?dl=1).

# Evaluate the predictions
After specifying the corret path of the predictions and testing datasets in the `sh/evaluation/evaluate_ExHiRD_h.sh` and `sh/evaluation/evaluate_ExHiRD_h.sh`, you can run the following command lines to evaluate the predictions of ExHiRD-h and ExHiRD-s:
'''
cd sh/evaluation/
sh evaluate_ExHiRD_h.sh
sh evaluate_ExHiRD_s.sh
'''

All the post-processing steps (including removing duplicated predictions, restricting the maximum number of single-word predictions (if set), and filtering predictions which contain dot, comma, or unk token.) and evaluation metrics are integrated in `evaluation_utils.py`.
The standard Macro-averaged F1@5 (i.e. Macro std_F1@5) and F1@M (i.e. Macro std_F1@M) are reported in the paper.


# Download our final predictions
You can download our [raw final predictions](https://www.dropbox.com/s/29wu7omj1vnsbxb/ExHiRD_final_predictions.zip?dl=1) of our ExHiRD-h and ExHiRD-s methods for the four testing datasets. In the folder, `historyK` means the window size of exclusive search is set as K. The prediction post-processing is integrated in `evaluation_utils.py` including removing duplicated predictions, restricting the maximum number of single-word predictions (if set), and filtering predictions which contain dot, comma, or unk token.
