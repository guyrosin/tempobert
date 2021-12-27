# Time Masking for Temporal Language Models

This repository provides a reference implementation of the paper:
>Time Masking for Temporal Language Models<br>
Guy D. Rosin, Ido Guy, and Kira Radinsky<br>
Accepted to WSDM 2022<br>
Preprint: https://arxiv.org/abs/2110.06366


Abstract:
>Our world is constantly evolving, and so is the content on the web. Consequently, our languages, often said to mirror the world, are dynamic in nature.
However, most current contextual language models are static and cannot adapt to changes over time.<br>
In this work, we propose a temporal contextual language model called TempoBERT, which uses time as an additional context of texts.
Our technique is based on modifying texts with temporal information and performing time masking - specific masking for the supplementary time information.<br>
We leverage our approach for the tasks of semantic change detection and sentence time prediction, experimenting on diverse datasets in terms of time, size, genre, and language.
Our extensive evaluation shows that both tasks benefit from exploiting time masking.

## Prerequisites

- Create an Anaconda environment with Python 3.8 and install requirements: 

        conda create -n tempobert python=3.8
        conda activate tempobert
        conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

- Obtain datasets for training and evaluation: 
    - For semantic change detection: [LiverpoolFC dataset](https://github.com/marcodel13/Short-term-meaning-shift) or the [SemEval-2020 Task 1 datasets](https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/).
    - For sentence time prediction: our NYT dataset can be found under `datasets`.

## Usage

- Train TempoBERT using `train_tempobert.py`. This script is similar to Hugging Face's language modeling training script ([link](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling)), and introduces two new arguments: `time_embedding_type`, that is set to "prepend_token" by default, and `time_mlm_probability`, which can used to set a custom probability for time masking.
- Evaluate TempoBERT using `semantic_change_detection.py` for semantic change detection and `sentence_time_prediction.py` for sentence time prediction.

## Pointers

- The modification to the input texts is performed in `tokenization_utils_fast.py`, in `TempoPreTrainedTokenizerFast._batch_encode_plus()`.
- Time masking is performed in `temporal_data_collator.py`.