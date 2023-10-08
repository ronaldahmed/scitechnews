#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

"""
# CHANGELOG
- do_predict: uses validation set, instead of test
"""

import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset

import evaluate
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import torch
import random

torch.manual_seed(42)
random.seed(42)

import pdb

MODEL_CACHE_DIR="<your cache dir here>"
METRICS_CACHE_DIR="<your cache dir here>"
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.23.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]

# RCT LABELS
_SUMMARY_ = "[SUMMARY]"
_CONTENT_ = "[CONTENT]"

SPECIAL_TOKENS = {
    'additional_special_tokens': [
        "[AUTHOR]","<null>","[ARTICLE]",
        "[none]", "[METHODS]", "[CONCLUSIONS]", "[RESULTS]", "[BACKGROUND]","[OBJECTIVE]"
    ]
}



@dataclass
class MyTrainingArguments(Seq2SeqTrainingArguments):
    """
    Arguments pertaining to training setup / decoding
    """
    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": (
                "Length penalty during decoding"
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    decoding_strat: Optional[str] = field(
        default="beam_search",
        metadata={
            "help": (
                "decoding / generation strategy [beam_search, ancestral, topk, nucleus, typical]"
            )
        },
    )
    top_k: Optional[int] = field(
        default=10,
        metadata={
            "help": (
                "topk"
            )
        },
    )
    nucleus: Optional[float] = field(
        default=0.9,
        metadata={
            "help": (
                "nucleus"
            )
        },
    )
    temperature: Optional[float] = field(
        default=1.0,
        metadata={
            "help": (
                "temperature"
            )
        },
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=MODEL_CACHE_DIR,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    min_target_length: Optional[int] = field(
        default=10,
        metadata={
            "help": (
                "The minimum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


def parse_generation_control_args(training_args,data_args,config):
    params = {
        "max_length": data_args.max_target_length if data_args is not None else config.max_length,
        "length_penalty": training_args.length_penalty,
        "num_return_sequences": 1,
        "num_beams" : None,
    }
    if   training_args.decoding_strat == "beam_search":
        if training_args.generation_num_beams is None:
            training_args.generation_num_beams = 1
        params["num_beams"] = max(training_args.generation_num_beams,1)

    elif training_args.decoding_strat == "ancestral":
        params["do_sample"] = True
        params["top_k"] = 0

    elif training_args.decoding_strat == "topk":
        params["do_sample"] = True
        params["top_k"] = training_args.top_k

    elif training_args.decoding_strat == "nucleus":
        params["do_sample"] = True
        params["top_k"] = 0
        params["top_p"] = training_args.nucleus
    
    params["output_scores"] = True
    params["return_dict_in_generate"] = True

    return params

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "GEM/xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
    "scientific_papers": ("article","abstract"),
}



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_prompt_generation", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer_dec = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer_dec.padding_side = "left"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    
    
    column_names = raw_datasets["test"].column_names
    
    

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    plan_column = "plan"
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False
    os.system("mkdir -p " + training_args.output_dir)

    gen_kwargs = parse_generation_control_args(training_args,data_args,config)


    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        # bos_token = tokenizer_dec.bos_token
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[plan_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[plan_column][i])

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=True, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        dec_input = tokenizer_dec(targets,padding=True, add_special_tokens=False)

        model_inputs["decoder_input_ids"] = dec_input["input_ids"]
        model_inputs["decoder_attention_mask"] = dec_input["attention_mask"]
        return model_inputs
    #

    
    max_target_length = data_args.max_target_length
    min_target_length = data_args.min_target_length
    if "test" not in raw_datasets:
        raise ValueError("--do_predict requires a test dataset")
    predict_dataset = raw_datasets["test"]                                ## CHANGE [test]
    if data_args.max_predict_samples is not None:
        max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
        predict_dataset = predict_dataset.select(range(max_predict_samples))
    with training_args.main_process_first(desc="prediction dataset map pre-processing"):
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    pred_dataloader = DataLoader(predict_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size)

    # Metric

    def postprocess_text(preds, labels=None):
        def clean_tok(text):
          clean_pred = text.replace("<pad>","").replace("<s>","").replace("</s>","").strip(" ")
          raw_prediction = None
          if "<n>" in clean_pred: raw_prediction = clean_pred.split("<n>")
          else:                   raw_prediction = nltk.sent_tokenize(clean_pred)
          raw_prediction = "\n".join([x.strip(" ") for x in raw_prediction if x.strip(" ")!=""])
          return raw_prediction
        preds = [clean_tok(pred) for pred in preds]
        # rougeLSum expects newline after each sentence
        if labels is not None:
            labels = [clean_tok(label) for label in labels]

        return preds, labels
    #

    logger.info("###################################################################################")
    logger.info(f">>  Device = { torch.cuda.is_available()} | {torch.cuda.current_device()}")
    logger.info("***** Running Generation *****")
    logger.info(f"  Num examples = {len(predict_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_eval_batch_size}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(len(pred_dataloader)))
    result = {}
    predictions = []
    model.eval()

    for step, batch in enumerate(pred_dataloader):
        with torch.no_grad():
            batch = batch.to(device)
            generated_bundle = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"],
                decoder_attention_mask=batch["decoder_attention_mask"],
                **gen_kwargs,
            )
            labels,decoded_labels = None,None
            # get prediction tokens
            generated_tokens = generated_bundle.sequences
            preds = tokenizer.batch_decode(
                    generated_bundle.sequences, clean_up_tokenization_spaces=True
                )
            batch_preds,_ = postprocess_text(preds)
            predictions.extend(batch_preds)
            progress_bar.update(1)

            # pdb.set_trace()
            # print(">>")
    #

    _basename = os.path.basename(data_args.test_file).rstrip(".json")

    output_prediction_file = os.path.join(training_args.output_dir,f"{_basename}_l{training_args.length_penalty}-pred.json")
    with open(output_prediction_file, "w") as writer:
        writer.write("\n".join([json.dumps({summary_column:x}) for x in predictions]))

    return


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
