"""Shared Hub login, Trackio, and logging setup for HF vision vertical trainers."""

from __future__ import annotations

import logging
import os
import sys

import trackio
import transformers
from transformers import TrainingArguments


def setup_hf_training_environment(training_args: TrainingArguments, *, logger: logging.Logger) -> None:
    """Log in to the Hub when ``HF_TOKEN`` / ``hfjob`` is set, init Trackio, configure logging."""
    from huggingface_hub import login

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("hfjob")
    if hf_token:
        login(token=hf_token)
        training_args.hub_token = hf_token
        logger.info("Logged in to Hugging Face Hub")
    elif training_args.push_to_hub:
        logger.warning("HF_TOKEN not found. Hub push will likely fail.")

    trackio.init(project=training_args.output_dir, name=training_args.run_name)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def finish_trackio_session() -> None:
    """Close the active Trackio run before emitting the summary block."""
    trackio.finish()
