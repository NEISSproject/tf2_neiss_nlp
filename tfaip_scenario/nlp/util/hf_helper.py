# (c) 2021, PLANET artificial intelligence GmbH
#
# @contact legal@planet-ai.de
# ==============================================================================
import logging

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def load_hf_tokenizer(model_name, cache_dir=None):
    try:
        logger.info("get cached tokenizer for: {}".format(model_name))
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
    except OSError as e:
        logger.info(f"Could not load model with model name or path from LOCAL files '{model_name}'")
        try:
            logger.info("(Down)loading tokenizer for: {}".format(model_name))
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        except EnvironmentError as e:
            logger.exception(e)
            raise ValueError(f"Could not load model with model name or path '{model_name}'")
    return tokenizer
