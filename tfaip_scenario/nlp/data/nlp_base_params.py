# Copyright 2020 The neiss authors. All Rights Reserved.
#
# This file is part of tf2_neiss_nlp.
#
# tf2_neiss_nlp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# tf2_neiss_nlp is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# tf2_neiss_nlp. If not, see http://www.gnu.org/licenses/.
# ==============================================================================
import os
from abc import ABC
from dataclasses import dataclass, field
from typing import List

from dataclasses_json import config
from paiargparse import pai_meta, pai_dataclass
from tensorflow_datasets.core.features.text import SubwordTextEncoder
from transformers import BertTokenizer, ElectraTokenizer

from tfaip import DataBaseParams
from tfaip.resource.resource import Resource

this_dir = os.path.dirname(os.path.realpath(__file__))


@pai_dataclass
@dataclass
class NLPDataParams(DataBaseParams, ABC):
    tokenizer: Resource = field(
        default=None,
        metadata={
            **pai_meta(help="File specifying the tokenizer used (incl. .subwords", required=True),
            **config(encoder=Resource.encode, decoder=Resource.decode),
        },
    )
    # types that are add features int or float
    add_types: List[float] = field(default_factory=lambda: [])
    magnitude: int = 3
    noise: str = "uniform"
    fixate_edges: bool = True
    map_edges: bool = False
    buffer: int = 50
    random_seed: int = None
    shuffle_filenames: bool = True
    shuffle_text_data: bool = True
    whole_word_masking: bool = False
    use_hf_model: bool = False
    use_hf_electra_model: bool = False
    pretrained_hf_model: str = ""
    cls_token_id_: int = None
    sep_token_id_: int = None
    pad_token_id_: int = None
    tok_vocab_size_: int = None
    whole_word_attention: bool = False
    paifile_input: bool = False

    def get_tokenizer(self):
        if self.use_hf_model:
            if self.use_hf_electra_model:
                tokenizer = ElectraTokenizer.from_pretrained(self.pretrained_hf_model)
            else:
                tokenizer = BertTokenizer.from_pretrained(self.pretrained_hf_model)
            self.cls_token_id_ = tokenizer.cls_token_id
            self.sep_token_id_ = tokenizer.sep_token_id
            self.pad_token_id_ = tokenizer.pad_token_id
        else:
            assert str(self.tokenizer).endswith(".subwords")
            tokenizer = SubwordTextEncoder.load_from_file(str(self.tokenizer).strip(".subwords"))
            self.cls_token_id_ = tokenizer.vocab_size
            self.sep_token_id_ = tokenizer.vocab_size + 1
            self.pad_token_id_ = 0
            self.tok_vocab_size_ = tokenizer.vocab_size
        return tokenizer

    @property
    def cls_token_id(self) -> int:
        if self.cls_token_id_ is None:
            self.get_tokenizer()
        return self.cls_token_id_

    @property
    def sep_token_id(self) -> int:
        if self.sep_token_id_ is None:
            self.get_tokenizer()
        return self.sep_token_id_

    @property
    def pad_token_id(self) -> int:
        if self.pad_token_id_ is None:
            self.get_tokenizer()
        return self.pad_token_id_

    @property
    def tok_vocab_size(self) -> int:
        if self.tok_vocab_size_ is None:
            self.get_tokenizer()
        return self.tok_vocab_size_
