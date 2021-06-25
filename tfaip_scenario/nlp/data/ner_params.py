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
from dataclasses import dataclass, field
from typing import Type

from dataclasses_json import config
from paiargparse import pai_meta, pai_dataclass
from tfaip.resource.resource import Resource
from tfaip_scenario.nlp.data.nlp_base_params import NLPDataParams
from tfaip_scenario.nlp.util.nlp_helper import get_ner_string_mapper

this_dir = os.path.dirname(os.path.realpath(__file__))


@pai_dataclass
@dataclass
class NERDataParams(NLPDataParams):
    @staticmethod
    def cls():
        from tfaip_scenario.nlp.data.ner import NERData

        return NERData

    # path to tag vocabulary
    tags: Resource = field(
        default=None,
        metadata={
            **pai_meta(help="File specifying the tag map used", required=True),
            **config(encoder=Resource.encode, decoder=Resource.decode),
        },
    )
    tokenizer_range: str = "sentence_v3"  # or sentence_v1
    bet_tagging: bool = False  # use split Begin/End and Class tags for better loss calculation
    wordwise_output: bool = False  # use as output only one vector per word regarding the specified method of wwo_mode
    wwo_mode: str = "first"  # use only the output of the first tokens of each word as outputs or 'mean' (mean of the tokenoutput per word) 'max' (elementwise max of the tokenoutput per word)
    max_words_per_sample_from_paifile: int = 1  # defines the maximum number of words a sample from the paifiles should contain. Lines will never be separated. Thus, if it is set to 1 every line from the paifile builds a sample
    mark_paifile_linebreaks: bool = False  # if True linebreaks from paifiles are marked by special tokens

    def get_tag_string_mapper(self):
        return get_ner_string_mapper(str(self.tags))
