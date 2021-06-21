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
# tfaip. If not, see http://www.gnu.org/licenses/.
# ==============================================================================
from tfaip_addons.util.file.stringmapper import get_sm


def load_txt_conll(filename):
    with open(filename) as f:
        text = f.read()
        if "\r" in text:
            raise ValueError(
                "file '"
                + filename
                + "' contains non unix line endings: try dos2unix "
                + filename
            )
        training_data = text.strip("\n\t ").split("\n\n")
        list = []
        for sentence in training_data:
            if " " not in sentence:
                continue
            sentence_split = []
            word_entities = sentence.strip("\n").split("\n")
            for word_entity in word_entities:
                word_entity_split = word_entity.split(" ")
                assert len(word_entity_split) == 2
                sentence_split.append(word_entity_split)
            list.append(sentence_split)
    return list


def get_ner_string_mapper(path, check="IOB"):
    sm = get_sm(path)
    if check == "IOB":
        for channel in range(sm.size()):
            if sm.get_value(channel) in sm.word_to_id_map:
                maybe_i_tag = sm.get_value(channel).replace("B-", "I-")
                maybe_b_tag = sm.get_value(channel).replace("I-", "B-")
                if "B-" not in sm.get_value(channel) and "I-" not in sm.get_value(
                    channel
                ):
                    assert sm.get_value(channel) in [
                        "O",
                        "UNK",
                    ], f"Tag is not B-x, I-x or O! Invalid Tag: {sm.get_value(channel)}:{channel} map!"
                assert sm.has_channel(maybe_i_tag) and sm.has_channel(
                    maybe_b_tag
                ), f"Tag-Map-Error: channel: {channel}->{sm.get_value(channel)} has no corresponding I/B-Tag!"
    return sm
