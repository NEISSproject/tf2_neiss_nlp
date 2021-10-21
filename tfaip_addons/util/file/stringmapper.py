# Copyright 2021 The neiss authors. All Rights Reserved.
#
# This file is part of tf_neiss_nlp.
#
# tf_neiss_nlp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# tf_neiss_nlp is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# tf_neiss_nlp. If not, see http://www.gnu.org/licenses/.
# ==============================================================================
"""Definition of string mapping helpers"""
import codecs
import os
import re
from typing import Tuple, Union, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from tfaip_addons.util.file.pai_file import Classification

unknown_word = "UNK"


def get_sm(path) -> "StringMapper":
    """
    Creates and returns a StringMap. The mapping is loaded from the file given by `path`.

    Args:
        path: path to text file, should contain the mappings like 'character=key'.

    Returns:
        StringMap with loaded mapping.
    """
    if path.lower() == "id":

        class MapperId:
            def get_channel_tuple(self, classifications_dict: Dict[str, "Classification"]) -> List[Tuple[str, int]]:
                res = []
                for k, v in classifications_dict.items():
                    res.extend([(f"{k}:{r.label}", 0) for r in v.results])
                return res

            def get_channel(self, string, or_unknown_id=False):
                assert or_unknown_id is False
                return int(string)

            def get_value(self, channel):
                return str(channel)

        return MapperId()

    sm = StringMapper()
    try:
        if path.endswith("tsv"):
            sm.load_mapping_from_freq(path)
        elif path.endswith("txt"):
            sm.load_mapping_from_txt(path)
        else:
            sm.load_mapping_from_sm(path)
    except Exception as inst:
        raise RuntimeError(f"Can't load StringMap! Check if this is a valid path: {os.path.abspath(path)}") from inst

    return sm


class ChannelType(str):
    INDEX = 1
    STRING_INDEX = 2


class StringMapper:
    def __init__(self, sep="=", force_zero_oov=False):
        self.word_to_id_map: Dict[str, Union[int, Tuple[str, int]]] = {}
        self.id_to_word_map: Dict[Union[int, Tuple[str, int]], str] = {}
        self.loaded: bool = False
        self.channel_type: ChannelType = None
        self.unknown_id = -1
        # self.unknown_id = 0
        self._freq_map = {}
        self.sep = sep
        self.force_zero_oov = force_zero_oov

    def get_instance_and_set_type(self, key: str, value: str) -> Union[int, Tuple[str, int]]:
        value = value.strip()
        if re.fullmatch("[0-9]+", value):
            if self.channel_type is not None and self.channel_type != ChannelType.INDEX:
                raise Exception(
                    f"channel type already set to {self.channel_type.name} but for value {value} "
                    f"channel type {ChannelType.INDEX.name} calculated."
                )
            self.channel_type = ChannelType.INDEX
            index = int(value) - 1
            if index < 0:
                raise Exception(f"index of key {key} is {index + 1} but has to be at least 1")
            return index
        if re.fullmatch("[a-zA-Z_+-]+:[0-9]+", value):
            if self.channel_type is not None and self.channel_type != ChannelType.STRING_INDEX:
                raise Exception(
                    f"channel type alread set to {self.channel_type.name} but for value {value} "
                    f"channel type {ChannelType.STRING_INDEX.name} calculated."
                )
            self.channel_type = ChannelType.STRING_INDEX
            parts = value.split(":")
            return parts[0], int(parts[1])
        else:
            raise Exception(f"cannot parse key '{key}' and value '{value}' of stringmapper")

    def __str__(self):
        return self.id_to_word_map.__str__()

    def get_oov_id(self):
        return self.unknown_id

    def has_channel(self, string):
        return string in self.word_to_id_map

    def get_channel(self, string, or_unknown_id=True):
        if self.has_channel(string):
            return self.word_to_id_map[string]
        elif or_unknown_id:
            # print("unknown word: ", string, " returning ", self.unknown_id)
            return self.unknown_id
        return None

    def get_channel_tuple(self, classifications_dict: Dict[str, "Classification"]) -> List[Tuple[str, int]]:
        if self.channel_type == ChannelType.STRING_INDEX:
            res = []
            for key, classification in classifications_dict.items():
                for result in classification.results:
                    label = self.get_channel(f"{key}:{result.label}", or_unknown_id=False)
                    if label is not None:
                        res.append(label)
            return res
        res = []
        for classification in classifications_dict.values():
            for result in classification.results:
                label = self.get_channel(result.label, or_unknown_id=False)
                if label is not None:
                    res.append(("ic", label))
        return res

    def get_value(self, channel):
        if channel < self.size():
            return self.id_to_word_map[channel]
        else:
            return unknown_word

    def size(self):
        return len(self.id_to_word_map)

    def add(self, string: str, channel: Union[int, Tuple[str, int]], force_override=False):
        # if channel == None:
        #     channel = len(self.word_to_id_map)
        # print("len = " + str(index))
        assert channel is not None
        assert string is not None
        assert isinstance(channel, (int, tuple))
        assert isinstance(string, str)
        self.word_to_id_map[string] = channel
        if force_override or channel not in self.id_to_word_map:
            self.id_to_word_map[channel] = string
        # print("pos = " + str(index))
        # self.unknown_id = len(self.id_to_word_map)
        return channel

    def get_mapping(self, file_path):
        with codecs.open(file_path, "r", encoding="utf-8") as cm_file:
            raw = cm_file.readlines()
            return raw

    def get_freq_from_id(self, id_):
        return self._freq_map[id_]

    def get_freq_from_word(self, word):
        return self._freq_map[self.word_to_id_map[word]]

    def load_mapping_from_sm(self, file_path):
        if self.loaded:
            raise RuntimeError("map already loaded")
        with codecs.open(file_path, "r", encoding="utf-8") as cm_file:
            raw = cm_file.readlines()
            for line in raw:
                line = line.strip(os.linesep)
                line = line.strip()
                if len(line) == 0:
                    continue
                split = line.rsplit(self.sep, 1)
                value = split[1]
                key = split[0]
                index = self.get_instance_and_set_type(key, value)
                if self.channel_type == ChannelType.INDEX:
                    if key[0] == "\\":
                        key = key[1:]
                    if key == unknown_word:
                        self.unknown_id = id
                # print("'" + key + "' ==> " + str(index))
                self.add(key, index)
        self.loaded = True
        # self.unknown_id = len(self.dictBwd)
        if self.unknown_id < 0 and self.channel_type == ChannelType.INDEX:
            self.unknown_id = len(self.id_to_word_map)
            self.add(unknown_word, self.unknown_id)
        if self.force_zero_oov:
            self.oov_to_zero()

    def load_mapping_from_freq(self, file_path):
        if self.loaded:
            raise RuntimeError("map already loaded")
        with codecs.open(file_path, "r", encoding="utf-8") as cm_file:
            raw = cm_file.readlines()
            id = 0
            for line in raw:
                line = line.strip(os.linesep)
                split = line.rsplit(self.sep, 1)
                key = split[0]
                if key == unknown_word:
                    self.unknown_id = id
                freq = float(split[1])

                # specific values which are escaped by '\': delete '\'
                if key[0] == "\\":
                    key = key[1:]
                # print("'" + key + "' ==> " + str(index))
                self.add(key, id)
                self._freq_map[id] = freq
                id += 1
        self.loaded = True
        # self.unknown_id = len(self.dictBwd)
        if self.unknown_id < 0:
            self.unknown_id = len(self.id_to_word_map)
            self.add(unknown_word, self.unknown_id)
            self._freq_map[self.unknown_id] = 0
        if self.force_zero_oov and self.unknown_id != 0:
            self.oov_to_zero()
        # print("unknown id: ", self.unknown_id)

    def load_mapping_from_txt(self, file_path):
        if self.loaded:
            raise RuntimeError("map already loaded")
        with codecs.open(file_path, "r", encoding="utf-8") as cm_file:
            raw = cm_file.readlines()
            id_ = 0
            for line in raw:
                line = line.strip(os.linesep)
                key = line
                # specific values which are escaped by '\': delete '\'
                if key[0] == "\\":
                    key = key[1:]
                # print("'" + key + "' ==> " + str(index))
                self.add(key, id_)
                if key == unknown_word:
                    self.unknown_id = id_
                id_ += 1
        self.loaded = True
        # self.unknown_id = len(self.dictBwd)
        if self.unknown_id < 0:
            self.unknown_id = len(self.id_to_word_map)
            self.add(unknown_word, self.unknown_id)
        if self.force_zero_oov:
            self.oov_to_zero()

    def save_mapping(self, file_path):
        with codecs.open(file_path, "w", encoding="utf-8") as cm_file:
            for key, value in self.word_to_id_map.iteritems():
                if key == "\\" or key == "=":
                    cm_file.write("\\")
                cm_file.write(key)
                cm_file.write("=")
                cm_file.write(str(value + 1))
                cm_file.write(os.linesep)
                # file.write('NaC')
                # file.write('=')
                # file.write(str(len(self.dictBwd)))
        self.loaded = True

    def oov_to_zero(self):

        unknown_freq = self._freq_map[self.unknown_id]
        self.unknown_id = 0
        last_word = self.id_to_word_map[0]
        last_freq = self._freq_map[0]
        self.add(unknown_word, self.unknown_id, force_override=True)
        self._freq_map[self.unknown_id] = unknown_freq
        id = 0
        while last_word != unknown_word:
            id += 1
            key = last_word
            freq = last_freq
            last_word = self.id_to_word_map[id]
            last_freq = self._freq_map[id]
            self.add(key, id, force_override=True)
            self._freq_map[id] = freq


if __name__ == "__main__":
    sm = StringMapper(sep=",", force_zero_oov=True)
    sm.load_mapping_from_freq("/home/tobias/devel/projects/autolm/labels.vocab")
    print(sm.id_to_word_map)
    print(sm.word_to_id_map)
    print(sm._freq_map)
