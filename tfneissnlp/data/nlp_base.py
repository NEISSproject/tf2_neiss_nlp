# Copyright 2020 The neiss authors. All Rights Reserved.
#
# This file is part of tf2_neiss_nlp.
#
# tfaip is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# tfaip is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# tfaip. If not, see http://www.gnu.org/licenses/.
# ==============================================================================
import json
import logging
import os
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import partial
import random
from typing import List, Callable

import tensorflow as tf
from dataclasses_json import dataclass_json

from tfaip.base.data.data import DataBase
from tfaip.base.data.data_base_params import DataBaseParams
from tfneissnlp.util.nlp_helper import load_txt_conll
from tfaip.util.multiprocessing.data.pipeline import DataPipeline
from tfaip.util.multiprocessing.data.worker import DataWorker

logger = logging.getLogger(__name__)
this_dir = os.path.dirname(os.path.realpath(__file__))


@dataclass_json
@dataclass
class NLPDataParams(DataBaseParams):
    # path to tag vocabulary
    tags: str = ''
    # types that are add features int or float
    add_types: List[float] = field(default_factory=lambda: [])
    magnitude: int = 3
    noise: str = 'uniform'
    fixate_edges: bool = True
    map_edges: bool = False
    buffer: int = 50
    random_seed: int = None
    whole_word_masking: bool = False
    train_shuffle: bool = True
    val_shuffle: bool = False


class NLPData(DataBase):
    @staticmethod
    def get_params_cls():
        return NLPDataParams

    def __init__(self, params: NLPDataParams):
        super().__init__(params)
        self._auto_batch = False  # TODO: @jochen consider using auto batching
        self._params = params
        self._shapes = None
        self._types = None
        self._defaults = None

    @abstractmethod
    def get_worker_cls(self):
        raise NotImplementedError

    def _get_train_data(self):
        logger.info("create train_data")
        pipeline = NLPPipeline(self, self._params, self._params.train_lists, auto_repeat_input=True,
                               shuffle=self._params.train_shuffle,
                               processes=self._params.train_num_processes)
        dataset = tf.data.Dataset.from_generator(pipeline.output_generator,
                                                 output_shapes=self._shapes,
                                                 output_types=self._types)
        dataset = dataset.padded_batch(self._params.train_batch_size, self._shapes, self._defaults)
        return dataset

    def _get_val_data(self, val_list):
        logger.info("create val_data")
        pipeline = NLPPipeline(self, self._params, val_list, limit=self._params.val_limit,
                               shuffle=self._params.val_shuffle,
                               processes=self._params.val_num_processes)
        dataset = tf.data.Dataset.from_generator(pipeline.output_generator,
                                                 output_shapes=self._shapes,
                                                 output_types=self._types)
        dataset = dataset.padded_batch(self._params.val_batch_size, self._shapes, self._defaults)
        return dataset


def create_worker(params, cls):
    return cls(params)


class NLPPipeline(DataPipeline):
    def __init__(self, data,
                 params, lst_list,
                 limit=-1,
                 processes=1, auto_repeat_input=False, shuffle=True
                 ):
        self._params = params
        self.filenames = []
        self._shuffle = shuffle
        if not isinstance(lst_list, list):
            lst_list = [lst_list]
        for list_ in lst_list:
            with open(list_, 'r') as f:
                self.filenames.extend(f.read().splitlines())
        self._data = data
        super().__init__(data, processes, limit, auto_repeat_input)

    def create_worker_func(self) -> Callable[[], DataWorker]:
        return partial(create_worker, self._params, self._data.get_worker_cls())

    def generate_input(self):
        if self._shuffle:
            random.shuffle(self.filenames)
        for filename in self.filenames:
            raw_data = self._load_file(filename)
            if self._shuffle:
                random.shuffle(raw_data)
            for element in raw_data:
                yield element

    @staticmethod
    def _load_file(filename):
        """load one data file .txt (conll-format) or .json into a nested list"""
        if filename.endswith(".txt"):
            training_data = (load_txt_conll(filename))
        elif filename.endswith(".json"):
            with open(filename) as f:
                training_data = json.load(f)
        else:
            raise IOError(f"Invalid file extension in: '{filename}', only '.txt' and '.json' is supported")
        return training_data
