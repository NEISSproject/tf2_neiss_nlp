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
import os
from abc import abstractmethod

from tensorflow.keras.backend import clear_session

from tfaip.util.testing.workdir import workdir_path
from tfaip.resource.resource import Resource


def set_test_trainer_params(p):
    p.random_seed = 123
    p.gen.setup.train.batch_size = 2
    p.gen.setup.train.prefetch = 1
    p.gen.setup.train.num_processes = 1
    p.gen.setup.val.batch_size = 2
    p.gen.setup.val.prefetch = 1
    p.gen.setup.val.num_processes = 1
    data = p.scenario.data
    data.tokenizer = Resource(workdir_path(__file__, "data", "tokenizer", "tokenizer_de.subwords"))

    model = p.scenario.model
    model.d_model = 2
    model.dff = 2
    model.num_layers = 1
    model.num_heads = 2
    return p


class AbstractTestNLPData(object):
    def setUp(self) -> None:
        os.chdir(workdir_path(__file__))

    def tearDown(self) -> None:
        clear_session()

    def data_loading(self, data_cls, trainer_params):
        data = data_cls(trainer_params.scenario.data)
        with trainer_params.gen.train_data(data).generate_input_samples() as samples:
            train_data = next(samples)
        with trainer_params.gen.train_data(data).generate_input_batches() as batches:
            train_data = next(batches)
        with trainer_params.gen.val_data(data).generate_input_batches() as batches:
            val_data = next(batches)
        self.check_batch_content(train_data, trainer_params)
        self.check_batch_content(val_data, trainer_params)

    @abstractmethod
    def check_batch_content(self, batch, trainer_params):
        raise NotImplementedError
