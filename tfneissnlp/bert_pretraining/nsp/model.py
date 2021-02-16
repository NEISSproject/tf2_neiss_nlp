# Copyright 2020 The tfaip authors. All Rights Reserved.
#
# This file is part of tfaip.
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
from dataclasses import dataclass
from typing import Dict
from typing import TYPE_CHECKING

import tensorflow as tf
import tensorflow.keras as keras
import tfneissnlp.util.transformer as transformers
from dataclasses_json import dataclass_json
from tfaip.base.model import GraphBase
from tfaip.base.model.modelbase import SimpleMetric
from tfaip.util.typing import AnyNumpy
from tfneissnlp.bert_pretraining.mlm.model import ModelMLMParams, Model as ModelMLM
from tfneissnlp.data.nsp import NSPData

if TYPE_CHECKING:
    pass


@dataclass_json
@dataclass
class ModelParams(ModelMLMParams):
    model: str = 'BERTNSP'


class Model(ModelMLM):
    @staticmethod
    def get_params_cls():
        return ModelParams

    def _loss(self, inputs, outputs):
        def loss_fn(args):
            # logits_ = args[1] * args[2]
            tags_nsp = args[2]
            tags_nsp = tags_nsp[:, 0]
            nsp_logits=tf.reduce_mean(tf.transpose(args[3],[0,2,1]),axis=-1)
            res_mlm = tf.losses.sparse_categorical_crossentropy(y_true=args[0], y_pred=args[1], from_logits=True)
            res_nsp = tf.expand_dims(tf.losses.sparse_categorical_crossentropy(y_true=tags_nsp, y_pred=nsp_logits, from_logits=True),-1)
            # res = res * args[2]
            return res_mlm+res_nsp

        return {"softmax_cross_entropy": keras.layers.Lambda(loss_fn, name="softmax_cross_entropy")(
            (inputs['tgt_mlm'], outputs['logits_mlm'],inputs['tgt_nsp'], outputs['logits_nsp']))}

    def _get_model_class(self):
        return globals()[self._params.model]

    def _metric(self) -> Dict[str, SimpleMetric]:
        metrics=super(Model,self)._metric()
        metrics["accuracy_nsp"]=SimpleMetric('tgt_nsp', 'pred_ids_nsp', MyAccuracyNSP(name="accuracy_nsp"))

        return metrics

    def _print_evaluate(self, inputs: Dict[str, AnyNumpy], outputs: Dict[str, AnyNumpy], targets: Dict[str, AnyNumpy],
                        data: NSPData, print_fn):
        pass


class BERTNSP(GraphBase):
    def __init__(self, params: ModelParams, name='model', **kwargs):
        super(BERTNSP, self).__init__(params, name=name, **kwargs)

        self._vocab_size = params.token_size_

        self.bert = transformers.BERT(params)

        self._last_layer_mlm = tf.keras.layers.Dense(self._vocab_size)
        self._last_layer_nsp = tf.keras.layers.Dense(2)

    def call(self, inputs, **kwargs):
        bert_out = self.bert(inputs, **kwargs)  # (batch_size, inp_seq_len, d_model)

        mlm_logits = self._last_layer_mlm(
            bert_out['enc_output'])  # (batch_size, tar_seq_len, target_vocab_size)
        nsp_logits = self._last_layer_nsp(bert_out['enc_output'])  # (batch_size, tar_seq_len, 2)
        mlm_pred_ids = tf.argmax(input=mlm_logits, axis=2, output_type=tf.int32)
        nsp_pred_ids = tf.argmax(input=nsp_logits, axis=2, output_type=tf.int32)
        self._graph_out = {"pred_ids_mlm": mlm_pred_ids, 'logits_mlm': mlm_logits, "pred_ids_nsp": nsp_pred_ids,
                           'logits_nsp': nsp_logits}

        return self._graph_out

class MyAccuracyNSP(tf.keras.metrics.Accuracy):
    def __init__(self, **kwargs):
        super(MyAccuracyNSP, self).__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        nsp_pred_ids=tf.cast(tf.round(tf.reduce_mean(tf.cast(y_pred,tf.float32),axis=-1)),tf.int32)
        super(MyAccuracyNSP, self).update_state(y_true=y_true[:, 0],
                                             y_pred=nsp_pred_ids,
                                             sample_weight=sample_weight)
