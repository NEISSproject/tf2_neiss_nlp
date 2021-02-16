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
from typing import Any, Dict
from typing import TYPE_CHECKING

import tensorflow as tf
import tensorflow.keras as keras
import tfneissnlp.util.transformer as transformers
from dataclasses_json import dataclass_json
from tfaip.base.model import ModelBaseParams, ModelBase, GraphBase
from tfaip.base.model.modelbase import SimpleMetric
from tfaip.util.typing import AnyNumpy

if TYPE_CHECKING:
    from tfneissnlp.data.mlm import MLMData


@dataclass_json
@dataclass
class ModelMLMParams(ModelBaseParams):
    model: str = 'BERTMLM'
    num_layers: int = 6
    d_model: int = 512
    num_heads: int = 8
    dff: int = 2048
    rate: float = 0.1
    rel_pos_enc: bool = True
    pos_enc_max_abs: int = 20000
    pos_enc_max_rel: int = 16
    hidden_activation: str = 'relu'
    target_vocab_size_: int = None
    whole_word_attention_: bool = False
    one_side_attention_window: int = 5


class Model(ModelBase):
    @staticmethod
    def get_params_cls():
        return ModelMLMParams

    def __init__(self, params: ModelMLMParams, *args, **kwargs):
        super(Model, self).__init__(params, *args, **kwargs)
        self.model_class = self._get_model_class()

    def _get_model_class(self):
        return globals()[self._params.model]

    def create_graph(self, params):
        return self.model_class(params)

    def _best_logging_settings(self):
        return "max", "accuracy_mlm"

    def _loss(self, inputs, outputs):
        def loss_fn(args):
            res = tf.losses.sparse_categorical_crossentropy(y_true=args[0], y_pred=args[1], from_logits=True)
            return res

        return {"softmax_cross_entropy": keras.layers.Lambda(loss_fn, name="softmax_cross_entropy")(
            (inputs['tgt_mlm'], outputs['logits_mlm']))}

    def sample_weights(self, inputs: Dict[str, tf.Tensor], targets: Dict[str, tf.Tensor]) -> Dict[str, Any]:
        return {
            "softmax_cross_entropy": inputs['mask_mlm'],
            "accuracy_mlm": inputs['mask_mlm'],
        }

    def _metric(self) -> Dict[str, SimpleMetric]:
        return {
            "accuracy_mlm": SimpleMetric('tgt_mlm', 'pred_ids_mlm', tf.keras.metrics.Accuracy(name="accuracy_mlm"))
        }

    @staticmethod
    def _get_additional_layers():
        return [BERTMLM]

    def _print_evaluate(self, inputs: Dict[str, AnyNumpy], outputs: Dict[str, AnyNumpy], targets: Dict[str, AnyNumpy],
                        data: 'MLMData', print_fn):
        sentence = inputs['text']
        pred = outputs['pred_ids_mlm']
        tgt = targets['tgt_mlm']
        mask = inputs['mask_mlm']
        tokens_str, tags_str, mask_str, preds_str = data.print_sentence(sentence, tgt, mask, pred)

        print_fn(f'\n'
                 f'in:  {tokens_str}\n'
                 f'mask:{mask_str}\n'
                 f'tgt: {tags_str}\n'
                 f'pred:{preds_str}\n')


class BERTMLM(GraphBase):
    def __init__(self, params: ModelMLMParams, name='model', **kwargs):
        super(BERTMLM, self).__init__(params, name=name, **kwargs)

        self._vocab_size = params.target_vocab_size_

        self.bert = transformers.BERT(params)

        self._last_layer = tf.keras.layers.Dense(self._vocab_size)

    @classmethod
    def params_cls(cls):
        return ModelMLMParams

    def call(self, inputs, **kwargs):
        bert_out = self.bert(inputs, **kwargs)  # (batch_size, inp_seq_len, d_model)
        final_output = self._last_layer(
            bert_out['enc_output'])  # (batch_size, tar_seq_len, target_vocab_size)
        pred_ids = tf.argmax(input=final_output, axis=2, output_type=tf.int32)
        graph_out = {"pred_ids_mlm": pred_ids,
                     'logits_mlm': final_output,
                     }
        return graph_out
