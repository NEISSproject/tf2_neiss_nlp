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
import logging
from dataclasses import dataclass
from typing import Any, Dict
from typing import TYPE_CHECKING

import tensorflow as tf
import tensorflow.keras as keras
from dataclasses_json import dataclass_json

import tfneissnlp.util.transformer as transformers
from tfaip.base.model import ModelBaseParams, ModelBase, GraphBase
from tfaip.base.model.modelbase import SimpleMetric
from tfneissnlp.data.ner import NERData
from tfneissnlp.util.ner_metrics import EntityF1, EntityPrecision, EntityRecall
from tfaip.util.typing import AnyNumpy

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class ModelParams(ModelBaseParams):
    # Bert-Params
    model: str = 'NERwithMiniBERT'
    num_layers: int = 6
    d_model: int = 512
    num_heads: int = 8
    dff: int = 512
    pos_enc_max_abs: int = 20000
    pos_enc_max_rel: int = 16
    rel_pos_enc: bool = False
    rate: float = 0.1
    bert_graph: str = "BERT"
    pretrained_bert: str = ""
    num_tags_: int = None
    tags_fn_: str = None
    target_vocab_size_: int = None
    oov_id_: int = None


class Model(ModelBase):
    @staticmethod
    def get_params_cls():
        return ModelParams

    def __init__(self, params: ModelParams, *args, **kwargs):
        super(Model, self).__init__(params, *args, **kwargs)
        self.model_class = globals()[params.model]

    def create_graph(self, params):
        return self.model_class(params)

    def _best_logging_settings(self):
        # return "min", "val_loss"
        return "max", "EntityF1"

    def _loss(self, inputs, outputs):
        def _loss_fn(args):
            res = tf.losses.sparse_categorical_crossentropy(y_true=tf.cast(args[0], tf.float32),
                                                            y_pred=args[1], from_logits=False)
            return res

        return {"softmax_cross_entropy": keras.layers.Lambda(_loss_fn, name="softmax_cross_entropy")(
            (inputs['tgt'], outputs['logits']))}

    @staticmethod
    def _get_additional_layers():
        return [NERwithMiniBERT, EntityRecall, EntityF1, EntityPrecision]

    def _metric(self):
        return {'simple_accuracy': SimpleMetric('tgt', 'pred_ids', tf.keras.metrics.Accuracy(name='simple_accuracy')),
                'no_oov_accuracy': SimpleMetric('tgt', 'pred_ids', AccuracyTokens(self._params.oov_id_, name='no_oov_accuracy')),
                'simple_precision': SimpleMetric('tgt', 'pred_ids',
                                                 PrecisionTokens(self._params.oov_id_, name='simple_precision')),
                'simple_recall': SimpleMetric('tgt', 'pred_ids', RecallTokens(self._params.oov_id_, name='simple_recall')),
                'simple_F1': SimpleMetric('tgt', 'pred_ids', MyF1Tokens(self._params.oov_id_, name='simple_F1')),
                'EntityF1': SimpleMetric('tgt', 'pred_ids', EntityF1(self._params.tags_fn_, name='EntityF1')),
                'PrecisionF1': SimpleMetric('tgt', 'pred_ids',
                                            EntityPrecision(self._params.tags_fn_, name='EntityPrecision')),
                'EntityRecall': SimpleMetric('tgt', 'pred_ids',
                                             EntityRecall(self._params.tags_fn_, name='EntityRecall')),
                }

    def _sample_weights(self, inputs, targets) -> Dict[str, Any]:
        return {'simple_accuracy': targets['targetmask'],
                'no_oov_accuracy': targets['targetmask'],
                'simple_precision': targets['targetmask'],
                'simple_recall': targets['targetmask'],
                'simple_F1': targets['targetmask'],
                }

    def build(self, inputs_targets: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        outputs = super(Model, self).build(inputs_targets)

        if self._params.pretrained_bert:
            self.init_new_training()

        return outputs

    def init_new_training(self):
        target_model = self._graph.pretrained_bert
        variable_names_ckpt = [x[0] for x in
                               tf.train.list_variables(tf.train.latest_checkpoint(self._params.pretrained_bert))]
        new_variable_names_ckpt = [x.replace("enc_layers/", "enc_layers_").replace('_tracked_layers', 'BERTMini')
                                   .replace("/bert/", "/")
                                   .replace("rel_pos_lookup", "scaled_dot_relative_attention/embedding")
                                   for x in variable_names_ckpt]
        if str(target_model.variables[0].name).startswith("keras_debug_model/"):
            logger.info("fix debug_model names")
            new_variable_names_ckpt = ["keras_debug_model/" + x for x in new_variable_names_ckpt]
        mapping = {new_var: old_name for old_name, new_var in zip(variable_names_ckpt, new_variable_names_ckpt)}
        to_load = []
        # print("### model vars ###")
        # for y in target_model.variables:
        #     print(y.name)
        # print("### checkpoint vars ###")
        # for x in variable_names_ckpt:
        #     print(x)

        for variable in target_model.variables:
            to_load.append(tf.train.load_variable(tf.train.latest_checkpoint(self._params.pretrained_bert),
                                                  mapping[variable.name[:-2] + '/.ATTRIBUTES/VARIABLE_VALUE']))
        target_model.set_weights(to_load)

    def _print_evaluate(self, inputs: Dict[str, AnyNumpy], outputs: Dict[str, AnyNumpy], targets: Dict[str, AnyNumpy],
                        data: NERData, print_fn):
        sentence = inputs['sentence']
        pred = outputs['pred_ids']
        tgt = targets['tgt']
        mask = targets['targetmask']
        tokens_str, tags_str, mask_str, preds_str = data.print_ner_sentence(sentence, tgt, mask, pred)

        f1_metric = EntityF1(self._params.tags_fn_, name='EntityF1_print')
        f1_metric.update_state(tf.expand_dims(tgt, axis=0), tf.expand_dims(pred, axis=0))
        f1 = f1_metric.result()
        f1_metric.reset_states()
        print_fn(f'\n'
                 f'in:  {tokens_str}\n'
                 f'mask:{mask_str}\n'
                 f'tgt: {tags_str}\n'
                 f'pred:{preds_str}\n'
                 f'F1: {f1}')


class NERwithMiniBERT(GraphBase):
    def __init__(self, params, name='model', **kwargs):
        super(NERwithMiniBERT, self).__init__(params, name=name, **kwargs)
        self._params.target_vocab_size = params.num_tags_ + 2
        self._tracked_layers = dict()
        self.pretrained_bert = getattr(transformers, self._params.bert_graph)(self._params)
        self._last_layer = tf.keras.layers.Dense(self._params.target_vocab_size,
                                                                   activation=tf.keras.activations.softmax,
                                                                   name="last_layer")
        self._softmax = tf.keras.layers.Softmax()

    @classmethod
    def params_cls(cls):
        return ModelParams

    def call(self, inputs, **kwargs):
        inp = dict()
        inp["text"] = inputs["sentence"]
        bert_graph_out = self.pretrained_bert(inp, **kwargs)
        final_output = self._last_layer(bert_graph_out["enc_output"])  # (batch_size, tar_seq_len, target_vocab_size)
        pred_ids = tf.argmax(input=final_output, axis=2, output_type=tf.int32)
        probabilities = self._softmax(final_output)
        return {"pred_ids": pred_ids, 'probabilities': probabilities, 'logits': final_output}


class AccuracyTokens(tf.keras.metrics.Accuracy):
    def __init__(self, oov_id, **kwargs):
        super(AccuracyTokens, self).__init__(**kwargs)
        self.oov_id = oov_id

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_positive_indexes = tf.cast(tf.raw_ops.Equal(x=y_true, y=y_pred), tf.int32)
        not_oov = tf.cast(tf.raw_ops.NotEqual(x=y_pred, y=self.oov_id), tf.int32)
        booth_oov = tf.raw_ops.LogicalAnd(x=tf.raw_ops.Equal(x=y_true, y=self.oov_id),
                                          y=tf.raw_ops.Equal(x=y_pred, y=self.oov_id))
        no_oov_weights = tf.cast(tf.raw_ops.LogicalNot(x=booth_oov), tf.int32) * sample_weight
        super(AccuracyTokens, self).update_state(y_true=true_positive_indexes,
                                                 y_pred=not_oov,
                                                 sample_weight=no_oov_weights)


class PrecisionTokens(tf.keras.metrics.Precision):
    def __init__(self, oov_id, **kwargs):
        super(PrecisionTokens, self).__init__(**kwargs)
        self.oov_id = oov_id

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_positive_indexes = tf.cast(tf.raw_ops.Equal(x=y_true, y=y_pred), tf.int32)
        not_oov = tf.cast(tf.raw_ops.NotEqual(x=y_pred, y=self.oov_id), tf.int32)
        booth_oov = tf.raw_ops.LogicalAnd(x=tf.raw_ops.Equal(x=y_true, y=self.oov_id),
                                          y=tf.raw_ops.Equal(x=y_pred, y=self.oov_id))
        no_oov_weights = tf.cast(tf.raw_ops.LogicalNot(x=booth_oov), tf.int32) * sample_weight
        super(PrecisionTokens, self).update_state(y_true=true_positive_indexes,
                                                  y_pred=not_oov,
                                                  sample_weight=no_oov_weights)


class RecallTokens(tf.keras.metrics.Recall):
    def __init__(self, oov_id, **kwargs):
        super(RecallTokens, self).__init__(**kwargs)
        self.oov_id = oov_id

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_positive_indexes = tf.cast(tf.raw_ops.Equal(x=y_true, y=y_pred), tf.int32)
        not_oov = tf.cast(tf.raw_ops.NotEqual(x=y_true, y=self.oov_id), tf.int32)
        booth_oov = tf.raw_ops.LogicalAnd(x=tf.raw_ops.Equal(x=y_true, y=self.oov_id),
                                          y=tf.raw_ops.Equal(x=y_pred, y=self.oov_id))
        no_oov_weights = tf.cast(tf.raw_ops.LogicalNot(x=booth_oov), tf.int32) * sample_weight
        super(RecallTokens, self).update_state(y_true=not_oov,
                                               y_pred=true_positive_indexes,
                                               sample_weight=no_oov_weights)


class MyF1Tokens(tf.keras.metrics.Mean):
    def __init__(self, oov_id, **kwargs):
        super(MyF1Tokens, self).__init__(**kwargs)
        self.oov_id = oov_id
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_positive_indexes = tf.cast(tf.raw_ops.Equal(x=y_true, y=y_pred), tf.int32)
        not_oov_tgt = tf.cast(tf.raw_ops.NotEqual(x=y_true, y=self.oov_id), tf.int32)
        not_oov_pred = tf.cast(tf.raw_ops.NotEqual(x=y_pred, y=self.oov_id), tf.int32)

        booth_oov = tf.raw_ops.LogicalAnd(x=tf.raw_ops.Equal(x=y_true, y=self.oov_id),
                                          y=tf.raw_ops.Equal(x=y_pred, y=self.oov_id))
        no_oov_weights = tf.cast(tf.raw_ops.LogicalNot(x=booth_oov), tf.int32) * sample_weight
        self.precision.update_state(true_positive_indexes, not_oov_pred, no_oov_weights)
        self.recall.update_state(not_oov_tgt, true_positive_indexes, no_oov_weights)

    def result(self):
        return 2 * self.precision.result() * self.recall.result() / tf.maximum(
            self.precision.result() + self.recall.result(), tf.keras.backend.epsilon())

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()
