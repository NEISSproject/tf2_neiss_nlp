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
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from typing import TYPE_CHECKING

import tensorflow as tf
import tfaip_scenario.nlp.util.transformer as transformers
from paiargparse import pai_dataclass
from tfaip import Sample
from tfaip.model.graphbase import GraphBase
from tfaip.util.tftyping import AnyTensor
from tfaip_scenario.nlp.bert_pretraining.mlm.model import ModelMLMParams, MLMModel

if TYPE_CHECKING:
    from tfaip_scenario.nlp.data.nsp import NSPData


@pai_dataclass
@dataclass
class ModelNSPParams(ModelMLMParams):
    @staticmethod
    def cls():
        return NSPModel

    def graph_cls(self):
        return globals()[self.model]

    model: str = "BERTNSP"


class NSPModel(MLMModel):
    @staticmethod
    def params_cls():
        return ModelNSPParams

    def _loss(self, inputs, targets, outputs) -> Dict[str, AnyTensor]:
        def loss_fn(args):
            # logits_ = args[1] * args[2]
            tags_nsp = args[2]
            tags_nsp = tags_nsp[:, 0]
            nsp_logits = tf.reduce_mean(tf.transpose(args[3], [0, 2, 1]), axis=-1)
            res_mlm = tf.losses.sparse_categorical_crossentropy(
                y_true=args[0], y_pred=args[1], from_logits=True
            )
            res_nsp = tf.expand_dims(
                tf.losses.sparse_categorical_crossentropy(
                    y_true=tags_nsp, y_pred=nsp_logits, from_logits=True
                ),
                -1,
            )
            # res = res * args[2]
            return res_mlm + res_nsp

        return {
            "softmax_cross_entropy": loss_fn(
                (
                    targets["tgt_mlm"],
                    outputs["logits_mlm"],
                    targets["tgt_nsp"],
                    outputs["logits_nsp"],
                )
            )
        }

    def _target_output_metric(self) -> List[Tuple[str, str, tf.keras.metrics.Metric]]:
        metrics = super()._target_output_metric()
        return metrics + [
            ("tgt_nsp", "pred_ids_nsp", MyAccuracyNSP(name="accuracy_nsp"))
        ]

    def _sample_weights(self, inputs, targets) -> Dict[str, Any]:
        dict = super()._sample_weights(inputs, targets)
        dict["accuracy_nsp"] = tf.cast(tf.math.not_equal(inputs["text"], 0), tf.float32)
        return dict

    def _print_evaluate(self, sample: Sample, data: "NSPData", print_fn):
        inputs, outputs, targets = sample.inputs, sample.outputs, sample.targets
        sentence = inputs["text"]
        pred_mlm = outputs["pred_ids_mlm"]
        preds_nsp = outputs["pred_ids_nsp"]
        tgt = targets["tgt_mlm"]
        mask = targets["mask_mlm"]
        tgt_nsp = targets["tgt_nsp"]
        tokens_str, mask_str, tags_str, preds_str, preds_nsp_str = data.print_sentence(
            sentence, mask, tgt, tgt_nsp, pred_mlm, preds_nsp=preds_nsp
        )

        print_fn(
            f"\n"
            f"in:  {tokens_str}\n"
            f"mask:{mask_str}\n"
            f"tgt: {tags_str}\n"
            f"pred:{preds_str}\n"
            f"{preds_nsp_str}"
        )


class BERTNSP(GraphBase[ModelNSPParams]):
    def __init__(self, params: ModelNSPParams, name="model", **kwargs):
        super(BERTNSP, self).__init__(params, name=name, **kwargs)

        self._vocab_size = params.target_vocab_size

        self.bert = transformers.BERT(params)

        self._last_layer_mlm = tf.keras.layers.Dense(self._vocab_size)
        self._last_layer_nsp = tf.keras.layers.Dense(2)

    def build_graph(self, inputs, training=None):
        bert_out = self.bert(
            inputs, training=training
        )  # (batch_size, inp_seq_len, d_model)

        mlm_logits = self._last_layer_mlm(
            bert_out["enc_output"]
        )  # (batch_size, tar_seq_len, target_vocab_size)
        nsp_logits = self._last_layer_nsp(
            bert_out["enc_output"]
        )  # (batch_size, tar_seq_len, 2)
        mlm_pred_ids = tf.argmax(input=mlm_logits, axis=2, output_type=tf.int32)
        nsp_pred_ids = tf.argmax(input=nsp_logits, axis=2, output_type=tf.int32)
        self._graph_out = {
            "pred_ids_mlm": mlm_pred_ids,
            "logits_mlm": mlm_logits,
            "enc_output": bert_out["enc_output"],
            "pred_ids_nsp": nsp_pred_ids,
            "logits_nsp": nsp_logits,
        }

        return self._graph_out


class MyAccuracyNSP(tf.keras.metrics.Accuracy):
    def __init__(self, **kwargs):
        super(MyAccuracyNSP, self).__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        float_res = tf.reduce_sum(
            tf.cast(y_pred, tf.float32) * tf.cast(sample_weight, tf.float32), axis=-1
        ) / tf.cast(tf.reduce_sum(sample_weight, axis=-1), tf.float32)
        nsp_pred_ids = tf.cast(tf.round(float_res), tf.int32)

        super(MyAccuracyNSP, self).update_state(
            y_true=y_true[:, 0], y_pred=nsp_pred_ids
        )
