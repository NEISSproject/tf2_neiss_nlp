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
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from typing import TYPE_CHECKING

import tensorflow as tf
from paiargparse import pai_dataclass

import tfaip_scenario.nlp.util.transformer as transformers
from tfaip import ModelBaseParams
from tfaip import Sample
from tfaip.model.graphbase import GraphBase
from tfaip.model.modelbase import ModelBase
from tfaip.util.tftyping import AnyTensor

if TYPE_CHECKING:
    from tfaip_scenario.nlp.data.mlm import MLMData


@pai_dataclass
@dataclass
class ModelMLMParams(ModelBaseParams):
    @staticmethod
    def cls():
        return MLMModel

    def graph_cls(self):
        return globals()[self.model]

    model: str = "BERTMLM"
    num_layers: int = 6
    d_model: int = 512
    num_heads: int = 8
    dff: int = 2048
    rate: float = 0.1
    rel_pos_enc: bool = True
    pos_enc_max_abs: int = 20000
    pos_enc_max_rel: int = 16
    hidden_activation: str = "relu"
    target_vocab_size: int = None
    whole_word_attention_: bool = False
    one_side_attention_window: int = 5


class MLMModel(ModelBase[ModelMLMParams]):
    def _best_logging_settings(self):
        return "max", "accuracy_mlm"

    def _loss(self, inputs, targets, outputs) -> Dict[str, AnyTensor]:
        def loss_fn(args):
            res = tf.losses.sparse_categorical_crossentropy(y_true=args[0], y_pred=args[1], from_logits=True)
            return res

        return {"softmax_cross_entropy": loss_fn((targets["tgt_mlm"], outputs["logits_mlm"]))}

    def _sample_weights(self, inputs: Dict[str, tf.Tensor], targets: Dict[str, tf.Tensor]) -> Dict[str, Any]:
        return {
            "softmax_cross_entropy": targets["mask_mlm"],
            "accuracy_mlm": targets["mask_mlm"],
        }

    def _target_output_metric(self) -> List[Tuple[str, str, tf.keras.metrics.Metric]]:
        return [("tgt_mlm", "pred_ids_mlm", tf.keras.metrics.Accuracy(name="accuracy_mlm"))]

    def _print_evaluate(self, sample: Sample, data: "MLMData", print_fn):
        inputs, outputs, targets = sample.inputs, sample.outputs, sample.targets
        sentence = inputs["text"]
        pred = outputs["pred_ids_mlm"]
        tgt = targets["tgt_mlm"]
        mask = targets["mask_mlm"]
        tokens_str, tags_str, mask_str, preds_str = data.print_sentence(sentence, tgt, mask, pred)

        print_fn(f"\n" f"in:  {tokens_str}\n" f"mask:{mask_str}\n" f"tgt: {tags_str}\n" f"pred:{preds_str}\n")

    def _export_graphs(
        self,
        inputs: Dict[str, tf.Tensor],
        outputs: Dict[str, tf.Tensor],
        targets: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.keras.Model]:
        default = super()._export_graphs(inputs, outputs, targets)
        bert_model = default["default"]  # Model -> PredictionModel -> RootModel -> Bert
        root_graph = bert_model.layers[-1]
        encoder_only = root_graph.graph.bert
        default["encoder_only"] = encoder_only
        return default


class BERTMLM(GraphBase):
    def __init__(self, params: ModelMLMParams, name="model", **kwargs):
        super(BERTMLM, self).__init__(params, name=name, **kwargs)

        self._vocab_size = params.target_vocab_size

        self.bert = transformers.BERT(params)

        self._last_layer = tf.keras.layers.Dense(self._vocab_size)

    @classmethod
    def params_cls(cls):
        return ModelMLMParams

    def build_graph(self, inputs, training=None):
        bert_out = self.bert(inputs, training=training)  # (batch_size, inp_seq_len, d_model)
        final_output = self._last_layer(bert_out["enc_output"])  # (batch_size, tar_seq_len, target_vocab_size)
        pred_ids = tf.argmax(input=final_output, axis=2, output_type=tf.int32)
        graph_out = {
            "pred_ids_mlm": pred_ids,
            "logits_mlm": final_output,
            "enc_output": bert_out["enc_output"],
        }
        return graph_out
