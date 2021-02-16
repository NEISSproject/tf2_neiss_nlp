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
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import tfneissnlp.util.transformer as transformers
from dataclasses_json import dataclass_json
from tfaip.base.model import ModelBaseParams, ModelBase, GraphBase
from tfaip.base.model.modelbase import SimpleMetric
from tfaip.util.typing import AnyNumpy
from tfneissnlp.data.ner import NERData
from tfneissnlp.util.ner_eval import EntityF1, EntityPrecision, EntityRecall, SeqEvalF1, \
    SeqEvalF1FP, ClassF1, StartF1, EndF1, BetMetricWrapper, FixRuleMetricWrapper
from tfneissnlp.util.ner_eval_token import MyF1Tokens
from tfneissnlp.util.stringmapper import get_sm
from transformers import TFBertModel, BertConfig

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
    dff: int = 2048
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
    use_hf_model_: bool = False
    pretrained_hf_model_: str = ""
    whole_word_attention_: bool = False
    one_side_attention_window: int = 5
    hidden_activation: str = 'relu'
    use_crf: bool = False
    use_ner_loss: bool = False

    use_entity_loss: bool = False
    use_entity_loss_max: float = 2.0
    dropout_last: float = 0.0
    bet_tagging_: bool = None
    loss_se_weight: float = 1.0
    loss_se_boost: float = 5.0
    feasible_pred_ids: bool = False
    crf_with_ner_rule: bool = False


class Model(ModelBase):
    @staticmethod
    def get_params_cls():
        return ModelParams

    def __init__(self, params: ModelParams, *args, **kwargs):
        super(Model, self).__init__(params, *args, **kwargs)
        self.model_class = globals()[params.model]
        self._tag_string_mapper = get_sm(self._params.tags_fn_)
        self._real_tag_num = self._tag_string_mapper.get_oov_id() // 2

    def create_graph(self, params):
        return self.model_class(params)

    def _best_logging_settings(self):
        # return "min", "val_loss"
        return "max", "SeqEvalF1FixRule"

    def calc_penalty_for_tag_pair(self, probs1, probs2):
        tprobs1 = tf.transpose(probs1)
        bvec1 = tf.transpose(tprobs1[0:self._real_tag_num])  # probs of 'B-'-tags of prob1
        ivec1 = tf.transpose(tprobs1[self._real_tag_num:2 * self._real_tag_num])  # probs of 'I-'-tags of prob1
        ivec2 = tf.transpose(
            tf.transpose(probs2)[self._real_tag_num:2 * self._real_tag_num])  # probs of 'I-'-tags of prob2
        oprob = tf.transpose(tprobs1[-3])  # prob of 'O'-tag of prob1
        sprob = tf.transpose(tprobs1[-2])  # prob of SOS-tag of prob1
        eprob = tf.transpose(tprobs1[-1])  # prob of EOS-tag of prob1
        ivec_sum = tf.reduce_sum(ivec2, axis=-1)  # sum of 'I-'-tag probs of prob2
        bpen = tf.reduce_sum(bvec1 * (ivec_sum[:, tf.newaxis] - ivec2),
                             axis=-1)  # sum of penaltys for every b-tag: after every b-tag every i-tag is not allowed except the corresponding
        ipen = tf.reduce_sum(ivec1 * (ivec_sum[:, tf.newaxis] - ivec2),
                             axis=-1)  # sum of penaltys for every i-tag: after every i-tag every other i-tag is not allowed
        open = oprob * ivec_sum  # penalty for other tag: after an 'O'-tag every i-tag is not allowed
        spen = sprob * ivec_sum  # penalty for sos tag: after an SOS-tag every i-tag is not allowed
        epen = eprob  # penalty for eos tag: after an EOS-tag no tag is allowed
        return bpen + ipen + open + spen + epen

    def calc_penalty_tag_sequence(self, probs, seq_length):
        tprobs = tf.transpose(probs, perm=[1, 0, 2])
        prob1 = tprobs[:-1]
        prob2 = tprobs[1:]
        mask = tf.cast(tf.sequence_mask(seq_length - 1), tf.float32)

        def calc_penalty_pair(element):
            prob1, prob2 = element
            return self.calc_penalty_for_tag_pair(prob1, prob2)

        pair_penalty = tf.transpose(tf.map_fn(calc_penalty_pair, (prob1, prob2), fn_output_signature=tf.float32),
                                    perm=[1, 0]) * mask
        dummy_pair_penalty = pair_penalty[:, 0] * 0.0
        dummy_pair_penalty = dummy_pair_penalty[:, tf.newaxis]
        result = tf.concat([dummy_pair_penalty, pair_penalty], -1) + tf.concat([pair_penalty, dummy_pair_penalty], -1)
        return result

    def _loss_weights(self) -> Optional[Dict[str, float]]:
        if self._params.bet_tagging_:
            return {"softmax_cross_entropy_cls": 1.0,
                    "mae_start": self._params.loss_se_weight,
                    "mae_end": self._params.loss_se_weight}

    def _loss_bet(self, inputs, outputs):
        return_dict = {}

        def _loss_cls(args):
            res = tf.losses.sparse_categorical_crossentropy(y_true=tf.cast(args[0], tf.float32),
                                                            y_pred=args[1], from_logits=False)
            return res

        def _loss_se(args):
            # tf.print(args[1])
            y_true = tf.cast(args[0], tf.float32)
            y_pred = args[1]
            # tf.print(y_pred, summarize=1000)
            mask = tf.cast(args[2], dtype=tf.float32)
            # drift = tf.abs(tf.reduce_sum(y_true * mask) - tf.reduce_sum(y_pred * mask))
            res = tf.abs(y_true - y_pred) * mask
            res += tf.abs(y_true - y_pred) * mask * y_true * self._params.loss_se_boost

            # tf.print(tf.shape(res))
            # tf.print(res[0])
            return res

        inputs_cls = tf.gather(inputs['tgt_cse'], 0, axis=-1)
        inputs_start = tf.gather(inputs['tgt_cse'], 1, axis=-1)
        inputs_end = tf.gather(inputs['tgt_cse'], 2, axis=-1)

        # print(self._tag_string_mapper.size())
        # print(inputs_start, outputs['probabilities_start'])

        return_dict["softmax_cross_entropy_cls"] = keras.layers.Lambda(_loss_cls, name="softmax_cross_entropy_cls")(
            (inputs_cls - self._tag_string_mapper.size() // 2, outputs['probabilities_cls']))
        return_dict["mae_start"] = keras.layers.Lambda(_loss_se, name="mae_start")(
            (inputs_start, outputs['probabilities_start'], inputs["targetmask"]))
        return_dict["mae_end"] = keras.layers.Lambda(_loss_se, name="mae_end")(
            (inputs_end, outputs['probabilities_end'], inputs["targetmask"]))

        return return_dict

    def _loss(self, inputs, outputs):
        return_dict = {}

        if self._params.bet_tagging_:
            return self._loss_bet(inputs, outputs)

        if self._params.use_crf:
            def _loss_fn(args):
                log_likelihood, _ = tfa.text.crf_log_likelihood(args[0], args[1], args[2], args[3])
                return -log_likelihood

            return {"crf_log_likelihood": keras.layers.Lambda(_loss_fn, name="crf_log_likelihood")(
                (outputs['logits'], inputs['tgt'], inputs["seq_length"][:, 0], outputs['trans_params'][0]))}
        elif self._params.use_ner_loss:
            def _loss_fn(args):
                res = tf.losses.sparse_categorical_crossentropy(y_true=tf.cast(args[0], tf.float32),
                                                                y_pred=args[1], from_logits=False)
                ner_loss = self.calc_penalty_tag_sequence(args[1], args[2])
                return res + ner_loss

            return {
                "softm_cr_entr_with_ner_penalty": keras.layers.Lambda(_loss_fn, name="softm_cr_entr_with_ner_penalty")(
                    (inputs['tgt'], outputs['probabilities'], inputs["seq_length"][:, 0]))}
        else:
            def _loss_fn(args):
                res = tf.losses.sparse_categorical_crossentropy(y_true=tf.cast(args[0], tf.float32),
                                                                y_pred=args[1], from_logits=False)
                return res
        return_dict["softmax_cross_entropy"] = keras.layers.Lambda(_loss_fn, name="softmax_cross_entropy")(
            (inputs['tgt'], outputs['probabilities']))

        # if self._params.use_entity_loss:
        #     def _loss_fn_2(args):
        #         def softargmax(x, beta=1e10):
        #             x = tf.convert_to_tensor(x)
        #             x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
        #             return tf.reduce_sum(tf.nn.softmax(x * beta) * x_range, axis=-1)
        #
        #         def entity_loss(y_true, y_pred, num_tag_cls):
        #             """normed loss between 0 (excellent) and 1 (worst)
        #              extra penalty if correct tokens of entities are not connected
        #              num_tag_cls - number of different tags excluding SOS, EOS, Other
        #             softargmax may become unstable for many different classes?"""
        #             assert num_tag_cls % 2 == 0, "every B-tag need to have an I-tag, num-tag_cls must be even"
        #             # print("Entity Loss")
        #             # print(y_true)
        #             # print(y_pred)
        #             # print(softargmax(y_pred))
        #             good_zero = tf.minimum(tf.abs(y_true - softargmax(y_pred, beta=1e5)), 1.0)
        #
        #             other_class_mask = tf.where(tf.less(y_true, num_tag_cls), 1.0, 0.0)
        #             # tf.print("om\n", other_class_mask, summarize=1000)
        #             hit = -(- tf.ones(shape=tf.shape(y_true)) + good_zero) * other_class_mask
        #
        #
        #             # tf.print("hit\n", hit, summarize=1000)
        #             res_0 = tf.zeros(tf.shape(y_true)[0], name="res__0")
        #             res_norm = tf.zeros(tf.shape(y_true)[0], name="res_norm")
        #             # print(res_0)
        #             buffer_0 = tf.zeros(tf.shape(y_true)[0], name="buffer__0")
        #             buffer_norm = tf.zeros(tf.shape(y_true)[0], name="buffer_norm")
        #             seq_pos = tf.constant(0, shape=[])
        #
        #             def cond(b, res, buffer, res_norm, buffer_norm):
        #                 return tf.less(b, tf.shape(y_true)[-1])
        #
        #             def loop(b, res_, buffer, res_norm_, buffer_norm_):
        #                 x = tf.gather(hit, b, axis=-1)
        #                 x_norm = tf.gather(other_class_mask, b, axis=-1)
        #                 buffer = tf.minimum((buffer + x) * tf.abs(x), self._params.use_entity_loss_max)
        #                 buffer_norm_ = tf.minimum((buffer_norm_ + x_norm) * tf.abs(x_norm), self._params.use_entity_loss_max)
        #                 res_ = res_ + buffer
        #                 res_norm_ = res_norm_ + buffer_norm_
        #                 return tf.add(b, 1), res_, buffer, res_norm_, buffer_norm_
        #
        #             _, res, _, res_norm, _ = tf.while_loop(cond, loop,
        #                                                    [seq_pos, res_0, buffer_0, res_norm, buffer_norm],
        #                                                    shape_invariants=[seq_pos.shape,
        #                                                                      tf.TensorShape([None]),
        #                                                                      tf.TensorShape([None]),
        #                                                                      tf.TensorShape([None]),
        #                                                                      tf.TensorShape([None])])
        #             # tf.print("res\n", res, summarize=1000)
        #             # tf.print("rn\n", res_norm, summarize=1000)
        #             res = res / tf.maximum(1.0, res_norm)
        #             return res
        #
        #         entity_loss_ = 1.0 - entity_loss(y_true=tf.cast(args[0], tf.float32), y_pred=args[1],
        #                                                        num_tag_cls=self._params.num_tags_ - 3)
        #         # tf.print(entity_loss_)
        #         return entity_loss_
        #
        # return_dict["entity_loss"] = keras.layers.Lambda(_loss_fn_2, name="entity_loss")(
        #     (inputs['tgt'], outputs['probabilities']))

        return return_dict

    @staticmethod
    def _get_additional_layers():
        return [NERwithMiniBERT, EntityRecall, EntityF1, EntityPrecision]

    def _metric(self):
        m_dict = {
            # 'simple_accuracy': SimpleMetric('tgt', 'pred_ids', tf.keras.metrics.Accuracy(name='simple_accuracy')),
            # 'no_oov_accuracy': SimpleMetric('tgt', 'pred_ids', AccuracyTokens(self._params.oov_id_, name='no_oov_accuracy')),
            # 'simple_precision': SimpleMetric('tgt', 'pred_ids', PrecisionTokens(self._params.oov_id_, name='simple_precision')),
            # 'simple_recall': SimpleMetric('tgt', 'pred_ids', RecallTokens(self._params.oov_id_, name='simple_recall')),
            'simple_F1': SimpleMetric('tgt', 'pred_ids', MyF1Tokens(self._params.oov_id_, name='simple_F1')),
            'simpleF1FixRule': SimpleMetric('tgt', 'pred_ids', FixRuleMetricWrapper(
                MyF1Tokens(self._params.oov_id_, name='simpleF1FixRule'), tags_fn=self._params.tags_fn_,
                oov_id=self._params.oov_id_, name='simpleF1FixRule')),
            'EntityF1': SimpleMetric('tgt', 'pred_ids', EntityF1(self._params.tags_fn_, name='EntityF1')),
            'EntityF1FixRule': SimpleMetric('tgt', 'pred_ids', FixRuleMetricWrapper(
                EntityF1(self._params.tags_fn_, name='EntityF1FixRule'), tags_fn=self._params.tags_fn_,
                oov_id=self._params.oov_id_)),
            # 'EntityF1FixRule2': SimpleMetric('tgt', 'pred_ids',
            #                                  EntityF1FixRule(self._params.tags_fn_, name='EntityF1FixRule2')),
            # 'EntityF1FixRuleFixRule': SimpleMetric('tgt', 'pred_ids', FixRuleMetricWrapper(
            #     EntityF1FixRule(self._params.tags_fn_, name='EntityF1FixRuleFixRule'), tags_fn=self._params.tags_fn_,
            #     oov_id=self._params.oov_id_)),
            'EntityPrecision': SimpleMetric('tgt', 'pred_ids',
                                            EntityPrecision(self._params.tags_fn_, name='EntityPrecision')),
            'EntityPrecisionFixRule': SimpleMetric('tgt', 'pred_ids', FixRuleMetricWrapper(
                EntityPrecision(self._params.tags_fn_, name='EntityPrecisionFixRule'), tags_fn=self._params.tags_fn_,
                oov_id=self._params.oov_id_)),
            'EntityRecall': SimpleMetric('tgt', 'pred_ids', EntityRecall(self._params.tags_fn_, name='EntityRecall')),
            'EntityRecallFixRule': SimpleMetric('tgt', 'pred_ids', FixRuleMetricWrapper(
                EntityRecall(self._params.tags_fn_, name='EntityRecallFixRule'), tags_fn=self._params.tags_fn_,
                oov_id=self._params.oov_id_)),
            'SeqEvalF1': SimpleMetric('tgt', 'pred_ids', SeqEvalF1(self._params.tags_fn_, name='SeqEvalF1')),
            'SeqEvalF1FixRule': SimpleMetric('tgt', 'pred_ids', FixRuleMetricWrapper(
                SeqEvalF1(self._params.tags_fn_, name='SeqEvalF1FixRule'), tags_fn=self._params.tags_fn_,
                oov_id=self._params.oov_id_)),
        }
        if self._params.feasible_pred_ids:
            if self._params.use_crf:
                m_dict['EntityF1FP'] = SimpleMetric('tgt', 'pred_idsfp',
                                                    EntityF1(self._params.tags_fn_, name='EntityF1FP'))
                m_dict['SeqEvalF1FP'] = SimpleMetric('tgt', 'pred_idsfp',
                                                     SeqEvalF1(self._params.tags_fn_, name='SeqEvalF1FP'))
            else:
                m_dict['SeqEvalF1FP'] = SimpleMetric('tgt', 'logits',
                                                     SeqEvalF1FP(self._params.tags_fn_, name='SeqEvalF1FP'))
        if self._params.bet_tagging_:
            bet_dict = {'ClassF1': SimpleMetric('tgt_cse', 'probabilities_cls',
                                                ClassF1(oov_id=self._params.oov_id_, num_tags=self._params.num_tags_,
                                                        name='ClassF1')),
                        'StartF1': SimpleMetric('tgt_cse', 'probabilities_start', StartF1(name='StartF1')),
                        'EndF1': SimpleMetric('tgt_cse', 'probabilities_end', EndF1(name='EndF1')),
                        }
            for key in m_dict:
                if m_dict[key].output == "pred_ids":
                    m_dict[key] = SimpleMetric(m_dict[key].target, "probabilities_cse",
                                               BetMetricWrapper(m_dict[key].metric, tags_fn=self._params.tags_fn_))
                else:
                    del m_dict[key]

            m_dict = {**m_dict, **bet_dict}
        return m_dict

    def _sample_weights(self, inputs, targets) -> Dict[str, Any]:

        sw_dict = {
            # 'simple_accuracy': targets['targetmask'],
            # 'no_oov_accuracy': targets['targetmask'],
            # 'simple_precision': targets['targetmask'],
            # 'simple_recall': targets['targetmask'],
            'simple_F1': targets['targetmask'],
            'simpleF1FixRule': targets['targetmask'],
            'EntityF1': targets['targetmask'],
            'EntityF1FixRule': targets['targetmask'],
            # 'EntityF1FixRule2': targets['targetmask'],
            # 'EntityF1FixRuleFixRule': targets['targetmask'],
            'EntityPrecision': targets['targetmask'],
            'EntityPrecisionFixRule': targets['targetmask'],
            'EntityRecall': targets['targetmask'],
            'EntityRecallFixRule': targets['targetmask'],
            'SeqEvalF1': targets['targetmask'],
            'SeqEvalF1FixRule': targets['targetmask'],
        }
        if self._params.feasible_pred_ids:
            if self._params.use_crf:
                sw_dict['EntityF1FP'] = targets['targetmask']
                sw_dict['SeqEvalF1FP'] = targets['targetmask']
            else:
                sw_dict['SeqEvalF1FP'] = targets['targetmask']
        if self._params.bet_tagging_:
            bet_sw_dict = {"StartF1": targets['targetmask'],
                           "EndF1": targets['targetmask'],
                           "ClassF1": targets['targetmask'],
                           }
            sw_dict = {**sw_dict, **bet_sw_dict}

        return sw_dict

    def build(self, inputs_targets: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        outputs = super(Model, self).build(inputs_targets)

        if self._params.pretrained_bert or self._params.use_hf_model_:
            self.init_new_training()

        return outputs

    def init_new_training(self):
        if self._params.use_hf_model_:
            self._graph.pretrained_bert = TFBertModel.from_pretrained(self._params.pretrained_hf_model_,
                                                                      return_dict=True)
        else:
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
        if self._params.use_hf_model_:
            sentence = inputs['input_ids']
        else:
            sentence = inputs['sentence']
        bet_str = ""
        if "pred_ids" not in outputs:
            bmw_obj = BetMetricWrapper(metric_obj=None, tags_fn=self._params.tags_fn_)  # no metric need, just use py_fn
            y_pred_arr = tf.py_function(bmw_obj.py_func2,
                                        [tf.expand_dims(outputs["probabilities_cse"], axis=0),
                                         tf.expand_dims(targets["targetmask"], axis=0)],
                                        Tout=[tf.int32])[0]
            outputs['pred_ids'] = tf.cast(tf.squeeze(y_pred_arr, axis=0), tf.int32).numpy()
            start_str = " ".join([f"{x:5.0f}" for x in outputs["probabilities_start"].tolist()])
            end_str = " ".join([f"{x:5.0f}" for x in outputs["probabilities_end"].tolist()])
            cls_str = " ".join([f"{self._tag_string_mapper.get_value(np.argmax(x)):>5}" for x in
                                outputs["probabilities_cls"].tolist()])
            bet_str += start_str + "\n" + end_str + "\n" + cls_str + "\n"

        pred = outputs['pred_ids']
        tgt = targets['tgt']
        mask = targets['targetmask']
        if self._params.feasible_pred_ids and self._params.use_crf == False:
            probs = outputs['logits']
            f1fpmetric = SeqEvalF1FP(self._params.tags_fn_, name='SeqEvalF1FP_print')
            seq_length = tf.argmax(tf.expand_dims(tgt, axis=0), axis=-1) + 1
            pred_fp = f1fpmetric.get_max_feasible_path_batch(tf.expand_dims(probs, axis=0), seq_length)
            tokens_str, tags_str, mask_str, preds_str = data.print_ner_sentence(sentence, tgt, mask, pred, pred_fp[0])
            f1fpmetric.update_state_with_fppreds(tf.expand_dims(tgt, axis=0), pred_fp)
            f1fp = f1fpmetric.result()
            f1fpmetric.reset_states()
        else:
            tokens_str, tags_str, mask_str, preds_str = data.print_ner_sentence(sentence, tgt, mask, pred)

        f1_metric = EntityF1(self._params.tags_fn_, name='EntityF1_print')
        f1_metric.update_state(tf.expand_dims(tgt, axis=0), tf.expand_dims(pred, axis=0))
        f1 = f1_metric.result()
        c, p, a = f1_metric._correct.numpy(), f1_metric._possible.numpy(), f1_metric._actual.numpy()
        f1_metric.reset_states()
        if c < p:
            error = "ERROR\n"
        else:
            error = ""
        if self._params.feasible_pred_ids and self._params.use_crf == False:
            print_fn(f'\n'
                     f'in:  {tokens_str}\n'
                     f'mask:{mask_str}\n'
                     f'tgt: {tags_str}\n'
                     f'pred:{preds_str}\n'
                     f'F1: {f1};\t correct: {c}, possible: {p}, actual: {a}\n{error}'
                     f'F1FP: {f1fp}')
        else:
            print_fn(f'\n'
                     f'in:  {tokens_str}\n'
                     f'mask:{mask_str}\n'
                     f'tgt: {tags_str}\n'
                     f'pred:{preds_str}\n'
                     f'F1: {f1};\t correct: {c}, possible: {p}, actual: {a}\n{error}{bet_str}')


class NERwithMiniBERT(GraphBase):
    def __init__(self, params, name='model', **kwargs):
        super(NERwithMiniBERT, self).__init__(params, name=name, **kwargs)
        self._tag_string_mapper = get_sm(self._params.tags_fn_)
        self._params.target_vocab_size = params.num_tags_ + 2
        self._tracked_layers = dict()
        self.pretrained_bert = getattr(transformers, self._params.bert_graph)(self._params)
        self._dropout = tf.keras.layers.Dropout(self._params.dropout_last)
        if self._params.bet_tagging_:
            # print(self._params.target_vocab_size-1)
            # half of the classes is used plus O-Class, sos, eos
            # print(self._params.num_tags_, int((self._params.num_tags_) // 2 + 3))
            self._layer_cls = tf.keras.layers.Dense(int((self._params.num_tags_) // 2 + 3),
                                                    activation=tf.keras.activations.softmax,
                                                    name="layer_cls")
            self._layer_start = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, name="layer_start")
            self._layer_end = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, name="layer_end")
        elif self._params.use_crf:
            self._last_layer = tf.keras.layers.Dense(self._params.target_vocab_size,
                                                     name="last_layer")
            self._trans_params = tf.keras.layers.Embedding(self._params.target_vocab_size,
                                                           self._params.target_vocab_size, name='trans_params')
            # ,embeddings_initializer=tf.keras.initializers.Constant(1))
            if self._params.crf_with_ner_rule:
                self._penalty_factor = tf.keras.layers.Embedding(1,
                                                                 1, name='penalty_factor')
                # ,embeddings_initializer=tf.keras.initializers.Constant(1))
                self._penalty_absolute = tf.keras.layers.Embedding(1,
                                                                   1, name='penalty_absolute')
                # ,embeddings_initializer=tf.keras.initializers.Constant(1))
            self.init_crf_with_ner_rule((self._params.target_vocab_size - 3) // 2)
        else:
            self._last_layer = tf.keras.layers.Dense(self._params.target_vocab_size,
                                                     activation=tf.keras.activations.softmax,
                                                     name="last_layer")

    def init_crf_with_ner_rule(self, real_tag_num):
        identity = tf.eye(num_rows=real_tag_num, dtype=tf.int32)
        onesmatrix = tf.ones([real_tag_num, real_tag_num], dtype=tf.int32)
        onescolumn = tf.ones([real_tag_num, 1], tf.int32)
        zerocolumn = tf.zeros([real_tag_num, 1], tf.int32)
        always_allowed_column = tf.concat(
            [tf.ones([1, real_tag_num], tf.int32), tf.zeros([1, real_tag_num], tf.int32), tf.ones([1, 1], tf.int32),
             tf.zeros([1, 1], tf.int32), tf.ones([1, 1], tf.int32)], axis=1)
        eos_allowed_column = tf.concat(
            [tf.zeros([1, 2 * real_tag_num], tf.int32), tf.ones([1, 1], tf.int32), tf.zeros([1, 2], tf.int32)], axis=1)
        bicolumns = tf.concat([onesmatrix, identity, onescolumn, zerocolumn, onescolumn], axis=1)
        allowed_transitions = tf.concat(
            [bicolumns, bicolumns, always_allowed_column, always_allowed_column, eos_allowed_column], axis=0)
        self._allowed_transitions = tf.cast(allowed_transitions, tf.float32)
        self._forbidden_transitions = tf.cast((1 - allowed_transitions), tf.float32)

    # def py_func2(self, y_pred):
    #     y_pred_arr = np.empty_like(y_pred.numpy()[:, :, 0], dtype=np.int)
    #     # tf.print(y_pred.numpy().shape)
    #     for s_idx, sample in enumerate(y_pred.numpy()):
    #         y_cls = sample[:, :-2]
    #         y_start = sample[:, -2]
    #         y_end = sample[:, -1]
    #         # if s_idx == 0:
    #         #     tf.print(y_start.shape)
    #         #     tf.print("start", tf.round(y_start), summarize=1000)
    #         #     tf.print("end  ", tf.round(y_end), summarize=1000)
    #
    #         targets = np.ones(y_start.shape) * self._tag_string_mapper.get_oov_id()
    #         # set sos and eos tag if they are predicted right
    #         targets[0] = np.argmax(y_cls[0]) + self._tag_string_mapper.size() // 2
    #         targets[-1] = np.argmax(y_cls[-1]) + self._tag_string_mapper.size() // 2
    #
    #
    #         start_buffer = -1
    #         end_buffer = -1
    #         last_end = 0
    #         for idx, token in enumerate(sample):
    #             if y_start[idx] > 0.5 and start_buffer < 0:
    #                 start_buffer = idx
    #                 # if s_idx ==0:
    #                 #     tf.print("startB:", start_buffer)
    #             # if to start tokens find highest end token in between
    #             elif y_start[idx] > 0.5 and start_buffer >= 0:
    #                 end_buffer = np.argmax(y_end[start_buffer:idx+1])
    #                 # if s_idx ==0:
    #                 #     tf.print("EndeB:", end_buffer)
    #
    #             if y_end[idx] > 0.5 and start_buffer >= 0:
    #                 end_buffer = idx
    #                 # if s_idx ==0:
    #                 #     tf.print("EndeB:", end_buffer)
    #             # if no start found get highest before last entity
    #             elif y_end[idx] > 0.5 and start_buffer < 0:
    #                 start_buffer = np.argmax(y_start[last_end:idx+1])
    #                 end_buffer = idx
    #             if start_buffer >= 0 and end_buffer >= 0:
    #                 entity_sum = np.sum(y_cls[start_buffer:end_buffer+1], axis=0)
    #                 cls = np.argmax(entity_sum) + self._tag_string_mapper.size() // 2
    #                 targets[start_buffer:end_buffer+1] = cls * np.ones_like(targets[start_buffer:end_buffer+1])
    #                 if cls != self._tag_string_mapper.get_oov_id():
    #                     targets[start_buffer] = cls - self._tag_string_mapper.size() // 2
    #                 start_buffer = -1
    #                 end_buffer = -1
    #                 last_end = idx
    #         y_pred_arr[s_idx] = targets
    #     return y_pred_arr

    @classmethod
    def params_cls(cls):
        return ModelParams

    def call(self, inputs, **kwargs):
        inp = dict()
        return_dict = dict()
        inp["text"] = inputs["sentence"]
        inp["seq_length"] = inputs["seq_length"]
        if self._params.whole_word_attention_:
            inp['word_length_vector'] = inputs['word_length_vector']
            inp['segment_ids'] = inputs['segment_ids']
        bert_graph_out = self.pretrained_bert(inp, **kwargs)
        bert_graph_out = self._dropout(bert_graph_out["enc_output"])
        if self._params.bet_tagging_:

            probs_cls = self._layer_cls(bert_graph_out)
            # tf.print("bert_graph_out", tf.shape(bert_graph_out))
            # tf.print("probs_cls", tf.shape(probs_cls))
            probs_start = self._layer_start(bert_graph_out)
            probs_end = self._layer_end(bert_graph_out)
            p_cse = tf.concat((probs_cls, probs_start, probs_end,), axis=-1)
            # pred_ids_ = tf.py_function(self.py_func2, [p_cse], Tout=[tf.int32])[0]
            return_dict = {"probabilities_cls": probs_cls,
                           "probabilities_start": tf.squeeze(probs_start, axis=-1),
                           "probabilities_end": tf.squeeze(probs_end, axis=-1),
                           "probabilities_cse": p_cse,
                           # "pred_ids": pred_ids_,
                           }
            # tf.print("probabilities_cse", return_dict["probabilities_cse"].shape)
            # tf.print("probabilities_start", return_dict["probabilities_start"].shape)
            # tf.print("probabilities_cls", return_dict["probabilities_cls"].shape)
        else:
            final_output = self._last_layer(bert_graph_out)  # (batch_size, tar_seq_len, target_vocab_size)
            if self._params.use_crf:
                trans_params = self._trans_params(tf.range(self._params.target_vocab_size))
                if self._params.crf_with_ner_rule:
                    penalty_factor = self._penalty_factor(tf.range(1))[0][0]
                    penalty_absolute = self._penalty_absolute(tf.range(1))[0][0]
                    factor = self._allowed_transitions + tf.math.scalar_mul(penalty_factor, self._forbidden_transitions)
                    absolute = tf.math.scalar_mul(penalty_absolute, self._forbidden_transitions)
                    trans_params = trans_params * factor - absolute
                # CRFs
                pred_ids, _ = tfa.text.crf_decode(final_output, trans_params, inputs["seq_length"][:, 0])
                pred_idsfp, _ = tfa.text.crf_decode(final_output, trans_params - 1000000 * self._forbidden_transitions,
                                                    inputs["seq_length"][:, 0])
                # broadcasting because of the lav engine: it needs netoutputs with the first shape dimension of the batch size
                trans_params = tf.broadcast_to(trans_params, [tf.shape(pred_ids)[0], tf.shape(trans_params)[0],
                                                              tf.shape(trans_params)[1]])
                return_dict = {"pred_ids": pred_ids, 'logits': final_output, 'probabilities': final_output,
                               'trans_params': trans_params, "pred_idsfp": pred_idsfp}
            else:
                pred_ids = tf.argmax(input=final_output, axis=2, output_type=tf.int32)
                return_dict = {"pred_ids": pred_ids, 'logits': final_output, 'probabilities': final_output}
        return return_dict


class NERwithHFBERT(GraphBase):
    def __init__(self, params, name='model', **kwargs):
        super(NERwithHFBERT, self).__init__(params, name=name, **kwargs)
        self._params.target_vocab_size = params.num_tags_ + 2
        self._tracked_layers = dict()
        self.pretrained_bert = TFBertModel(BertConfig.from_pretrained(params.pretrained_hf_model_))
        self._last_layer = tf.keras.layers.Dense(self._params.target_vocab_size,
                                                 activation=tf.keras.activations.softmax,
                                                 name="last_layer")

    @classmethod
    def params_cls(cls):
        return ModelParams

    def call(self, inputs, **kwargs):
        bert_graph_out = self.pretrained_bert(inputs, return_dict=True)
        final_output = self._last_layer(
            bert_graph_out.last_hidden_state)  # (batch_size, tar_seq_len, target_vocab_size)
        pred_ids = tf.argmax(input=final_output, axis=2, output_type=tf.int32)
        return {"pred_ids": pred_ids, 'logits': final_output, 'probabilities': final_output}
