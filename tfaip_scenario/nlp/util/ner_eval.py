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
import abc
import logging
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import seqeval
import tensorflow as tf
from paiargparse import pai_dataclass
from seqeval.metrics import f1_score, sequence_labeling
from tensorflow.keras.metrics import Mean

from tfaip import EvaluatorParams, Sample
from tfaip.evaluator.evaluator import EvaluatorBase
from tfaip_addons.util.file.stringmapper import get_sm, StringMapper
from tfaip_scenario.nlp.util.nlp_helper import get_ner_string_mapper


@pai_dataclass
@dataclass
class NEREvaluatorParams(EvaluatorParams):
    tags_fn: str = None


class SeqEvalF1HugFace(EvaluatorBase[NEREvaluatorParams]):
    def __init__(self, params: NEREvaluatorParams):
        super(SeqEvalF1HugFace, self).__init__(params)
        self._tag_string_mapper = get_sm(params.tags_fn)
        self.oov_id = self._tag_string_mapper.get_oov_id()
        self._truth_tags_ = []
        self._pred_tags_ = []

    def seq_f1_score_sw(self, y_true, y_pred, sample_weight):

        cur_truth_tag_el = []
        cur_pred_tag_el = []
        for i in range(len(y_true)):
            #     cur_truth_tag_el.append(self._tag_string_mapper.get_value(self.oov_id).replace('UNK', 'O'))
            #     cur_pred_tag_el.append(self._tag_string_mapper.get_value(self.oov_id).replace('UNK', 'O'))
            # else:
            if sample_weight[i] != 0:
                cur_truth_tag_el.append(self._tag_string_mapper.get_value(y_true[i]).replace("UNK", "O"))
                cur_pred_tag_el.append(self._tag_string_mapper.get_value(y_pred[i]).replace("UNK", "O"))
        return cur_truth_tag_el, cur_pred_tag_el

    def update_state(self, sampel: Sample):
        truth_tag_list, pred_tag_list = self.seq_f1_score_sw(
            y_true=sampel.targets["tgt"], y_pred=sampel.outputs["pred_ids"], sample_weight=sampel.targets["targetmask"]
        )
        self._truth_tags_.append(truth_tag_list)
        self._pred_tags_.append(pred_tag_list)

    def result(self):
        if len(self._truth_tags_) == 0:
            return {}  # No data

        report = seqeval.metrics.classification_report(
            y_true=self._truth_tags_, y_pred=self._pred_tags_, suffix=False, output_dict=True
        )
        overall_score = report.pop("micro avg")
        print(overall_score)
        overall_score["support"] = int(overall_score["support"])
        return overall_score


class SeqEvalF1Old(Mean):
    def __init__(self, tags_fn, **kwargs):
        super(SeqEvalF1Old, self).__init__(**kwargs)
        self._tag_string_mapper = get_sm(str(tags_fn))
        self._counter = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="counter")
        self._current_sum = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="current_sum")
        self.oov_id = self._tag_string_mapper.get_oov_id()

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            cur_f1_score = tf.py_function(self.seq_f1_score_sw, [y_true, y_pred, sample_weight], Tout=tf.float32)
        else:
            cur_f1_score = tf.py_function(self.seq_f1_score, [y_true, y_pred], Tout=tf.float32)
        self._counter.assign_add(1.0)
        self._current_sum.assign_add(cur_f1_score)

    def seq_f1_score(self, y_true, y_pred):
        truth_tag_list = []
        pred_tag_list = []
        for truth_sample, pred_sample in zip(y_true.numpy(), y_pred.numpy()):
            truth_tag_list.append([self._tag_string_mapper.get_value(x).replace("UNK", "O") for x in truth_sample])
            pred_tag_list.append([self._tag_string_mapper.get_value(x).replace("UNK", "O") for x in pred_sample])
        cur_f1_score = f1_score(truth_tag_list, pred_tag_list)
        return cur_f1_score

    def seq_f1_score_sw(self, y_true, y_pred, sample_weight):
        truth_tag_list = []
        pred_tag_list = []

        for truth_sample, pred_sample, sw_sample in zip(y_true.numpy(), y_pred.numpy(), sample_weight.numpy()):
            cur_truth_tag_el = []
            cur_pred_tag_el = []
            for i in range(len(truth_sample)):
                if sw_sample[i] == 0:
                    cur_truth_tag_el.append(self._tag_string_mapper.get_value(self.oov_id).replace("UNK", "O"))
                    cur_pred_tag_el.append(self._tag_string_mapper.get_value(self.oov_id).replace("UNK", "O"))
                else:
                    cur_truth_tag_el.append(self._tag_string_mapper.get_value(truth_sample[i]).replace("UNK", "O"))
                    cur_pred_tag_el.append(self._tag_string_mapper.get_value(pred_sample[i]).replace("UNK", "O"))
            truth_tag_list.append(cur_truth_tag_el)
            pred_tag_list.append(cur_pred_tag_el)
        cur_f1_score = f1_score(truth_tag_list, pred_tag_list)
        return cur_f1_score

    def result(self):
        return self._current_sum / tf.maximum(self._counter, 1.0)

    def reset_states(self):
        self._counter.assign(0.0)
        self._current_sum.assign(0.0)


class SeqEvalF1FPOld(SeqEvalF1Old):
    def __init__(self, tags_fn, **kwargs):
        super(SeqEvalF1FPOld, self).__init__(tags_fn, **kwargs)
        self._real_tag_num = self._tag_string_mapper.get_oov_id() // 2
        self.init_feas_succesor_list()

    def init_feas_succesor_list(self):
        self._feas_succ_mapping = []
        always_allowed_succesors = list(range(self._real_tag_num))
        always_allowed_succesors.append(2 * self._real_tag_num)
        # add possible succesors for B- and I- tags
        for i in range(self._real_tag_num):
            allowed_succesors = always_allowed_succesors.copy()
            allowed_succesors.append(i + self._real_tag_num)
            self._feas_succ_mapping.append(allowed_succesors)
        for i in range(self._real_tag_num):
            self._feas_succ_mapping.append(self._feas_succ_mapping[i].copy())
        self._feas_succ_mapping.append(always_allowed_succesors)

    def seq_f1_score(self, y_true, y_pred):
        seq_length = tf.argmax(y_true, axis=-1) + 1
        y_pred_fp = self.get_max_feasible_path_batch(y_pred, seq_length)
        truth_tag_list = []
        pred_tag_list = []
        for truth_sample, pred_sample in zip(y_true.numpy(), y_pred_fp.numpy()):
            truth_tag_list.append([self._tag_string_mapper.get_value(x).replace("UNK", "O") for x in truth_sample])
            pred_tag_list.append([self._tag_string_mapper.get_value(x).replace("UNK", "O") for x in pred_sample])
        cur_f1_score = f1_score(truth_tag_list, pred_tag_list)
        return cur_f1_score

    def seq_f1_score_sw(self, y_true, y_pred, sample_weight):
        seq_length = tf.argmax(y_true, axis=-1) + 1
        y_pred_fp = self.get_max_feasible_path_batch(y_pred, seq_length)
        truth_tag_list = []
        pred_tag_list = []

        for truth_sample, pred_sample, sw_sample in zip(y_true.numpy(), y_pred_fp.numpy(), sample_weight.numpy()):
            cur_truth_tag_el = []
            cur_pred_tag_el = []
            for i in range(len(truth_sample)):
                if sw_sample[i] == 0:
                    cur_truth_tag_el.append(self._tag_string_mapper.get_value(self.oov_id).replace("UNK", "O"))
                    cur_pred_tag_el.append(self._tag_string_mapper.get_value(self.oov_id).replace("UNK", "O"))
                else:
                    cur_truth_tag_el.append(self._tag_string_mapper.get_value(truth_sample[i]).replace("UNK", "O"))
                    cur_pred_tag_el.append(self._tag_string_mapper.get_value(pred_sample[i]).replace("UNK", "O"))
            truth_tag_list.append(cur_truth_tag_el)
            pred_tag_list.append(cur_pred_tag_el)
        cur_f1_score = f1_score(truth_tag_list, pred_tag_list)
        return cur_f1_score

    def initialize_max_feasible_path_search(self, seq_length, init_probs):
        # only start with B- or O-tag
        for i in range(self._real_tag_num):
            init_probs[i + self._real_tag_num] = 0
        init_probs[2 * self._real_tag_num + 1] = 0
        init_probs[2 * self._real_tag_num + 2] = 0
        self._pathprobs = [init_probs]
        self._predecessors = [[2 * self._real_tag_num + 1] * (2 * self._real_tag_num + 3)]
        for i in range(seq_length - 3):
            self._pathprobs.append([0.0] * (2 * self._real_tag_num + 3))
            self._predecessors.append([2 * self._real_tag_num + 1] * (2 * self._real_tag_num + 3))

    def prob_update(self, cur_pos, from_index, to_index, to_prob):
        alternative = self._pathprobs[cur_pos][from_index] * to_prob
        if alternative > self._pathprobs[cur_pos + 1][to_index]:
            self._pathprobs[cur_pos + 1][to_index] = alternative
            self._predecessors[cur_pos + 1][to_index] = from_index

    def get_max_feasible_path(self, sample_prob, seq_len):
        self.initialize_max_feasible_path_search(seq_len, sample_prob[1])
        for cur_pos in range(seq_len - 3):
            for from_index in range(2 * self._real_tag_num + 1):
                for to_index in self._feas_succ_mapping[from_index]:
                    self.prob_update(cur_pos, from_index, to_index, sample_prob[cur_pos + 2][to_index])
        last_tag_index = np.argmax(self._pathprobs[-1], axis=-1)
        max_path = (
            [last_tag_index] + [2 * self._real_tag_num + 2] + [2 * self._real_tag_num] * (len(sample_prob) - seq_len)
        )
        for cur_pos in reversed(range(seq_len - 2)):
            last_tag_index = self._predecessors[cur_pos][last_tag_index]
            max_path = [last_tag_index] + max_path
        vgl = np.argmax(sample_prob, axis=-1)
        return max_path

    def get_max_feasible_path_batch(self, probabilities, seq_length):
        pred_ids = []
        for sample_probs, seq_len in zip(probabilities.numpy(), seq_length.numpy()):
            sample_pred_ids = self.get_max_feasible_path(sample_probs, seq_len)  # np.argmax(sample_probs,axis=-1)
            pred_ids.append(sample_pred_ids)
        return tf.constant(pred_ids, tf.int32)

    def update_state_with_fppreds(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            cur_f1_score = tf.py_function(
                self.seq_f1_score_with_fppreds_sw, [y_true, y_pred, sample_weight], Tout=tf.float32
            )
        else:
            cur_f1_score = tf.py_function(self.seq_f1_score_with_fppreds, [y_true, y_pred], Tout=tf.float32)
        self._counter.assign_add(1.0)
        self._current_sum.assign_add(cur_f1_score)

    def seq_f1_score_with_fppreds(self, y_true, y_pred):
        truth_tag_list = []
        pred_tag_list = []
        for truth_sample, pred_sample in zip(y_true.numpy(), y_pred.numpy()):
            truth_tag_list.append([self._tag_string_mapper.get_value(x).replace("UNK", "O") for x in truth_sample])
            pred_tag_list.append([self._tag_string_mapper.get_value(x).replace("UNK", "O") for x in pred_sample])
        cur_f1_score = f1_score(truth_tag_list, pred_tag_list)
        return cur_f1_score

    def seq_f1_score_with_fppreds_sw(self, y_true, y_pred, sample_weight):
        truth_tag_list = []
        pred_tag_list = []
        for truth_sample, pred_sample, sw_sample in zip(y_true.numpy(), y_pred.numpy(), sample_weight.numpy()):
            cur_truth_tag_el = []
            cur_pred_tag_el = []
            for i in range(len(truth_sample)):
                if sw_sample[i] == 0:
                    cur_truth_tag_el.append(self._tag_string_mapper.get_value(self.oov_id).replace("UNK", "O"))
                    cur_pred_tag_el.append(self._tag_string_mapper.get_value(self.oov_id).replace("UNK", "O"))
                else:
                    cur_truth_tag_el.append(self._tag_string_mapper.get_value(truth_sample[i]).replace("UNK", "O"))
                    cur_pred_tag_el.append(self._tag_string_mapper.get_value(pred_sample[i]).replace("UNK", "O"))
            truth_tag_list.append(cur_truth_tag_el)
            pred_tag_list.append(cur_pred_tag_el)
        cur_f1_score = f1_score(truth_tag_list, pred_tag_list)
        return cur_f1_score


class BaseF1(Mean):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._correct = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="correct")
        self._possible = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="possible")
        self._actual = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="actual")

    @abc.abstractmethod
    def update_state(self, y_true, y_pred, sample_weight=None):
        raise NotImplementedError

    def result(self):
        precision = self._correct / tf.maximum(self._actual, 1.0)
        recall = self._correct / tf.maximum(self._possible, 1.0)
        return 2 * precision * recall / tf.maximum(precision + recall, 1.0)

    def reset_states(self):
        self._correct.assign(0.0)
        self._possible.assign(0.0)
        self._actual.assign(0.0)


class StartF1(BaseF1):
    def update_state(self, y_true, y_pred, sample_weight=None):
        sample_weight = tf.cast(sample_weight, dtype=tf.float32)
        y_true = tf.cast(tf.gather(y_true, 1, axis=-1), dtype=tf.float32)
        y_pred_round = tf.cast(tf.round(y_pred), dtype=tf.float32)
        # tf.print(y_pred_round[0], summarize=1000)
        possible = tf.reduce_sum(y_true * sample_weight)
        actual = tf.reduce_sum(y_pred_round * sample_weight)
        correct = tf.reduce_sum(y_true * y_pred_round * sample_weight)
        self._correct.assign_add(correct)
        self._possible.assign_add(possible)
        self._actual.assign_add(actual)


class EndF1(BaseF1):
    def update_state(self, y_true, y_pred, sample_weight=None):
        sample_weight = tf.cast(sample_weight, dtype=tf.float32)
        y_true = tf.cast(tf.gather(y_true, 2, axis=-1), dtype=tf.float32)
        y_pred_round = tf.cast(tf.round(y_pred), dtype=tf.float32)
        possible = tf.reduce_sum(y_true * sample_weight * sample_weight)
        actual = tf.reduce_sum(y_pred_round * sample_weight)
        correct = tf.reduce_sum(y_true * y_pred_round * sample_weight)
        # tf.print(y_true)

        self._correct.assign_add(correct)
        self._possible.assign_add(possible)
        self._actual.assign_add(actual)


class ClassF1(BaseF1):
    def __init__(self, oov_id, num_tags, **kwargs):
        super().__init__(**kwargs)
        self.oov_id = oov_id
        self.num_tags = num_tags

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = y_true - self.num_tags // 2
        # y_true = tf.one_hot(y_true, depth=y_pred.shape()[-1]-2, axis=-1, dtype=tf.float32)
        y_pred_cut = tf.cast(tf.argmax(tf.stack(tf.unstack(y_pred, axis=-1)[:-2], axis=-1), axis=-1), dtype=tf.int32)
        # tf.print(tf.shape(y_pred_cut), tf.shape(y_pred))
        correct = tf.reduce_sum(
            tf.cast(tf.raw_ops.Equal(x=tf.gather(y_true, 0, axis=-1), y=y_pred_cut), tf.float32)
            * tf.cast(sample_weight, dtype=tf.float32)
        )
        actual = tf.reduce_sum(
            tf.cast(tf.raw_ops.NotEqual(x=y_pred_cut, y=self.oov_id), tf.float32)
            * tf.cast(sample_weight, dtype=tf.float32)
        )
        possible = tf.reduce_sum(
            tf.cast(tf.raw_ops.NotEqual(x=tf.gather(y_true, 0, axis=-1), y=self.oov_id), tf.float32)
            * tf.cast(sample_weight, dtype=tf.float32)
        )
        # possible = tf.reduce_sum(y_true)
        # actual = tf.reduce_sum(y_pred_round)
        # correct = tf.reduce_sum(y_true * y_pred_round)
        # tf.print(correct, possible, actual)
        self._correct.assign_add(correct)
        self._possible.assign_add(possible)
        self._actual.assign_add(actual)


class SeqEvalF1(BaseF1):
    """padding-tag and 'O' tag MUST NOT be in the tag-map"""

    def __init__(self, tags_fn, **kwargs):
        super(SeqEvalF1, self).__init__(**kwargs)
        self._tag_string_mapper = get_ner_string_mapper(tags_fn)
        self.oov_id = self._tag_string_mapper.get_oov_id()

    def extract_tp_actual_correct(self, y_true, y_pred, suffix):
        entities_true = sequence_labeling.defaultdict(set)
        entities_pred = sequence_labeling.defaultdict(set)
        for type_name, start, end in sequence_labeling.get_entities(y_true, suffix):
            entities_true[type_name].add((start, end))
        for type_name, start, end in sequence_labeling.get_entities(y_pred, suffix):
            entities_pred[type_name].add((start, end))

        target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

        tp_sum = np.array([], dtype=np.int32)
        pred_sum = np.array([], dtype=np.int32)
        true_sum = np.array([], dtype=np.int32)
        for type_name in target_names:
            entities_true_type = entities_true.get(type_name, set())
            entities_pred_type = entities_pred.get(type_name, set())
            tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
            pred_sum = np.append(pred_sum, len(entities_pred_type))
            true_sum = np.append(true_sum, len(entities_true_type))

        return pred_sum, tp_sum, true_sum

    def seq_f1_score_sw(self, y_true, y_pred, sample_weight):
        truth_tag_list = []
        pred_tag_list = []

        for truth_sample, pred_sample, sw_sample in zip(y_true.numpy(), y_pred.numpy(), sample_weight.numpy()):
            cur_truth_tag_el = []
            cur_pred_tag_el = []
            for i in range(len(truth_sample)):
                if sw_sample[i] == 0:
                    cur_truth_tag_el.append(self._tag_string_mapper.get_value(self.oov_id).replace("UNK", "O"))
                    cur_pred_tag_el.append(self._tag_string_mapper.get_value(self.oov_id).replace("UNK", "O"))
                else:
                    cur_truth_tag_el.append(self._tag_string_mapper.get_value(truth_sample[i]).replace("UNK", "O"))
                    cur_pred_tag_el.append(self._tag_string_mapper.get_value(pred_sample[i]).replace("UNK", "O"))
            truth_tag_list.append(cur_truth_tag_el)
            pred_tag_list.append(cur_pred_tag_el)
        pred_sum, tp_sum, true_sum = self.extract_tp_actual_correct(truth_tag_list, pred_tag_list, False)
        correct = sum(tp_sum)
        possible = sum(true_sum)
        actual = sum(pred_sum)
        return correct, possible, actual

    def update_state(self, y_true, y_pred, sample_weight=None):
        correct, possible, actual = tf.py_function(
            self.seq_f1_score_sw, [y_true, y_pred, sample_weight], Tout=[tf.float32, tf.float32, tf.float32]
        )

        self._correct.assign_add(correct)
        self._possible.assign_add(possible)
        self._actual.assign_add(actual)


class SeqEvalF1FP(SeqEvalF1):
    def __init__(self, tags_fn, **kwargs):
        super(SeqEvalF1FP, self).__init__(tags_fn, **kwargs)
        self._real_tag_num = self._tag_string_mapper.get_oov_id() // 2
        self.init_feas_succesor_list()

    def init_feas_succesor_list(self):
        self._feas_succ_mapping = []
        always_allowed_succesors = list(range(self._real_tag_num))
        always_allowed_succesors.append(2 * self._real_tag_num)
        # add possible succesors for B- and I- tags
        for i in range(self._real_tag_num):
            allowed_succesors = always_allowed_succesors.copy()
            allowed_succesors.append(i + self._real_tag_num)
            self._feas_succ_mapping.append(allowed_succesors)
        for i in range(self._real_tag_num):
            self._feas_succ_mapping.append(self._feas_succ_mapping[i].copy())
        self._feas_succ_mapping.append(always_allowed_succesors)

    def extract_tp_actual_correct(self, y_true, y_pred, suffix):
        entities_true = sequence_labeling.defaultdict(set)
        entities_pred = sequence_labeling.defaultdict(set)
        for type_name, start, end in sequence_labeling.get_entities(y_true, suffix):
            entities_true[type_name].add((start, end))
        for type_name, start, end in sequence_labeling.get_entities(y_pred, suffix):
            entities_pred[type_name].add((start, end))

        target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

        tp_sum = np.array([], dtype=np.int32)
        pred_sum = np.array([], dtype=np.int32)
        true_sum = np.array([], dtype=np.int32)
        for type_name in target_names:
            entities_true_type = entities_true.get(type_name, set())
            entities_pred_type = entities_pred.get(type_name, set())
            tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
            pred_sum = np.append(pred_sum, len(entities_pred_type))
            true_sum = np.append(true_sum, len(entities_true_type))

        return pred_sum, tp_sum, true_sum

    def seq_f1_score_sw(self, y_true, y_pred, sample_weight):
        seq_length = tf.argmax(y_true, axis=-1) + 1
        y_pred_fp = self.get_max_feasible_path_batch(y_pred, seq_length)
        truth_tag_list = []
        pred_tag_list = []

        for truth_sample, pred_sample, sw_sample in zip(y_true.numpy(), y_pred_fp.numpy(), sample_weight.numpy()):
            cur_truth_tag_el = []
            cur_pred_tag_el = []
            for i in range(len(truth_sample)):
                if sw_sample[i] == 0:
                    cur_truth_tag_el.append(self._tag_string_mapper.get_value(self.oov_id).replace("UNK", "O"))
                    cur_pred_tag_el.append(self._tag_string_mapper.get_value(self.oov_id).replace("UNK", "O"))
                else:
                    cur_truth_tag_el.append(self._tag_string_mapper.get_value(truth_sample[i]).replace("UNK", "O"))
                    cur_pred_tag_el.append(self._tag_string_mapper.get_value(pred_sample[i]).replace("UNK", "O"))
            truth_tag_list.append(cur_truth_tag_el)
            pred_tag_list.append(cur_pred_tag_el)
        pred_sum, tp_sum, true_sum = self.extract_tp_actual_correct(truth_tag_list, pred_tag_list, False)
        correct = sum(tp_sum)
        possible = sum(true_sum)
        actual = sum(pred_sum)
        return correct, possible, actual

    def initialize_max_feasible_path_search(self, seq_length, init_probs):
        # only start with B- or O-tag
        for i in range(self._real_tag_num):
            init_probs[i + self._real_tag_num] = 0
        init_probs[2 * self._real_tag_num + 1] = 0
        init_probs[2 * self._real_tag_num + 2] = 0
        self._pathprobs = [init_probs]
        self._predecessors = [[2 * self._real_tag_num + 1] * (2 * self._real_tag_num + 3)]
        for i in range(seq_length - 3):
            self._pathprobs.append([0.0] * (2 * self._real_tag_num + 3))
            self._predecessors.append([2 * self._real_tag_num + 1] * (2 * self._real_tag_num + 3))

    def prob_update(self, cur_pos, from_index, to_index, to_prob):
        alternative = self._pathprobs[cur_pos][from_index] * to_prob
        if alternative > self._pathprobs[cur_pos + 1][to_index]:
            self._pathprobs[cur_pos + 1][to_index] = alternative
            self._predecessors[cur_pos + 1][to_index] = from_index

    def get_max_feasible_path(self, sample_prob, seq_len):
        self.initialize_max_feasible_path_search(seq_len, sample_prob[1])
        for cur_pos in range(seq_len - 3):
            for from_index in range(2 * self._real_tag_num + 1):
                for to_index in self._feas_succ_mapping[from_index]:
                    self.prob_update(cur_pos, from_index, to_index, sample_prob[cur_pos + 2][to_index])
        last_tag_index = np.argmax(self._pathprobs[-1], axis=-1)
        max_path = (
            [last_tag_index] + [2 * self._real_tag_num + 2] + [2 * self._real_tag_num] * (len(sample_prob) - seq_len)
        )
        for cur_pos in reversed(range(seq_len - 2)):
            last_tag_index = self._predecessors[cur_pos][last_tag_index]
            max_path = [last_tag_index] + max_path
        vgl = np.argmax(sample_prob, axis=-1)
        return max_path

    def get_max_feasible_path_batch(self, probabilities, seq_length):
        pred_ids = []
        for sample_probs, seq_len in zip(probabilities.numpy(), seq_length.numpy()):
            sample_pred_ids = self.get_max_feasible_path(sample_probs, seq_len)  # np.argmax(sample_probs,axis=-1)
            pred_ids.append(sample_pred_ids)
        return tf.constant(pred_ids, tf.int32)

    def update_state_with_fppreds(self, y_true, y_pred, sample_weight=None):
        correct, possible, actual = tf.py_function(
            self.seq_f1_score_with_fppreds_sw,
            [y_true, y_pred, sample_weight],
            Tout=[tf.float32, tf.float32, tf.float32],
        )

        self._correct.assign_add(correct)
        self._possible.assign_add(possible)
        self._actual.assign_add(actual)

    def seq_f1_score_with_fppreds_sw(self, y_true, y_pred, sample_weight):
        truth_tag_list = []
        pred_tag_list = []
        for truth_sample, pred_sample, sw_sample in zip(y_true.numpy(), y_pred.numpy(), sample_weight.numpy()):
            cur_truth_tag_el = []
            cur_pred_tag_el = []
            for i in range(len(truth_sample)):
                if sw_sample[i] == 0:
                    cur_truth_tag_el.append(self._tag_string_mapper.get_value(self.oov_id).replace("UNK", "O"))
                    cur_pred_tag_el.append(self._tag_string_mapper.get_value(self.oov_id).replace("UNK", "O"))
                else:
                    cur_truth_tag_el.append(self._tag_string_mapper.get_value(truth_sample[i]).replace("UNK", "O"))
                    cur_pred_tag_el.append(self._tag_string_mapper.get_value(pred_sample[i]).replace("UNK", "O"))
            truth_tag_list.append(cur_truth_tag_el)
            pred_tag_list.append(cur_pred_tag_el)
        pred_sum, tp_sum, true_sum = self.extract_tp_actual_correct(truth_tag_list, pred_tag_list, False)
        correct = sum(tp_sum)
        possible = sum(true_sum)
        actual = sum(pred_sum)
        return correct, possible, actual


class EntityF1(BaseF1):
    """padding-tag and 'O' tag MUST NOT be in the tag-map"""

    def __init__(self, tags_fn, **kwargs):
        super(EntityF1, self).__init__(**kwargs)
        self._tag_string_mapper = get_sm(str(tags_fn))
        self.oov_id = self._tag_string_mapper.get_oov_id()
        self._possible_tags = [
            self._tag_string_mapper.get_value(x).replace("B-", "")
            for x in range(self._tag_string_mapper.size())
            if "B-" in self._tag_string_mapper.get_value(x)
        ]

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            correct, possible, actual = tf.py_function(
                self.py_func_sw, [y_true, y_pred, sample_weight], Tout=[tf.float32, tf.float32, tf.float32]
            )
        else:
            correct, possible, actual = tf.py_function(
                self.py_func, [y_true, y_pred], Tout=[tf.float32, tf.float32, tf.float32]
            )
        self._correct.assign_add(correct)
        self._possible.assign_add(possible)
        self._actual.assign_add(actual)

    def py_func(self, y_true, y_pred):
        truth_tag_list = []
        pred_tag_list = []

        for truth_sample, pred_sample in zip(y_true.numpy(), y_pred.numpy()):
            # tf.print(truth_sample)
            # tf.print(pred_sample)
            truth_tag_list.append([self._tag_string_mapper.get_value(x) for x in truth_sample])
            pred_tag_list.append([self._tag_string_mapper.get_value(x) for x in pred_sample])
        # tf.print("\n" + "pred:" + " ".join(pred_tag_list[0]))
        # tf.print("tgt :" + " ".join(truth_tag_list[0]))
        evaluator = Evaluator(truth_tag_list, pred_tag_list, self._possible_tags)
        result, _ = evaluator.evaluate()
        return result["strict"]["correct"], result["strict"]["possible"], result["strict"]["actual"]

    def py_func_sw(self, y_true, y_pred, sample_weight):
        truth_tag_list = []
        pred_tag_list = []

        for truth_sample, pred_sample, sw_sample in zip(y_true.numpy(), y_pred.numpy(), sample_weight.numpy()):
            cur_truth_tag_el = []
            cur_pred_tag_el = []
            for i in range(len(truth_sample)):
                if sw_sample[i] == 0:
                    cur_truth_tag_el.append(self._tag_string_mapper.get_value(self.oov_id))
                    cur_pred_tag_el.append(self._tag_string_mapper.get_value(self.oov_id))
                else:
                    cur_truth_tag_el.append(self._tag_string_mapper.get_value(truth_sample[i]))
                    cur_pred_tag_el.append(self._tag_string_mapper.get_value(pred_sample[i]))
            truth_tag_list.append(cur_truth_tag_el)
            pred_tag_list.append(cur_pred_tag_el)
        evaluator = Evaluator(truth_tag_list, pred_tag_list, self._possible_tags)
        result, _ = evaluator.evaluate()
        return result["strict"]["correct"], result["strict"]["possible"], result["strict"]["actual"]


class BetMetricWrapper(Mean):
    def __init__(self, metric_obj: Mean, tags_fn, **kwargs):
        super().__init__(name=metric_obj.name, **kwargs)
        self._tag_string_mapper = get_sm(str(tags_fn))
        self._metric_obj = metric_obj

    def py_func2(self, y_pred, sample_weights):
        sample_weights = tf.cast(sample_weights, dtype=tf.float32)
        # y_pred = y_pred_sample_weights[0]
        # sample_weights = y_pred_sample_weights[1]
        y_pred_arr = np.empty_like(y_pred.numpy()[:, :, 0], dtype=np.int)
        # tf.print(y_pred.numpy().shape)
        for s_idx, sample, weights in zip(range(y_pred.numpy().shape[0]), y_pred.numpy(), sample_weights.numpy()):
            y_cls = sample[:, :-2]
            y_start = sample[:, -2] * weights
            y_end = sample[:, -1] * weights
            # if s_idx == 0:
            #     tf.print(y_start.shape)
            #     tf.print("start", tf.round(y_start), summarize=1000)
            #     tf.print("end  ", tf.round(y_end), summarize=1000)

            targets = np.ones(y_start.shape) * self._tag_string_mapper.get_oov_id()
            # set sos and eos tag if they are predicted right
            targets[0] = np.argmax(y_cls[0]) + self._tag_string_mapper.size() // 2
            targets[-1] = np.argmax(y_cls[-1]) + self._tag_string_mapper.size() // 2

            # global start_buffer
            start_buffer = -1
            # global end_b
            end_buffer = -1
            last_end = 0

            for idx, token in enumerate(sample):
                if y_start[idx] > 0.5 and start_buffer < 0:
                    start_buffer = idx
                    # if s_idx ==0:
                    #     tf.print("startB:", start_buffer)
                # if to start tokens find highest end token in between

                # if s_idx ==0:
                #     tf.print("EndeB:", end_buffer)

                if y_end[idx] > 0.5 and start_buffer >= 0:
                    end_buffer = idx
                    # if s_idx ==0:
                    #     tf.print("EndeB:", end_buffer)
                # if no start found get highest before last entity
                elif y_end[idx] > 0.5 and start_buffer < 0:
                    # tf.print("missing START TOKEN")
                    # +1 to avoid maybe override of previous entity token
                    start_buffer = last_end + 1 + np.argmax(y_start[last_end + 1 : idx + 1])
                    end_buffer = idx

                if start_buffer >= 0 and end_buffer < 0 and idx + 1 < len(sample) and y_start[idx + 1] > 0.5:
                    # it should be possible to query idx+1 since the last index is mask due to <eos>
                    end_buffer = start_buffer + np.argmax(y_end[start_buffer : idx + 1])

                if start_buffer >= 0 and end_buffer >= 0:
                    entity_sum = np.sum(y_cls[start_buffer : end_buffer + 1], axis=0)
                    cls = np.argmax(entity_sum) + self._tag_string_mapper.size() // 2
                    targets[start_buffer : end_buffer + 1] = cls * np.ones_like(targets[start_buffer : end_buffer + 1])
                    if cls < self._tag_string_mapper.get_oov_id():
                        targets[start_buffer] = cls - self._tag_string_mapper.size() // 2
                    elif cls > self._tag_string_mapper.get_oov_id():
                        targets[start_buffer] = self._tag_string_mapper.get_oov_id()
                    start_buffer = -1
                    end_buffer = -1
                    last_end = idx
            y_pred_arr[s_idx] = targets
        return y_pred_arr

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, dtype=tf.float32)
        y_pred_arr = tf.py_function(self.py_func2, [y_pred, sample_weight], Tout=[tf.int32])[0]
        y_pred_arr = tf.cast(y_pred_arr, tf.int32)
        # tf.print(y_pred_arr, summarize=1000)
        # tf.print(y_true, summarize=1000)
        self._metric_obj.update_state(y_true, y_pred_arr, sample_weight)

    def result(self):
        return self._metric_obj.result()

    def reset_states(self):
        self._metric_obj.reset_states()


class FixRuleMetricWrapper(Mean):
    def __init__(self, metric_obj: Mean, tags_fn, oov_id, **kwargs):
        super().__init__(name=metric_obj.name, **kwargs)
        self._tag_string_mapper: StringMapper = get_sm(str(tags_fn))
        self._oov_id = oov_id
        self._metric_obj = metric_obj

    def py_func(self, y_pred, sample_weight):
        pred_tag_list = []
        for pred_sample in y_pred.numpy():
            pred_tag_list.append([self._tag_string_mapper.get_value(x) for x in pred_sample])
        # Replace I-tags if there is no B-tag or I-tag of same class in the tag before

        for sample_idx, sentence in enumerate(pred_tag_list):
            for token_idx, tag in enumerate(sentence):
                if token_idx > 0 and str(tag).startswith("I-"):
                    if sample_weight.numpy()[sample_idx, token_idx - 1] == 0:
                        pred_tag_list[sample_idx][token_idx] = pred_tag_list[sample_idx][token_idx].replace("I-", "B-")
                    elif str(sentence[token_idx - 1]).replace("B-", "I-") != tag:
                        pred_tag_list[sample_idx][token_idx] = str(sentence[token_idx - 1]).replace("B-", "I-")
                # if token_idx > 0 and str(tag).startswith("I-") and str(sentence[token_idx - 1]).replace("B-", "I-") != tag:
                #     pred_tag_list[sample_idx][token_idx] = str(sentence[token_idx - 1]).replace("B-", "I-")
        # convert back to channels
        pred_channel_list = []
        for pred_sample in pred_tag_list:
            pred_channel_list.append(
                np.array([self._tag_string_mapper.get_channel(x) for x in pred_sample], dtype=np.int32)
            )
        return np.array(pred_channel_list, dtype=np.int32)

    #
    # def py_func_sw(self, y_true, y_pred,sample_weight):
    #     truth_tag_list = []
    #     pred_tag_list = []
    #     y_pred_fixed_sw =  y_pred.numpy()
    #     y_true_fixed_sw = y_true.numpy()
    #     for truth_sample, pred_sample, sw_sample in zip(y_true.numpy(), y_pred.numpy(), sample_weight.numpy()):
    #         cur_truth_tag_el = []
    #         cur_pred_tag_el = []
    #         for i in range(len(truth_sample)):
    #             # if sw_sample[i] == 0:
    #             #     cur_truth_tag_el.append(self._tag_string_mapper.get_value(self._oov_id))
    #             #     cur_pred_tag_el.append(self._tag_string_mapper.get_value(self._oov_id))
    #             # else:
    #                 cur_truth_tag_el.append(self._tag_string_mapper.get_value(truth_sample[i]))
    #                 cur_pred_tag_el.append(self._tag_string_mapper.get_value(pred_sample[i]))
    #         truth_tag_list.append(cur_truth_tag_el)
    #         pred_tag_list.append(cur_pred_tag_el)
    #     # Replace I-tags if there is no B-tag or I-tag of same class in the tag before
    #
    #     for sample_idx, sentence in enumerate(pred_tag_list):
    #         for token_idx, tag in enumerate(sentence):
    #             if token_idx > 0 and str(tag).startswith("I-"):#  and str(sentence[token_idx - 1]).replace("B-", "I-") != tag:
    #             #     pred_tag_list[sample_idx][token_idx] = str(sentence[token_idx - 1]).replace("B-", "I-")
    #             #     y_pred_fixed_sw[sample_idx][token_idx] = self._tag_string_mapper.get_channel(str(sentence[token_idx - 1]).replace("B-", "I-"))
    #                 if sample_weight.numpy()[sample_idx, token_idx - 1] == 0:
    #                     pred_tag_list[sample_idx][token_idx] = str(tag).replace("B-", "I-")
    #                 elif str(sentence[token_idx - 1]).replace("B-", "I-") != tag:
    #                     pred_tag_list[sample_idx][token_idx] = str(sentence[token_idx - 1]).replace("B-", "I-")
    #     # convert back to channels
    #     pred_channel_list = []
    #     truth_channel_list = []
    #     for pred_sample, truth_sample in zip(pred_tag_list, truth_tag_list):
    #         pred_channel_list.append(np.array([self._tag_string_mapper.get_channel(x) for x in pred_sample], dtype=np.int32))
    #         truth_channel_list.append(np.array([self._tag_string_mapper.get_channel(x) for x in truth_sample], dtype=np.int32))
    #     y_pred_fixed_sw = np.array(pred_channel_list, dtype=np.int32)
    #     y_true_fixed_sw = np.array(pred_channel_list, dtype=np.int32)
    #     return y_pred_fixed_sw

    def update_state(self, y_true, y_pred, sample_weight=None):
        assert sample_weight is not None, f"Sample_weight is None in {self._metric_obj.name}!"
        # y_pred_ = tf.py_function(self.py_func_sw, [y_true, y_pred, sample_weight], Tout=[tf.int32])[0]

        y_pred_ = tf.py_function(self.py_func, [y_pred, sample_weight], Tout=[tf.int32])[0]
        self._metric_obj.update_state(y_true, y_pred_, sample_weight)

    def result(self):
        return self._metric_obj.result()

    def reset_states(self):
        self._metric_obj.reset_states()


class EntityF1FixRule(EntityF1):

    # def py_func(self, y_true, y_pred):
    #     truth_tag_list = []
    #     pred_tag_list = []
    #     for truth_sample, pred_sample in zip(y_true.numpy(), y_pred.numpy()):
    #         truth_tag_list.append([self._tag_string_mapper.get_value(x) for x in truth_sample])
    #         pred_tag_list.append([self._tag_string_mapper.get_value(x) for x in pred_sample])
    #     # Replace I-tags if there is no B-tag or I-tag of same class in the tag before
    #
    #     for sample_idx, sentence in enumerate(pred_tag_list):
    #         for token_idx, tag in enumerate(sentence):
    #             if token_idx > 0 and str(tag).startswith("I-") and str(sentence[token_idx - 1]).replace("B-", "I-") != tag:
    #                 pred_tag_list[sample_idx][token_idx] = str(sentence[token_idx - 1]).replace("B-", "I-")
    #
    #     evaluator = Evaluator(truth_tag_list, pred_tag_list, self._possible_tags)
    #     result, _ = evaluator.evaluate()
    #     return result['strict']['correct'], result['strict']['possible'], result['strict']['actual']

    def py_func_sw(self, y_true, y_pred, sample_weight):
        truth_tag_list = []
        pred_tag_list = []

        for truth_sample, pred_sample, sw_sample in zip(y_true.numpy(), y_pred.numpy(), sample_weight.numpy()):
            cur_truth_tag_el = []
            cur_pred_tag_el = []
            for i in range(len(truth_sample)):
                if sw_sample[i] == 0:
                    cur_truth_tag_el.append(self._tag_string_mapper.get_value(self.oov_id))
                    cur_pred_tag_el.append(self._tag_string_mapper.get_value(self.oov_id))
                else:
                    cur_truth_tag_el.append(self._tag_string_mapper.get_value(truth_sample[i]))
                    cur_pred_tag_el.append(self._tag_string_mapper.get_value(pred_sample[i]))
            truth_tag_list.append(cur_truth_tag_el)
            pred_tag_list.append(cur_pred_tag_el)
        # Replace I-tags if there is no B-tag or I-tag of same class in the tag before

        for sample_idx, sentence in enumerate(pred_tag_list):
            for token_idx, tag in enumerate(sentence):
                if token_idx > 0 and str(tag).startswith("I-"):
                    if sample_weight.numpy()[sample_idx, token_idx - 1] == 0:
                        pred_tag_list[sample_idx][token_idx] = str(tag).replace("I-", "B-")
                    elif str(sentence[token_idx - 1]).replace("B-", "I-") != tag:
                        pred_tag_list[sample_idx][token_idx] = str(sentence[token_idx - 1]).replace("B-", "I-")

        evaluator = Evaluator(truth_tag_list, pred_tag_list, self._possible_tags)
        result, _ = evaluator.evaluate()
        return result["strict"]["correct"], result["strict"]["possible"], result["strict"]["actual"]

    def update_state(self, y_true, y_pred, sample_weight=None):
        assert sample_weight is not None, f"sample weight is None in EntityF1FixRule"
        correct, possible, actual = tf.py_function(
            self.py_func_sw, [y_true, y_pred, sample_weight], Tout=[tf.float32, tf.float32, tf.float32]
        )
        # else:
        #
        #     correct, possible, actual = tf.py_function(self.py_func, [y_true, y_pred],
        #                                            Tout=[tf.float32, tf.float32, tf.float32])
        self._correct.assign_add(correct)
        self._possible.assign_add(possible)
        self._actual.assign_add(actual)


class EntityPrecision(EntityF1):
    def result(self):
        return self._correct / tf.maximum(self._actual, 1)


class EntityRecall(EntityF1):
    def result(self):
        return self._correct / tf.maximum(self._possible, 1)


Entity = namedtuple("Entity", "e_type start_offset end_offset")


class Evaluator:
    def __init__(self, true, pred, tags):
        """ """

        if len(true) != len(pred):
            raise ValueError("Number of predicted documents does not equal true")

        self.true = true
        self.pred = pred
        self.tags = tags

        # Setup dict into which metrics will be stored.

        self.metrics_results = {
            "correct": 0,
            "incorrect": 0,
            "partial": 0,
            "missed": 0,
            "spurious": 0,
            "possible": 0,
            "actual": 0,
            "precision": 0,
            "recall": 0,
        }

        # Copy results dict to cover the four schemes.

        self.results = {
            "strict": deepcopy(self.metrics_results),
            "ent_type": deepcopy(self.metrics_results),
            "partial": deepcopy(self.metrics_results),
            "exact": deepcopy(self.metrics_results),
        }

        # Create an accumulator to store results

        self.evaluation_agg_entities_type = {e: deepcopy(self.results) for e in tags}

    def evaluate(self):

        logging.debug("Imported %s predictions for %s true examples", len(self.pred), len(self.true))

        for true_ents, pred_ents in zip(self.true, self.pred):

            # Check that the length of the true and predicted examples are the
            # same. This must be checked here, because another error may not
            # be thrown if the lengths do not match.

            if len(true_ents) != len(pred_ents):
                raise ValueError("Prediction length does not match true example length")

            # Compute results for one message

            tmp_results, tmp_agg_results = compute_metrics(
                collect_named_entities(true_ents), collect_named_entities(pred_ents), self.tags
            )

            # Cycle through each result and accumulate

            # TODO: Combine these loops below:

            for eval_schema in self.results:

                for metric in self.results[eval_schema]:
                    self.results[eval_schema][metric] += tmp_results[eval_schema][metric]

            # Calculate global precision and recall

            self.results = compute_precision_recall_wrapper(self.results)

            # Aggregate results by entity type

            for e_type in self.tags:

                for eval_schema in tmp_agg_results[e_type]:

                    for metric in tmp_agg_results[e_type][eval_schema]:
                        self.evaluation_agg_entities_type[e_type][eval_schema][metric] += tmp_agg_results[e_type][
                            eval_schema
                        ][metric]

                # Calculate precision recall at the individual entity level

                self.evaluation_agg_entities_type[e_type] = compute_precision_recall_wrapper(
                    self.evaluation_agg_entities_type[e_type]
                )

        return self.results, self.evaluation_agg_entities_type


def collect_named_entities(tokens):
    """
    Creates a list of Entity named-tuples, storing the entity type and the start and end
    offsets of the entity.

    :param tokens: a list of tags
    :return: a list of Entity named-tuples
    """

    named_entities = []
    start_offset = None
    end_offset = None
    ent_type = None

    for offset, token_tag in enumerate(tokens):

        if token_tag == "O":
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                named_entities.append(Entity(ent_type, start_offset, end_offset))
                start_offset = None
                end_offset = None
                ent_type = None

        elif ent_type is None:
            ent_type = token_tag[2:]
            start_offset = offset

        elif ent_type != token_tag[2:] or (ent_type == token_tag[2:] and token_tag[:1] == "B"):

            end_offset = offset - 1
            named_entities.append(Entity(ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token

    if ent_type is not None and start_offset is not None and end_offset is None:
        named_entities.append(Entity(ent_type, start_offset, len(tokens) - 1))

    return named_entities


def compute_metrics(true_named_entities, pred_named_entities, tags):
    eval_metrics = {"correct": 0, "incorrect": 0, "partial": 0, "missed": 0, "spurious": 0, "precision": 0, "recall": 0}

    # overall results

    evaluation = {
        "strict": deepcopy(eval_metrics),
        "ent_type": deepcopy(eval_metrics),
        "partial": deepcopy(eval_metrics),
        "exact": deepcopy(eval_metrics),
    }

    # results by entity type

    evaluation_agg_entities_type = {e: deepcopy(evaluation) for e in tags}

    # keep track of entities that overlapped

    true_which_overlapped_with_pred = []

    # Subset into only the tags that we are interested in.
    # NOTE: we remove the tags we don't want from both the predicted and the
    # true entities. This covers the two cases where mismatches can occur:
    #
    # 1) Where the model predicts a tag that is not present in the true data
    # 2) Where there is a tag in the true data that the model is not capable of
    # predicting.

    true_named_entities = [ent for ent in true_named_entities if ent.e_type in tags]
    pred_named_entities = [ent for ent in pred_named_entities if ent.e_type in tags]

    # go through each predicted named-entity

    for pred in pred_named_entities:
        found_overlap = False

        # Check each of the potential scenarios in turn. See
        # http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
        # for scenario explanation.

        # Scenario I: Exact match between true and pred

        if pred in true_named_entities:
            true_which_overlapped_with_pred.append(pred)
            evaluation["strict"]["correct"] += 1
            evaluation["ent_type"]["correct"] += 1
            evaluation["exact"]["correct"] += 1
            evaluation["partial"]["correct"] += 1

            # for the agg. by e_type results
            evaluation_agg_entities_type[pred.e_type]["strict"]["correct"] += 1
            evaluation_agg_entities_type[pred.e_type]["ent_type"]["correct"] += 1
            evaluation_agg_entities_type[pred.e_type]["exact"]["correct"] += 1
            evaluation_agg_entities_type[pred.e_type]["partial"]["correct"] += 1

        else:

            # check for overlaps with any of the true entities

            for true in true_named_entities:

                pred_range = range(pred.start_offset, pred.end_offset)
                true_range = range(true.start_offset, true.end_offset)

                # Scenario IV: Offsets match, but entity type is wrong

                if (
                    true.start_offset == pred.start_offset
                    and pred.end_offset == true.end_offset
                    and true.e_type != pred.e_type
                ):

                    # overall results
                    evaluation["strict"]["incorrect"] += 1
                    evaluation["ent_type"]["incorrect"] += 1
                    evaluation["partial"]["correct"] += 1
                    evaluation["exact"]["correct"] += 1

                    # aggregated by entity type results
                    evaluation_agg_entities_type[true.e_type]["strict"]["incorrect"] += 1
                    evaluation_agg_entities_type[true.e_type]["ent_type"]["incorrect"] += 1
                    evaluation_agg_entities_type[true.e_type]["partial"]["correct"] += 1
                    evaluation_agg_entities_type[true.e_type]["exact"]["correct"] += 1

                    true_which_overlapped_with_pred.append(true)
                    found_overlap = True

                    break

                # check for an overlap i.e. not exact boundary match, with true entities

                elif find_overlap(true_range, pred_range):

                    true_which_overlapped_with_pred.append(true)

                    # Scenario V: There is an overlap (but offsets do not match
                    # exactly), and the entity type is the same.
                    # 2.1 overlaps with the same entity type

                    if pred.e_type == true.e_type:

                        # overall results
                        evaluation["strict"]["incorrect"] += 1
                        evaluation["ent_type"]["correct"] += 1
                        evaluation["partial"]["partial"] += 1
                        evaluation["exact"]["incorrect"] += 1

                        # aggregated by entity type results
                        evaluation_agg_entities_type[true.e_type]["strict"]["incorrect"] += 1
                        evaluation_agg_entities_type[true.e_type]["ent_type"]["correct"] += 1
                        evaluation_agg_entities_type[true.e_type]["partial"]["partial"] += 1
                        evaluation_agg_entities_type[true.e_type]["exact"]["incorrect"] += 1

                        found_overlap = True

                        break

                    # Scenario VI: Entities overlap, but the entity type is
                    # different.

                    else:
                        # overall results
                        evaluation["strict"]["incorrect"] += 1
                        evaluation["ent_type"]["incorrect"] += 1
                        evaluation["partial"]["partial"] += 1
                        evaluation["exact"]["incorrect"] += 1

                        # aggregated by entity type results
                        # Results against the true entity

                        evaluation_agg_entities_type[true.e_type]["strict"]["incorrect"] += 1
                        evaluation_agg_entities_type[true.e_type]["partial"]["partial"] += 1
                        evaluation_agg_entities_type[true.e_type]["ent_type"]["incorrect"] += 1
                        evaluation_agg_entities_type[true.e_type]["exact"]["incorrect"] += 1

                        # Results against the predicted entity

                        # evaluation_agg_entities_type[pred.e_type]['strict']['spurious'] += 1

                        found_overlap = True

                        break

            # Scenario II: Entities are spurious (i.e., over-generated).

            if not found_overlap:

                # Overall results

                evaluation["strict"]["spurious"] += 1
                evaluation["ent_type"]["spurious"] += 1
                evaluation["partial"]["spurious"] += 1
                evaluation["exact"]["spurious"] += 1

                # Aggregated by entity type results

                # NOTE: when pred.e_type is not found in tags
                # or when it simply does not appear in the test set, then it is
                # spurious, but it is not clear where to assign it at the tag
                # level. In this case, it is applied to all target_tags
                # found in this example. This will mean that the sum of the
                # evaluation_agg_entities will not equal evaluation.

                for true in tags:
                    evaluation_agg_entities_type[true]["strict"]["spurious"] += 1
                    evaluation_agg_entities_type[true]["ent_type"]["spurious"] += 1
                    evaluation_agg_entities_type[true]["partial"]["spurious"] += 1
                    evaluation_agg_entities_type[true]["exact"]["spurious"] += 1

    # Scenario III: Entity was missed entirely.

    for true in true_named_entities:
        if true in true_which_overlapped_with_pred:
            continue
        else:
            # overall results
            evaluation["strict"]["missed"] += 1
            evaluation["ent_type"]["missed"] += 1
            evaluation["partial"]["missed"] += 1
            evaluation["exact"]["missed"] += 1

            # for the agg. by e_type
            evaluation_agg_entities_type[true.e_type]["strict"]["missed"] += 1
            evaluation_agg_entities_type[true.e_type]["ent_type"]["missed"] += 1
            evaluation_agg_entities_type[true.e_type]["partial"]["missed"] += 1
            evaluation_agg_entities_type[true.e_type]["exact"]["missed"] += 1

    # Compute 'possible', 'actual' according to SemEval-2013 Task 9.1 on the
    # overall results, and use these to calculate precision and recall.

    for eval_type in evaluation:
        evaluation[eval_type] = compute_actual_possible(evaluation[eval_type])

    # Compute 'possible', 'actual', and precision and recall on entity level
    # results. Start by cycling through the accumulated results.

    for entity_type, entity_level in evaluation_agg_entities_type.items():

        # Cycle through the evaluation types for each dict containing entity
        # level results.

        for eval_type in entity_level:
            evaluation_agg_entities_type[entity_type][eval_type] = compute_actual_possible(entity_level[eval_type])

    return evaluation, evaluation_agg_entities_type


def find_overlap(true_range, pred_range):
    """Find the overlap between two ranges

    Find the overlap between two ranges. Return the overlapping values if
    present, else return an empty set().

    Examples:

    >>> find_overlap((1, 2), (2, 3))
    2
    >>> find_overlap((1, 2), (3, 4))
    set()
    """

    true_set = set(true_range)
    pred_set = set(pred_range)

    overlaps = true_set.intersection(pred_set)

    return overlaps


def compute_actual_possible(results):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with actual, possible populated.

    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    correct = results["correct"]
    incorrect = results["incorrect"]
    partial = results["partial"]
    missed = results["missed"]
    spurious = results["spurious"]

    # Possible: number annotations in the gold-standard which contribute to the
    # final score

    possible = correct + incorrect + partial + missed

    # Actual: number of annotations produced by the NER system

    actual = correct + incorrect + partial + spurious

    results["actual"] = actual
    results["possible"] = possible

    return results


def compute_precision_recall(results, partial_or_type=False):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with precison and recall populated.

    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    actual = results["actual"]
    possible = results["possible"]
    partial = results["partial"]
    correct = results["correct"]

    if partial_or_type:
        precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
        recall = (correct + 0.5 * partial) / possible if possible > 0 else 0

    else:
        precision = correct / actual if actual > 0 else 0
        recall = correct / possible if possible > 0 else 0

    results["precision"] = precision
    results["recall"] = recall

    return results


def compute_precision_recall_wrapper(results):
    """
    Wraps the compute_precision_recall function and runs on a dict of results
    """

    results_a = {
        key: compute_precision_recall(value, True) for key, value in results.items() if key in ["partial", "ent_type"]
    }
    results_b = {key: compute_precision_recall(value) for key, value in results.items() if key in ["strict", "exact"]}

    results = {**results_a, **results_b}

    return results
