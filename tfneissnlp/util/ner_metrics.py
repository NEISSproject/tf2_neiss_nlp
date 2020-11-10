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


import tensorflow as tf
from tensorflow.keras.metrics import Mean
from tfneissnlp.util.thirdparty.ner_eval import Evaluator

from tfneissnlp.util.stringmapper import get_sm


class EntityF1(Mean):
    """padding-tag and 'O' tag MUST NOT be in the tag-map"""

    def __init__(self, tags_fn, **kwargs):
        super(EntityF1, self).__init__(**kwargs)
        self._tag_string_mapper = get_sm(tags_fn)
        self.oov_id = self._tag_string_mapper.get_oov_id()
        self._possible_tags = [self._tag_string_mapper.get_value(x).replace("B-", "")
                               for x in range(self._tag_string_mapper.size()) if "B-"
                               in self._tag_string_mapper.get_value(x)]
        self._correct = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self._possible = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self._actual = tf.Variable(0.0, dtype=tf.float32, trainable=False)

    def update_state(self, y_true, y_pred, sample_weight=None):
        correct, possible, actual = tf.py_function(self.py_func, [y_true, y_pred],
                                                   Tout=[tf.float32, tf.float32, tf.float32])
        self._correct.assign_add(correct)
        self._possible.assign_add(possible)
        self._actual.assign_add(actual)

    def py_func(self, y_true, y_pred):
        truth_tag_list = []
        pred_tag_list = []
        for truth_sample, pred_sample in zip(y_true.numpy(), y_pred.numpy()):
            truth_tag_list.append([self._tag_string_mapper.get_value(x) for x in truth_sample])
            pred_tag_list.append([self._tag_string_mapper.get_value(x) for x in pred_sample])
        evaluator = Evaluator(truth_tag_list, pred_tag_list, self._possible_tags)
        result, _ = evaluator.evaluate()
        return result['strict']['correct'], result['strict']['possible'], result['strict']['actual']

    def result(self):
        precision = self._correct / tf.maximum(self._actual, 1.0)
        recall = self._correct / tf.maximum(self._possible, 1.0)
        return 2 * precision * recall / tf.maximum(precision + recall, 1.0)

    def reset_states(self):
        self._correct.assign(0.0)
        self._possible.assign(0.0)
        self._actual.assign(0.0)


class EntityPrecision(EntityF1):
    def result(self):
        return self._correct / tf.maximum(self._actual, 1)


class EntityRecall(EntityF1):
    def result(self):
        return self._correct / tf.maximum(self._possible, 1)
