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
import tensorflow as tf


class AccuracyTokens(tf.keras.metrics.Accuracy):
    def __init__(self, oov_id, **kwargs):
        super(AccuracyTokens, self).__init__(**kwargs)
        self.oov_id = oov_id

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_positive_indexes = tf.cast(tf.raw_ops.Equal(x=y_true, y=y_pred), tf.int32)
        not_oov = tf.cast(tf.raw_ops.NotEqual(x=y_pred, y=self.oov_id), tf.int32)
        booth_oov = tf.raw_ops.LogicalAnd(
            x=tf.raw_ops.Equal(x=y_true, y=self.oov_id), y=tf.raw_ops.Equal(x=y_pred, y=self.oov_id)
        )
        no_oov_weights = tf.cast(tf.raw_ops.LogicalNot(x=booth_oov), tf.int32) * tf.cast(sample_weight, tf.int32)
        super(AccuracyTokens, self).update_state(
            y_true=true_positive_indexes, y_pred=not_oov, sample_weight=no_oov_weights
        )


class PrecisionTokens(tf.keras.metrics.Precision):
    def __init__(self, oov_id, **kwargs):
        super(PrecisionTokens, self).__init__(**kwargs)
        self.oov_id = oov_id

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_positive_indexes = tf.cast(tf.raw_ops.Equal(x=y_true, y=y_pred), tf.int32)
        not_oov = tf.cast(tf.raw_ops.NotEqual(x=y_pred, y=self.oov_id), tf.int32)
        booth_oov = tf.raw_ops.LogicalAnd(
            x=tf.raw_ops.Equal(x=y_true, y=self.oov_id), y=tf.raw_ops.Equal(x=y_pred, y=self.oov_id)
        )
        no_oov_weights = tf.cast(tf.raw_ops.LogicalNot(x=booth_oov), tf.int32) * tf.cast(sample_weight, tf.int32)
        super(PrecisionTokens, self).update_state(
            y_true=true_positive_indexes, y_pred=not_oov, sample_weight=no_oov_weights
        )


class RecallTokens(tf.keras.metrics.Recall):
    def __init__(self, oov_id, **kwargs):
        super(RecallTokens, self).__init__(**kwargs)
        self.oov_id = oov_id

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_positive_indexes = tf.cast(tf.raw_ops.Equal(x=y_true, y=y_pred), tf.int32)
        not_oov = tf.cast(tf.raw_ops.NotEqual(x=y_true, y=self.oov_id), tf.int32)
        booth_oov = tf.raw_ops.LogicalAnd(
            x=tf.raw_ops.Equal(x=y_true, y=self.oov_id), y=tf.raw_ops.Equal(x=y_pred, y=self.oov_id)
        )
        no_oov_weights = tf.cast(tf.raw_ops.LogicalNot(x=booth_oov), tf.int32) * tf.cast(sample_weight, tf.int32)
        super(RecallTokens, self).update_state(
            y_true=not_oov, y_pred=true_positive_indexes, sample_weight=no_oov_weights
        )


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

        booth_oov = tf.raw_ops.LogicalAnd(
            x=tf.raw_ops.Equal(x=y_true, y=self.oov_id), y=tf.raw_ops.Equal(x=y_pred, y=self.oov_id)
        )
        no_oov_weights = tf.cast(tf.raw_ops.LogicalNot(x=booth_oov), tf.int32) * tf.cast(sample_weight, tf.int32)
        self.precision.update_state(true_positive_indexes, not_oov_pred, no_oov_weights)
        self.recall.update_state(not_oov_tgt, true_positive_indexes, no_oov_weights)

    def result(self):
        return (
            2
            * self.precision.result()
            * self.recall.result()
            / tf.maximum(self.precision.result() + self.recall.result(), tf.keras.backend.epsilon())
        )

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()
