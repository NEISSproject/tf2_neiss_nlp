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
import tensorflow.keras as keras
import tensorflow as tf


def activation_by_str(a: str):
    if a is None:
        return
    elif a == "leaky_relu":
        return leaky_relu()
    else:
        return getattr(keras.activations, a)


def leaky_relu(leak=0.1, name="leakyRelu"):
    return keras.layers.LeakyReLU(alpha=leak, name=name)


def relu(features, name=None):
    return tf.nn.relu(features, name=name)
