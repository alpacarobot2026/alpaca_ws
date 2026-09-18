# # Copyright 2023 The human_scene_transformer Authors.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# """Contains embedding layers."""

import tensorflow as tf


# class SinusoidalEmbeddingLayer(tf.keras.layers.Layer):
#   """Sinusoidal Postional Embedding for xyz and time."""

#   def __init__(self, min_freq=4, max_freq=256, hidden_size=256):
#     super().__init__()
#     self.min_freq = float(min_freq)
#     self.max_freq = float(max_freq)
#     self.hidden_size = hidden_size
#     if hidden_size % 2 != 0:
#       raise ValueError('hidden_size ({hidden_size}) must be divisible by 2.')
#     self.num_freqs_int32 = hidden_size // 2
#     self.num_freqs = tf.cast(self.num_freqs_int32, dtype=tf.float32)

#   def build(self, input_shape):
#     log_freq_increment = (
#         tf.math.log(float(self.max_freq) / float(self.min_freq)) /
#         tf.maximum(1.0, self.num_freqs - 1))
#     # [num_freqs]
#     self.inv_freqs = self.min_freq * tf.exp(
#         tf.range(self.num_freqs, dtype=tf.float32) * -log_freq_increment)

#   def call(self, input_tensor):
#     # [..., num_freqs]
#     input_tensor = tf.repeat(
#         input_tensor[..., tf.newaxis], self.num_freqs_int32, axis=-1)
#     # [..., h]
#     embedded = tf.concat([
#         tf.sin(input_tensor * self.inv_freqs),
#         tf.cos(input_tensor * self.inv_freqs)
#     ],
#                          axis=-1)
#     return embedded


# import tensorflow as tf
# import numpy as np



# class SinusoidalEmbeddingLayer(tf.keras.layers.Layer):
#   """Sinusoidal Postional Embedding for xyz and time."""

#   def __init__(self, min_freq=4, max_freq=256, hidden_size=256):
#     super().__init__()
#     self.min_freq = float(min_freq)
#     self.max_freq = float(max_freq)
#     self.hidden_size = hidden_size
#     if hidden_size % 2 != 0:
#       raise ValueError('hidden_size ({hidden_size}) must be divisible by 2.')
#     self.num_freqs_int32 = hidden_size // 2
    
#     # Calculate inv_freqs using numpy to avoid TF graph/scope issues in ROS
#     num_freqs = self.num_freqs_int32
#     log_freq_increment = (
#         np.log(float(self.max_freq) / float(self.min_freq)) /
#         max(1.0, num_freqs - 1))
#     inv_freqs_np = self.min_freq * np.exp(
#         np.arange(num_freqs, dtype=np.float32) * -log_freq_increment)
    
#     # Store as TF constant
#     self.inv_freqs = tf.constant(inv_freqs_np, dtype=tf.float32)

#   def call(self, input_tensor):
#     # [..., num_freqs]
#     input_tensor = tf.repeat(
#         input_tensor[..., tf.newaxis], self.num_freqs_int32, axis=-1)
#     # [..., h]
#     embedded = tf.concat([
#         tf.sin(input_tensor * self.inv_freqs),
#         tf.cos(input_tensor * self.inv_freqs)
#     ],
#                          axis=-1)
#     return embedded


"""Contains embedding layers."""

import tensorflow as tf
import numpy as np



class SinusoidalEmbeddingLayer(tf.keras.layers.Layer):
  """Sinusoidal Postional Embedding for xyz and time."""

  def __init__(self, min_freq=4, max_freq=256, hidden_size=256):
    super().__init__()
    self.min_freq = float(min_freq)
    self.max_freq = float(max_freq)
    self.hidden_size = hidden_size
    if hidden_size % 2 != 0:
      raise ValueError('hidden_size ({hidden_size}) must be divisible by 2.')
    self.num_freqs_int32 = hidden_size // 2
    
    # Calculate inv_freqs using numpy to avoid TF graph/scope issues in ROS
    num_freqs = self.num_freqs_int32
    log_freq_increment = (
        np.log(float(self.max_freq) / float(self.min_freq)) /
        max(1.0, num_freqs - 1))
    inv_freqs_np = self.min_freq * np.exp(
        np.arange(num_freqs, dtype=np.float32) * -log_freq_increment)
    
    # Store as TF constant
    self.inv_freqs = tf.constant(inv_freqs_np, dtype=tf.float32)

  def call(self, input_tensor):
    # input_tensor: [..., D] -> [..., D, 1]
    # self.inv_freqs: [F]
    # Broadcast multiply: [..., D, F]
    scaled = input_tensor[..., tf.newaxis] * self.inv_freqs
    
    # [..., D, 2*F] = [..., D, H]
    embedded = tf.concat([
        tf.sin(scaled),
        tf.cos(scaled)
    ],
                         axis=-1)
    return embedded

# class SinusoidalEmbeddingLayer(tf.keras.layers.Layer):
#   """Sinusoidal positional embedding for per-channel scalars (e.g., x,y,t)."""

#   def __init__(self, min_freq=4, max_freq=256, hidden_size=256):
#     super().__init__()
#     self.min_freq = float(min_freq)
#     self.max_freq = float(max_freq)
#     self.hidden_size = int(hidden_size)
#     if self.hidden_size % 2 != 0:
#       raise ValueError(f'hidden_size ({hidden_size}) must be divisible by 2.')
#     self.num_freqs_int32 = self.hidden_size // 2

#   def build(self, input_shape):
#     num_freqs = tf.cast(self.num_freqs_int32, tf.float32)
#     log_freq_increment = (
#         tf.math.log(self.max_freq / self.min_freq) /
#         tf.maximum(1.0, num_freqs - 1.0)
#     )
#     # shape: [num_freqs]
#     self.inv_freqs = self.min_freq * tf.exp(
#         tf.range(self.num_freqs_int32, dtype=tf.float32) * -log_freq_increment
#     )

#   def call(self, input_tensor):
#     x = tf.cast(input_tensor, tf.float32)          # [..., C]
#     # angles: [..., C, num_freqs]
#     angles = x[..., tf.newaxis] * self.inv_freqs   # broadcast multiply
#     # emb: [..., C, hidden_size]
#     emb = tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)

#     # Flatten channel dim C into the embedding dim -> [..., C*hidden_size]
#     shape_prefix = tf.shape(x)[:-1]
#     C = tf.shape(x)[-1]
#     out = tf.reshape(emb, tf.concat([shape_prefix, [C * self.hidden_size]], axis=0))
#     return out
