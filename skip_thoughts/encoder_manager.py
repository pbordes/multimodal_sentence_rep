# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Manager class for loading and encoding with multiple skip-thoughts models.

If multiple models are loaded at once then the encode() function returns the
concatenation of the outputs of each model.

Example usage:
  manager = EncoderManager()
  manager.load_model(model_config_1, vocabulary_file_1, embedding_matrix_file_1,
                     checkpoint_path_1)
  manager.load_model(model_config_2, vocabulary_file_2, embedding_matrix_file_2,
                     checkpoint_path_2)
  encodings = manager.encode(data)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os.path

import numpy as np
import tensorflow as tf

import skip_thoughts_encoder


class EncoderManager(object):
  """Manager class for loading and encoding with skip-thoughts models."""

  def __init__(self,  textual, dim = 620):
    self.encoders = []
    self.sessions = []
    self.dim = dim
    self.textual = textual

  def load_model(self, model_config, vocabulary_file, embedding_matrix_file,
                 checkpoint_path):

    tf.logging.info("Reading vocabulary from %s", vocabulary_file)
    with tf.gfile.GFile(vocabulary_file, mode="r") as f:
      lines = list(f.readlines())
    reverse_vocab = [line.decode("utf-8").strip() for line in lines]
    tf.logging.info("Loaded vocabulary with %d words.", len(reverse_vocab))

    tf.logging.info("Loading embedding matrix from %s", embedding_matrix_file)
    # Note: tf.gfile.GFile doesn't work here because np.load() calls f.seek()
    # with 3 arguments.
    embedding_matrix = np.load(embedding_matrix_file)
    tf.logging.info("Loaded embedding matrix with shape %s",
                    embedding_matrix.shape)

    word_embeddings = collections.OrderedDict(
        zip(reverse_vocab, embedding_matrix))

    g = tf.Graph()
    with g.as_default():
      encoder = skip_thoughts_encoder.SkipThoughtsEncoder(word_embeddings)
      restore_model = encoder.build_graph_from_config(model_config,
                                                      checkpoint_path)

    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    sess = tf.Session(graph=g, config=config_proto)
    restore_model(sess)

    self.encoders.append(encoder)
    self.sessions.append(sess)


  def encode(self,
             data,
             use_norm=False,
             verbose=False,
             batch_size=1024,
             use_eos=False):


    encoded = []
    for encoder, sess in zip(self.encoders, self.sessions):
      encoded.append(
          np.array(
              encoder.encode(
                  sess,
                  data,
                  textual = self.textual,
                  use_norm=use_norm,
                  verbose=verbose,
                  batch_size=batch_size,
                  use_eos=use_eos,
                  dim = self.dim)))

    S = np.concatenate(encoded, axis=1)

    return S 



  def encode_video(self,
             data_clip,
             data_length,
             use_norm=False,
             batch_size=100):
    """Encodes a sequence of videos as skip-vision vectors.
    """
    if not self.encoders:
      raise ValueError(
          "Must call load_model at least once before calling encode.")

    encoded = []
    for encoder, sess in zip(self.encoders, self.sessions):
      encoded.append(
          np.array(
              encoder.encode_video(
                  sess,
                  data_clip,
                  data_length)))

    return np.concatenate(encoded, axis=1)

  def close(self):
    """Closes the active TensorFlow Sessions."""
    for sess in self.sessions:
      sess.close()



class EncoderConcat(object):
  """Manager class for loading and encoding with skip-thoughts models."""

  def __init__(self, encoder_multimodal, encoder_SK):
    self.encoder_multimodal = encoder_multimodal
    self.encoder_SK = encoder_SK

  def encode(self,
             data,
             use_norm=False,
             verbose=False,
             batch_size=100,
             use_eos=False):

    S_multi = self.encoder_multimodal.encode(data,use_norm=use_norm, verbose=verbose,batch_size=batch_size,use_eos=use_eos)
    S_SK = self.encoder_SK.encode(data,use_norm=use_norm, verbose=verbose,batch_size=batch_size,use_eos=use_eos)

    return np.concatenate([S_multi,S_SK],1)
