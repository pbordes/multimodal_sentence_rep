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
"""Input ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


import tensorflow as tf

# A SentenceBatch is a pair of Tensors:
#  ids: Batch of input sentences represented as sequences of word ids: an int64
#    Tensor with shape [batch_size, padded_length].
#  mask: Boolean mask distinguishing real words (1) from padded words (0): an
#    int32 Tensor with shape [batch_size, padded_length].
SentenceBatch = collections.namedtuple("SentenceBatch", ("ids", "mask"))
ClipBatch = collections.namedtuple("ClipBatch", ("ids", "mask"))



def prefetch_input_data(reader,
                        file_pattern,
                        shuffle,
                        capacity,
                        dataset,
                        num_reader_threads=1):
  """Prefetches string values from disk into an input queue.

  Args:
    reader: Instance of tf.ReaderBase.
    file_pattern: Comma-separated list of file patterns (e.g.
        "/tmp/train_data-?????-of-00100", where '?' acts as a wildcard that
        matches any character).
    shuffle: Boolean; whether to randomly shuffle the input data.
    capacity: Queue capacity (number of records).
    num_reader_threads: Number of reader threads feeding into the queue.

  Returns:
    A Queue containing prefetched string values.
  """
  data_files = []
  print("file_pattern is ", file_pattern)
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s", len(data_files), file_pattern)

  filename_queue = tf.train.string_input_producer(data_files, shuffle=shuffle, capacity=16, name="filename_queue" + dataset)

  if shuffle:
    min_after_dequeue = int(0.6 * capacity)
    values_queue = tf.RandomShuffleQueue(
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        dtypes=[tf.string],
        shapes=[[]],
        name="random_input_queue"+dataset)
  else:
    values_queue = tf.FIFOQueue(
        capacity=capacity,
        dtypes=[tf.string],
        shapes=[[]],
        name="fifo_input_queue"+dataset)

  enqueue_ops = []
  for _ in range(num_reader_threads):
    key, value = reader.read(filename_queue)
    enqueue_ops.append(values_queue.enqueue([value]))
    

  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(values_queue, enqueue_ops))
  #tf.train.queue_runner.add_queue_runner(video_queue)       ### MAYBE IMPORTANT
  #tf.summary.scalar("queue/%s/fraction_of_%d_full" % (values_queue.name,capacity),tf.cast(values_queue.size(), tf.float32) * (1.0 / capacity))

  return values_queue



def parse_video_example_batch(serialized):
  """Parses a batch of tf.Example protos.

  Args:
    serialized: A 1-D string Tensor; a batch of serialized tf.Example protos.
  Returns:
    encode: A SentenceBatch of encode sentences.
    decode_pre: A SentenceBatch of "previous" sentences to decode.
    decode_post: A SentenceBatch of "post" sentences to decode.
  """
  features = tf.parse_example(
      serialized,
      features={
          "encode": tf.VarLenFeature(dtype=tf.int64),
          "clip": tf.VarLenFeature(dtype=tf.float32),
          "nb_frames": tf.FixedLenFeature([],dtype = tf.int64)})
  
  def _sparse_to_batch(sparse):
    ids = tf.sparse_tensor_to_dense(sparse)  # Padding with zeroes.
    mask = tf.sparse_to_dense(sparse.indices, sparse.dense_shape,
                                  tf.ones_like(sparse.values, dtype=tf.int32))
    return SentenceBatch(ids=ids, mask=mask)

  return (_sparse_to_batch(features["encode"]),tf.sparse_tensor_to_dense(features["clip"]), features["nb_frames"])



def parse_image_example_batch(serialized):
  """Parses a batch of tf.Example protos.

  Args:
    serialized: A 1-D string Tensor; a batch of serialized tf.Example protos.
  Returns:
    encode: A SentenceBatch of encode sentences.
    decode_pre: A SentenceBatch of "previous" sentences to decode.
    decode_post: A SentenceBatch of "post" sentences to decode.
  """
  features = tf.parse_example(
      serialized,
      features={
          "sentence": tf.VarLenFeature(dtype=tf.int64),
          "image": tf.VarLenFeature(dtype=tf.float32)})
  
  def _sparse_to_batch(sparse):
    ids = tf.sparse_tensor_to_dense(sparse)  # Padding with zeroes.
    mask = tf.sparse_to_dense(sparse.indices, sparse.dense_shape,
                                  tf.ones_like(sparse.values, dtype=tf.int32))
    return SentenceBatch(ids=ids, mask=mask)

  return (_sparse_to_batch(features["sentence"]),tf.sparse_tensor_to_dense(features["image"]))

def parse_pair_example_batch(serialized):
  """Parses a batch of tf.Example protos.

  Args:
    serialized: A 1-D string Tensor; a batch of serialized tf.Example protos.
  Returns:
    encode: A SentenceBatch of encode sentences.
    decode_pre: A SentenceBatch of "previous" sentences to decode.
    decode_post: A SentenceBatch of "post" sentences to decode.
  """
  features = tf.parse_example(
      serialized,
      features={
          "s1": tf.VarLenFeature(dtype=tf.int64),
          "s2": tf.VarLenFeature(dtype=tf.int64),})
  
  def _sparse_to_batch(sparse):
    ids = tf.sparse_tensor_to_dense(sparse)  # Padding with zeroes.
    mask = tf.sparse_to_dense(sparse.indices, sparse.dense_shape,
                                  tf.ones_like(sparse.values, dtype=tf.int32))
    return SentenceBatch(ids=ids, mask=mask)

  return (_sparse_to_batch(features["s1"]),_sparse_to_batch(features["s2"]))

def parse_HL_batch(serialized):
  """Parses a batch of tf.Example protos.

  Args:
    serialized: A 1-D string Tensor; a batch of serialized tf.Example protos.
  Returns:
    encode: A SentenceBatch of encode sentences.
    decode_pre: A SentenceBatch of "previous" sentences to decode.
    decode_post: A SentenceBatch of "post" sentences to decode.
  """
  features = tf.parse_example(
      serialized,
      features={
          "encode": tf.VarLenFeature(dtype=tf.int64),
          "feature": tf.VarLenFeature(dtype=tf.float32)})
  
  def _sparse_to_batch(sparse):
    ids = tf.sparse_tensor_to_dense(sparse)  # Padding with zeroes.
    mask = tf.sparse_to_dense(sparse.indices, sparse.dense_shape,
                                  tf.ones_like(sparse.values, dtype=tf.int32))
    return SentenceBatch(ids=ids, mask=mask)

  return (_sparse_to_batch(features["encode"]),tf.sparse_tensor_to_dense(features["feature"]))





def parse_text_batch(serialized):
  """Parses a batch of tf.Example protos.

  Args:
    serialized: A 1-D string Tensor; a batch of serialized tf.Example protos.
  Returns:
    encode: A SentenceBatch of encode sentences.
    decode_pre: A SentenceBatch of "previous" sentences to decode.
    decode_post: A SentenceBatch of "post" sentences to decode.
  """
  features = tf.parse_example(
      serialized,
      features={
          "encode": tf.VarLenFeature(dtype=tf.int64),
          "decode_pre": tf.VarLenFeature(dtype=tf.int64),
          "decode_post": tf.VarLenFeature(dtype=tf.int64)})

  def _sparse_to_batch(sparse):
    ids = tf.sparse_tensor_to_dense(sparse)  # Padding with zeroes.
    mask = tf.sparse_to_dense(sparse.indices, sparse.dense_shape,
                              tf.ones_like(sparse.values, dtype=tf.int32))
    return SentenceBatch(ids=ids, mask=mask)

  output_names = ("encode", "decode_pre", "decode_post")
  return tuple(_sparse_to_batch(features[x]) for x in output_names)



