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
"""Converts a set of text files to TFRecord format with Example protos.

Each Example proto in the output contains the following fields:

  decode_pre: list of int64 ids corresponding to the "previous" sentence.
  encode: list of int64 ids corresponding to the "current" sentence.
  decode_post: list of int64 ids corresponding to the "post" sentence.

In addition, the following files are generated:

  vocab.txt: List of "<word> <id>" pairs, where <id> is the integer
             encoding of <word> in the Example protos.
  word_counts.txt: List of "<word> <count>" pairs, where <count> is the number
                   of occurrences of <word> in the input files.

The vocabulary of word ids is constructed from the top --num_words by word
count. All other words get the <unk> word id.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import tensorflow as tf

import special_words
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("textual",None,"")

tf.flags.DEFINE_string("set","train","Can be train or test")

tf.flags.DEFINE_integer("Lambda",40,"Number of sentences per vector")

tf.flags.DEFINE_string("video_embedding",None,"")

tf.flags.DEFINE_integer("train_output_shards", 100,
                        "Number of output shards for the training set.")

tf.flags.DEFINE_integer("validation_output_shards", 1,
                        "Number of output shards for the validation set.")

tf.flags.DEFINE_integer("num_validation_sentences", 0,
                        "Number of output shards for the validation set.")

tf.flags.DEFINE_integer("max_sentences", 0,
                        "If > 0, the maximum number of sentences to output.")

tf.flags.DEFINE_integer("max_sentence_length", 30,
                        "If > 0, exclude sentences whose encode, decode_pre OR"
                        "decode_post sentence exceeds this length.")

tf.flags.DEFINE_integer("max_clip_length", 80, "")

tf.flags.DEFINE_boolean("add_eos", True,
                        "Whether to add end-of-sentence ids to the output.")

tf.logging.set_verbosity(tf.logging.INFO)

vocab_file = "/local/bordes/binaries/bookcorpus_data_"+FLAGS.textual+"/vocab.txt"

#output_dir = "/local/bordes/binaries/A_" +FLAGS.set + "_" + FLAGS.textual 

if FLAGS.video_embedding == "IF":
    feature_file = "/local/bordes/data/image/coco_features.npy"
    input_f = ["/local/bordes/data/image/coco_val.en"]
    indices = np.load("/local/bordes/data/image/coco_val.npy")
    #feature_file = "/local/bordes/datasets/COCO/coco_val_ims_R.npy"
    #input_f = ["/local/bordes/data/image/coco_val.en"]
elif FLAGS.video_embedding == "ranking_random":
    feature_file = "/local/bordes/datasets/MSVD/random_baseline/"
    output_dir = "/local/bordes/binaries/" + str(FLAGS.Lambda) + "_train_" + FLAGS.textual + "_R"
    input_f = ["/local/bordes/datasets/MSVD/sents_all_lc_nopunc.txt"]
elif FLAGS.video_embedding == "A":
    input_f = ["/local/bordes/datasets/MSVD/sents_"+FLAGS.set + "_lc_nopunc.txt"]


output_dir = "/local/bordes/binaries/A_train_SK_R"
input_f = ["/local/bordes/data/image/coco.en"]
sent = []
with open("/local/bordes/data/image/coco.en","r") as f:
    for line in f:
        sent.append(line.replace("\n",""))
indices = np.load("/local/bordes/data/image/coco.npy")
#tab = np.random.randn(np.max(indices)+1,2048)
tab = np.load("/local/bordes/data/image/coco_features.npy")
np.random.shuffle(tab)

def _build_vocabulary(input_files):
  """Loads or builds the model vocabulary.

  Args:
    input_files: List of pre-tokenized input .txt files.

  Returns:
    vocab: A dictionary of word to id.
  """

  tf.logging.info("Loading existing vocab file.")
  vocab = collections.OrderedDict()
  with tf.gfile.GFile(vocab_file, mode="r") as f:
    for i, line in enumerate(f):
      word = line.decode("utf-8").strip()
      if word not in vocab:
        vocab[word] = i
        #assert word not in vocab, "Attempting to add word twice: %s" % word
  tf.logging.info("Read vocab of size %d from %s", len(vocab), vocab_file)
  return vocab


def _int64_feature(value):
    """Helper for creating an Int64 Feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(
      value=[int(v) for v in value]))


def _sentence_to_ids(sentence, vocab):
  """Helper for converting a sentence (list of words) to a list of ids."""

  if FLAGS.textual == "SK":
    ids = [vocab.get(w, special_words.UNK_ID) for w in sentence]
  else:
    ids = []
    for w in sentence:
      if w in vocab:
        ids.append(vocab.get(w, -1))
    if FLAGS.add_eos:
      ids.append(special_words.EOS_ID)
  return ids

    
def _float_feature(value):
    """Helper for creating a FLOAT Feature."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.reshape(-1)))


def _create_serialized_example(current, vocab, clip):
    """Helper for creating a serialized Example proto."""
    example = tf.train.Example(features=tf.train.Features(feature={
      "sentence": _int64_feature(_sentence_to_ids(current, vocab)),
      "image": _float_feature(clip),
  }))
    return example.SerializeToString()


def _process_input_file(filename, vocab, stats):
    """Processes the sentences in an input file.

  Args:
    filename: Path to a pre-tokenized input .txt file.
    vocab: A dictionary of word to id.
    stats: A Counter object for statistics.

  Returns:
    processed: A list of serialized Example protos
  """
    tf.logging.info("Processing input file: %s", filename)
    processed = []

    predecessor = None  # Predecessor sentence (list of words).
    current = None  # Current sentence (list of words).
    successor = None  # Successor sentence (list of words).
    counts = {}

    if FLAGS.video_embedding not in ["A","ranking_random","one_baseline_random"]:
      #image_array = np.load(feature_file)
      image_array = tab
    sen_id = 0

    for successor_str in tf.gfile.FastGFile(filename):
        stats.update(["sentences_seen"])

        if FLAGS.video_embedding in ["ranking_random","one_baseline_random"]:
            clip_id = successor_str.split("\t")[0]
            successor = successor_str.split("\t")[1]
            successor = successor.split()
            image = np.load(feature_file + clip_id + ".npy")
        elif FLAGS.video_embedding == "A":
            clip_id = successor_str.split("\t")[0]
            try:
                successor = successor_str.split("\t")[1]
                successor = successor.split()
            except:
                successor = " "
            feature_id = "/local/bordes/datasets/MSVD/VGG_features/" + clip_id + ".npy"
            try:
                clip = np.load(feature_id)
            except:
                tf.logging.info("Clip %s", feature_id)
                continue
            image = np.mean(clip,0)
        else:
            clip_id = indices[sen_id]
            successor = successor_str.split()
            image = image_array[clip_id]

        sen_id += 1

        if clip_id not in counts:
          counts[clip_id] = 1
        else:
          counts[clip_id] += 1

        if counts[clip_id] <= FLAGS.Lambda:

          stats.update(["sentences_considered"])
        # Note that we are going to insert <EOS> later, so we only allow
        # sentences with strictly less than max_sentence_length to pass.
          if FLAGS.max_sentence_length and (len(successor) >= FLAGS.max_sentence_length):
              stats.update(["sentences_too_long"])
          else:
              serialized = _create_serialized_example(successor, vocab, image)
              processed.append(serialized)
              stats.update(["sentences_output"])

          sentences_seen = stats["sentences_seen"]
          sentences_output = stats["sentences_output"]

          if sentences_seen and sentences_seen % 1300 == 0:
              tf.logging.info("Processed %d sentences (%d output)", sentences_seen,
                            sentences_output)
          if FLAGS.max_sentences and sentences_output >= FLAGS.max_sentences:
              break

    tf.logging.info("Completed processing file %s", filename)
    return processed


def _write_shard(filename, dataset, indices):
    """Writes a TFRecord shard."""
    with tf.python_io.TFRecordWriter(filename) as writer:
        for j in indices:
            writer.write(dataset[j])


def _write_dataset(name, dataset, indices, num_shards):
    """Writes a sharded TFRecord dataset.

  Args:
    name: Name of the dataset (e.g. "train").
    dataset: List of serialized Example protos.
    indices: List of indices of 'dataset' to be written.
    num_shards: The number of output shards.
  """
    tf.logging.info("Writing dataset %s", name)
    borders = np.int32(np.linspace(0, len(indices), num_shards + 1))
    for i in range(num_shards):
        filename = os.path.join(output_dir, "%s-%.5d-of-%.5d" % (name, i,
                                                                   num_shards))
        shard_indices = indices[borders[i]:borders[i + 1]]
        _write_shard(filename, dataset, shard_indices)
        tf.logging.info("Wrote dataset indices [%d, %d) to output shard %s",
                        borders[i], borders[i + 1], filename)
    tf.logging.info("Finished writing %d sentences in dataset %s.",
                  len(indices), name)


def main(unused_argv):
    
    
    if not tf.gfile.IsDirectory(output_dir):
        tf.gfile.MakeDirs(output_dir)
    
    vocab = _build_vocabulary(input_f)

    tf.logging.info("Generating dataset.")
    stats = collections.Counter()
    dataset = []
    for filename in input_f:
        dataset.extend(_process_input_file(filename, vocab, stats))
        if FLAGS.max_sentences and stats["sentences_output"] >= FLAGS.max_sentences:
            break

    tf.logging.info("Generated dataset with %d sentences.", len(dataset))
    for k, v in stats.items():
        tf.logging.info("%s: %d", k, v)

    tf.logging.info("Shuffling dataset.")
    np.random.seed(123)
    shuffled_indices = np.random.permutation(len(dataset))
    val_indices = shuffled_indices[:FLAGS.num_validation_sentences]
    train_indices = shuffled_indices[FLAGS.num_validation_sentences:]

    _write_dataset("train", dataset, train_indices, FLAGS.train_output_shards)
    _write_dataset("validation", dataset, val_indices,
                     FLAGS.validation_output_shards)


if __name__ == "__main__":
    tf.app.run()
