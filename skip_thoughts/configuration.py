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
"""Default configuration for model architecture and training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def parse_flags():

    tf.flags.DEFINE_string("gpu", "", "the gpu number to use, empty string to use cpu")
    tf.flags.DEFINE_float("lambda_text", 0, "influence of textual model")
    tf.flags.DEFINE_float("lambda_rho", 0, "parameter for loss_rho")
    tf.flags.DEFINE_float("lambda_rho_f", 0, "parameter for loss_rho_f")
    tf.flags.DEFINE_float("lambda_ranking", 0, "parameter for loss_ranking")
    tf.flags.DEFINE_float("lambda_ranking_text", 0, "parameter for loss_ranking_text")
    tf.flags.DEFINE_float("lambda_ranking_text_f", 0, "parameter for loss_ranking_text_f")
    tf.flags.DEFINE_float("lambda_cap2cap", 0, "")
    tf.flags.DEFINE_float("lambda_infer", 0, "")
    tf.flags.DEFINE_float("lambda_Iinfer", 0, "")
    tf.flags.DEFINE_float("lambda_quick", 0, "")
    tf.flags.DEFINE_float("lambda_nn", 0, "")

    tf.flags.DEFINE_string("distance", "rho", "rho or L2")
    tf.flags.DEFINE_string("rank_distance", "triplet", "triplet or L2")
    tf.flags.DEFINE_integer("nb_layer", 0, "nb hidden layer (0 or 1)")
    tf.flags.DEFINE_integer("hidden_dim", 2048, "hidden dim in mlp")
    tf.flags.DEFINE_integer("encoder_train", 1,"1 if we retroprogate through the encoder")
    tf.flags.DEFINE_integer("pretrained", 1,"1 if we load a pretrained SK model")
    
    tf.flags.DEFINE_string("textual", "SK" ,"Textual loss: either SK (Skip-Thought) or FS (FastSent) or FSAE")
    tf.flags.DEFINE_integer("Lambda", 40 ,"number of sentences per visual vector")
    tf.flags.DEFINE_string("input_visual_pattern",None, "")
    tf.flags.DEFINE_string("train_folder",None, "Directory containing model checkpoints.")
    tf.flags.DEFINE_float("lr", 0.0008 ,"learning rate")
    tf.flags.DEFINE_float("lr_decay", 0.5 ,"learning rate")
    tf.flags.DEFINE_integer("nb_steps", 410000 , "Number of optimization steps.")
    tf.flags.DEFINE_integer("neg_words", 1 ,"number of negative words for FastSent")
    tf.flags.DEFINE_float("mu_reg", 0.00025 ,"regularization constant") #0.00025
    tf.flags.DEFINE_integer("v_batch", 32 ,"visual batch size")
    tf.flags.DEFINE_integer("word_embedding_dim", 620 ,"word embedding dimension")
    tf.flags.DEFINE_integer("encoder_dim",2048 ,"dimension of sentence vector")
    tf.flags.DEFINE_integer("ST", 1 ,"0 is for the kiela baseline")
    tf.flags.DEFINE_integer("projected_im", 1000 ,"dimension of linear image projection")
    tf.flags.DEFINE_string("video_embedding","IF" ,"R, VG-F, VG-A or TG")

    return tf.flags.FLAGS





class _HParams(object):
  """Wrapper for configuration parameters."""
  pass


def model_config(bookcorpus_pattern=None,
                 validation_pattern = None,
                 val_sim_pattern = None,
                 visual_pattern=None,
                 sim_pattern=None,
                 SICK_trial=None,
                 rank_distance= None,
                 train_folder = None,
                 ST = None,
                 nn_pattern = None,
                 val_nn_pattern = None,
                 lambda_nn = None,
                 distance = None,
                 word_embedding_dim=None,
                 nb_layer = None,
                 encoder_dim= 2400,
                 lambda_Iinfer= None,
                 textual = None,
                 bookcorpus_batch_size=None,
                 lambda_text = 0,
                 video_embedding = None,
                 video_encoder_dim=2048,
                 lambda_rho = 0,
                 lambda_cap2cap = 0,
                 lambda_quick = 0,
                 lambda_infer = 0,
                 lambda_rho_f = 0,
                 lambda_ranking = 0,
                 lambda_ranking_text = 0,
                 lambda_ranking_text_f = 0,
                 hidden_dim=2048,
                 val_batch_size = 100,
                 sim_val_batch_size = 100,
                 sim_batch_size = 32,
                 word_encode_dim = 100,
                 len_sentence = 30,
                 text_negative_number = 5,
                 bookcorpus_capacity=640000,
                 visual_capacity=1000,
                 bin_dir = None,
                 num_input_reader_threads=1,
                 shuffle_input_data=True,
                 uniform_init_scale=0.1,
                 mu_reg = 0.00025,
                 visual_batch_size = 32,
                 bidirectional_encoder=False,
                 n_video_lstm_step = 80,
                 projected_image_dim = 1000,
                 n_caption_lstm_step = 30,
                 gamma_sentences = 0.5,
                 gamma_videos = 0.5,
                 dim_image = 2048,
                 visual_batch = 100):

  config = _HParams()
  config.lambda_ranking = lambda_ranking
  config.bin_dir = bin_dir
  config.lambda_quick = lambda_quick
  config.rank_distance = rank_distance
  config.SICK_trial = SICK_trial
  config.nb_layer = nb_layer
  config.lambda_ranking_text = lambda_ranking_text
  config.lambda_rho = lambda_rho
  config.lambda_cap2cap = lambda_cap2cap
  config.lambda_rho_f = lambda_rho_f
  config.visual_batch = visual_batch
  config.n_video_lstm_step = n_video_lstm_step
  config.n_caption_lstm_step = n_caption_lstm_step
  config.video_embedding = video_embedding
  config.lambda_nn = lambda_nn
  config.text_negative_number = text_negative_number
  config.len_sentence = len_sentence
  config.train_folder = train_folder
  config.word_encode_dim = word_encode_dim
  config.projected_image_dim = projected_image_dim
  config.mu_reg = mu_reg
  config.lambda_ranking_text_f = lambda_ranking_text_f
  config.hidden_dim = hidden_dim
  config.lambda_infer = lambda_infer
  config.lambda_Iinfer = lambda_Iinfer

  config.nn_pattern = nn_pattern
  config.val_nn_pattern = val_nn_pattern
  config.validation_pattern = validation_pattern
  config.bookcorpus_pattern = bookcorpus_pattern
  config.visual_pattern = visual_pattern
  config.val_sim_pattern = val_sim_pattern
  config.sim_pattern = sim_pattern
  config.bookcorpus_capacity = bookcorpus_capacity
  config.visual_capacity = visual_capacity
  config.visual_batch_size = visual_batch_size
  config.val_batch_size = val_batch_size
  config.sim_batch_size = sim_batch_size
  config.sim_val_batch_size = sim_val_batch_size
  config.ST = ST
  config.distance = distance

  config.num_input_reader_threads = num_input_reader_threads
  config.shuffle_input_data = shuffle_input_data
  config.uniform_init_scale = uniform_init_scale
  config.bidirectional_encoder = bidirectional_encoder

  config.dim_image = dim_image
  if (video_embedding == "VG-T"):
    config.video_encoder_dim = 1000
  elif (video_embedding == "O"):
    config.video_encoder_dim = 80
  else:
    config.video_encoder_dim = 2048

  config.word_embedding_dim = word_embedding_dim
  if textual == "SK":

    config.encoder_dim = encoder_dim
    config.vocab_size=20000
  else:
    config.word_embedding_dim = 100
    config.encoder_dim = config.word_embedding_dim
    config.vocab_size=200000

  config.bookcorpus_batch_size = bookcorpus_batch_size
  config.gamma_sentences = gamma_sentences
  config.gamma_videos = gamma_videos
  config.lambda_text = lambda_text
  config.textual = textual

  return config


def training_config(lr=0.0008,
                    learning_rate_decay_factor=0.5,
                    learning_rate_decay_steps=400000,
                    clip_gradient_norm=5.0,
                    save_model_secs=60,
                    save_summaries_secs=10):
  """Creates a training configuration object.

  Args:
    learning_rate: Initial learning rate.
    learning_rate_decay_factor: If > 0, the learning rate decay factor.
    learning_rate_decay_steps: The number of steps before the learning rate
      decays by learning_rate_decay_factor.
    number_of_steps: The total number of training steps to run. Passing None
      will cause the training script to run indefinitely.
    clip_gradient_norm: If not None, then clip gradients to this value.
    save_model_secs: How often (in seconds) to save model checkpoints.
    save_summaries_secs: How often (in seconds) to save model summaries.

  Returns:
    An object containing training configuration parameters.

  Raises:
    ValueError: If learning_rate_decay_factor is set and
      learning_rate_decay_steps is unset.
  """
  if learning_rate_decay_factor and not learning_rate_decay_steps:
    raise ValueError(
        "learning_rate_decay_factor requires learning_rate_decay_steps.")

  config = _HParams()

  config.lr = lr
  config.learning_rate_decay_factor = learning_rate_decay_factor
  config.learning_rate_decay_steps = learning_rate_decay_steps

  config.clip_gradient_norm = clip_gradient_norm
  config.save_model_secs = save_model_secs
  config.save_summaries_secs = save_summaries_secs
  return config


