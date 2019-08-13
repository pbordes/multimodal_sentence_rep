#!/usr/bin/env python2
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import configuration
from configuration import parse_flags
from utils import select_gpu
import skip_thoughts_model
import subprocess
import evaluation


tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = parse_flags()
select_gpu(FLAGS.gpu)

lr = float(FLAGS.lr)
lr_decay = float(FLAGS.lr_decay)

train_dir = os.environ["GROUNDSENT_DIR"]

if FLAGS.train_folder :
  train_dir = FLAGS.train_folder
else:
  train_dir += "NEW_text" + str(FLAGS.lambda_text) + "_rho" + str(FLAGS.lambda_rho)  + "_rank" + str(FLAGS.lambda_ranking) + "_rt" + str(FLAGS.lambda_ranking_text)+ "_rtf" + str(FLAGS.lambda_ranking_text_f) + "_rhof" + str(FLAGS.lambda_rho_f)+ "_encode" + str(FLAGS.encoder_train) # + "_cap2cap" + str(FLAGS.lambda_cap2cap) + "_Sinfer" + str(FLAGS.lambda_infer) + "_Iinfer" + str(FLAGS.lambda_Iinfer)
  if FLAGS.textual == "FS":
    train_dir += "_" + FLAGS.textual
  if FLAGS.mu_reg != 0.00025:
    train_dir += "_" + str(FLAGS.mu_reg)
  if int(FLAGS.pretrained) == 0:
    train_dir += "_scratch"
  if FLAGS.encoder_dim != 1024:
    train_dir += "_" + str(FLAGS.encoder_dim) 
  # Random exp_id
  exp_id = np.random.randint(999999)
  train_dir += "/" + str(exp_id)


bin_dir = os.environ["GROUNDSENT_BIN_DIR"]
if int(FLAGS.pretrained) == 1:
  os.system("mkdir -p "  + train_dir)
  os.system("cp " + bin_dir + FLAGS.textual + "_" + str(FLAGS.encoder_dim) + "_2" +  "/* " + train_dir)


input_bookcorpus_pattern = bin_dir + "bookcorpus_data_" + FLAGS.textual + "/train-?????-of-00100"
if FLAGS.input_visual_pattern:
  input_visual_pattern = FLAGS.input_visual_pattern
else:
  input_visual_pattern = bin_dir + str(FLAGS.Lambda)+"_train_" + FLAGS.textual + "/train-?????-of-00100"
validation_pattern = bin_dir + "40_test_" + FLAGS.textual + "/train-?????-of-00100"

if FLAGS.video_embedding == "R":
  input_visual_pattern = bin_dir + "A_train_" + FLAGS.textual + "_R/train-?????-of-00100"
  validation_pattern = bin_dir + "A_test_" + FLAGS.textual + "_R/train-?????-of-00100"

sim_pattern = bin_dir + str(FLAGS.Lambda) +"_sim_train_" + FLAGS.textual + "/train-?????-of-00100"
val_sim_pattern = bin_dir + "sim_test_" + FLAGS.textual + "/train-?????-of-00100"
if FLAGS.video_embedding == "IF":
  input_visual_pattern = bin_dir + "A_train_" + FLAGS.textual + "/train-?????-of-00100"
  validation_pattern = bin_dir + "A_test_" + FLAGS.textual + "/train-?????-of-00100"

sim_pattern = bin_dir + str(FLAGS.Lambda) +"_sim_train_" + FLAGS.textual + "_COCO/train-?????-of-00100"
val_sim_pattern = bin_dir + str(FLAGS.Lambda) +"_sim_test_" + FLAGS.textual + "_COCO/train-?????-of-00100"

nn_pattern = bin_dir + "NN_10_train_COCO/train-?????-of-00100"
val_nn_pattern = bin_dir + "NN_10_val_COCO/train-?????-of-00100"

if FLAGS.pretrained == 0:
  batch = 128
else:
  batch = 128


def optimistic_restore_vars(model_checkpoint_path):

    reader = tf.train.NewCheckpointReader(model_checkpoint_path)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables() if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    return restore_vars


def _setup_learning_rate(config, global_step, setting):

  l_r = float(config.lr)
  if config.learning_rate_decay_factor > 0:
    learning_rate = tf.train.exponential_decay(
        learning_rate=l_r,
        global_step=global_step,
        decay_steps=config.learning_rate_decay_steps,
        decay_rate=config.learning_rate_decay_factor,
        staircase=False)
  else:
    learning_rate = tf.constant(config.lr)
  return learning_rate


def main(unused_argv):

  model_config = configuration.model_config(
      lambda_rho = float(FLAGS.lambda_rho),
      lambda_rho_f = float(FLAGS.lambda_rho_f),
      lambda_ranking = float(FLAGS.lambda_ranking),
      lambda_ranking_text = float(FLAGS.lambda_ranking_text),
      lambda_ranking_text_f = float(FLAGS.lambda_ranking_text_f),
      lambda_cap2cap = float(FLAGS.lambda_cap2cap),
      lambda_infer = float(FLAGS.lambda_infer),
      lambda_quick = float(FLAGS.lambda_quick),
      lambda_Iinfer = float(FLAGS.lambda_Iinfer),
      lambda_nn = float(FLAGS.lambda_nn),
      hidden_dim = FLAGS.hidden_dim,
      train_folder = train_dir,
      distance = FLAGS.distance,
      rank_distance = FLAGS.rank_distance,
      nb_layer = FLAGS.nb_layer,
      word_embedding_dim = FLAGS.word_embedding_dim,
      SICK_trial = os.environ["SICK_TRIAL"],
      bin_dir = os.environ["GROUNDSENT_BIN_DIR"],
      mu_reg = FLAGS.mu_reg,
      val_sim_pattern = val_sim_pattern,
      sim_pattern = sim_pattern,
      nn_pattern = nn_pattern,
      val_nn_pattern = val_nn_pattern,
      validation_pattern = validation_pattern,
      bookcorpus_pattern=input_bookcorpus_pattern,
      visual_pattern=input_visual_pattern,
      encoder_dim = FLAGS.encoder_dim,
      ST = FLAGS.ST,
      video_embedding = FLAGS.video_embedding,
      projected_image_dim = int(FLAGS.projected_im),
      text_negative_number = int(FLAGS.neg_words),
      visual_batch_size = int(FLAGS.v_batch),
      lambda_text = float(FLAGS.lambda_text),
      textual = FLAGS.textual,
      visual_batch = int(FLAGS.v_batch),
      bookcorpus_batch_size = batch)

  training_config = configuration.training_config(lr = lr, learning_rate_decay_factor = lr_decay)

  tf.logging.info("Building training graph.")
  g = tf.Graph()

  with g.as_default():

    model = skip_thoughts_model.SkipThoughtsModel(model_config, mode="train")
    model.build()
    optimizer = tf.train.AdamOptimizer(_setup_learning_rate(training_config, model.global_step,"total"))

    ckpt = tf.train.get_checkpoint_state(train_dir)

    train_total = tf.contrib.slim.learning.create_train_op(
        total_loss=model.total_loss,
        optimizer=optimizer,
        global_step=model.global_step,
        clip_gradient_norm=training_config.clip_gradient_norm,
        variables_to_train=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    if ckpt:
      #ckpt_file = train_dir + "/model.ckpt-" + ckpt.model_checkpoint_path.split("-")[1]
      ckpt_file =  ckpt.model_checkpoint_path
      variables_to_restore = optimistic_restore_vars(ckpt_file)
    else:
      variables_to_restore = None

    saver = tf.train.Saver(var_list=variables_to_restore, max_to_keep=2)

    if FLAGS.encoder_train == 0:
      encoder_variables = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if ("encoder" in v.name) or ("w_embedding" in v.name)]
      saver_encoder = tf.train.Saver(var_list=encoder_variables, max_to_keep=2)
    else:
      encoder_variables = []


    trainable_variables = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v not in encoder_variables]


    train_total = tf.contrib.slim.learning.create_train_op(
        total_loss=model.total_loss,
        optimizer=optimizer,
        global_step=model.global_step,
        clip_gradient_norm=training_config.clip_gradient_norm,
        variables_to_train=trainable_variables)

    if ckpt:
      var_to_init = []
      for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if v not in variables_to_restore:
          if v not in encoder_variables:
            var_to_init.append(v)
      local_init_op = tf.variables_initializer(var_to_init)
    else:
      local_init_op = tf.global_variables_initializer()

    
    #tf.logging.info("Variables to init")
    #for v in var_to_init:
    #  print(v.name)
    #tf.logging.info("Variables to restore")
    #for v in variables_to_restore:
    #  print(v.name)
    
    tf.logging.info("Variables to train")
    for v in trainable_variables:
      print(v.name)


    if ckpt:
      sess = tf.InteractiveSession()
      saver.restore(sess,ckpt.model_checkpoint_path)
      sess.run(local_init_op)
      saver_2 = tf.train.Saver(max_to_keep=2)
      saver_2.save(sess,ckpt.model_checkpoint_path)
      current_step = sess.run(model.global_step)
      sess.close()
    else:
      saver_2 = tf.train.Saver(max_to_keep=2)


    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True


    tf.contrib.slim.learning.train(
      session_config = session_config,
      train_op=train_total,
      saver=saver_2,
      local_init_op=local_init_op,
      logdir=train_dir,
      graph=g,
      global_step=model.global_step,
      number_of_steps=FLAGS.nb_steps,
      save_summaries_secs=training_config.save_summaries_secs,
      save_interval_secs=training_config.save_model_secs)


  evaluation.evaluate(train_dir, FLAGS.encoder_dim, "T", FLAGS.textual, 0)



if __name__ == "__main__":
  tf.app.run()
