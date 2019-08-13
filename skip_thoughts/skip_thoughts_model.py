from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell

from ops import gru_cell
from ops import input_ops
import subprocess
import encoder_manager
import senteval
import configuration

def get_zipf_mat(vocab_size):
    m = np.arange(vocab_size)
    m = (np.log(m + 2) - np.log(m + 1)) / np.log(vocab_size + 1)
    return m


def random_orthonormal_initializer(shape, dtype=tf.float32,
                                   partition_info=None):  # pylint: disable=unused-argument
  """Variable initializer that produces a random orthonormal matrix."""
  if len(shape) != 2 or shape[0] != shape[1]:
    raise ValueError("Expecting square shape, got %s" % shape)
  _, u, _ = tf.svd(tf.random_normal(shape, dtype=dtype), full_matrices=True)
  return u

def pad_up_to(t, max_in_dims, constant_values):
  s = tf.shape(t)
  paddings = [[0, m-s[i]] for (i,m) in enumerate(max_in_dims)]
  return tf.pad(t, paddings, 'CONSTANT')


class SkipThoughtsModel(object):
  """Skip-thoughts model."""

  def __init__(self, config, mode="train", input_reader=None):

    if mode not in ["train", "eval", "encode"]:
      raise ValueError("Unrecognized mode: %s" % mode)

    self.config = config
    self.mode = mode
    self.reader_bookcorpus = input_reader if input_reader else tf.TFRecordReader()
    self.reader_visual = input_reader if input_reader else tf.TFRecordReader()
    self.reader_val = input_reader if input_reader else tf.TFRecordReader()
    self.reader_sim = input_reader if input_reader else tf.TFRecordReader()
    self.reader_sim_val = input_reader if input_reader else tf.TFRecordReader()
    self.reader_nn = input_reader if input_reader else tf.TFRecordReader()
    self.reader_nn_val = input_reader if input_reader else tf.TFRecordReader()

    # Initializer used for non-recurrent weights.
    self.uniform_initializer = tf.random_uniform_initializer(
        minval=-self.config.uniform_init_scale,
        maxval=self.config.uniform_init_scale)

    self.n_video_lstm_step = config.n_video_lstm_step
    self.n_caption_lstm_step = config.n_caption_lstm_step
    
    if self.config.textual == "SK":
      name = "w_embedding/word_embedding"
    else:
      name = "word_embedding"

    word_emb = tf.get_variable(
        name = name,
        shape=[self.config.vocab_size, self.config.word_embedding_dim],
        initializer=self.uniform_initializer,
        trainable = True)

    self.word_emb = word_emb





    if self.config.textual != "SK":
      self.word_target = tf.get_variable(
          name = "word_target",
          shape=[self.config.vocab_size, self.config.word_embedding_dim],
          initializer=self.uniform_initializer,
          trainable = True)

    if self.config.video_embedding != "T":
      with tf.variable_scope('mapping') as scope:
          self.W1 = tf.get_variable(name='W1',
                            shape=[self.config.encoder_dim,self.config.hidden_dim],
                            initializer=tf.random_uniform_initializer(minval=-0.05,maxval=0.05),
                            trainable = True)
          self.b1 = tf.get_variable(name='b1',
                            shape=[self.config.hidden_dim],
                            initializer=tf.random_uniform_initializer(minval=-0.05,maxval=0.05),
                            trainable = True)
          self.W2 = tf.get_variable(name='W2',
                            shape=[self.config.hidden_dim,self.config.video_encoder_dim],
                            initializer=tf.random_uniform_initializer(minval=-0.05,maxval=0.05),
                            trainable = True)
          self.b2 = tf.get_variable(name='b2',
                            shape=[self.config.video_encoder_dim],
                            initializer=tf.random_uniform_initializer(minval=-0.05,maxval=0.05),
                            trainable = True)
    """
    with tf.variable_scope('mapping_image') as scope:
        self.W1_image = tf.get_variable(name='W1_image',
                          shape=[self.config.video_encoder_dim,self.config.video_encoder_dim],
                          initializer=tf.random_uniform_initializer(minval=-0.05,maxval=0.05),
                          trainable = True)
        self.b1_image = tf.get_variable(name='b1_image',
                          shape=[self.config.video_encoder_dim],
                          initializer=tf.random_uniform_initializer(minval=-0.05,maxval=0.05),
                          trainable = True)
        self.W2_image = tf.get_variable(name='W2_image',
                          shape=[self.config.video_encoder_dim,self.config.video_encoder_dim],
                          initializer=tf.random_uniform_initializer(minval=-0.05,maxval=0.05),
                          trainable = True)
        self.b2_image = tf.get_variable(name='b2_image',
                          shape=[self.config.video_encoder_dim],
                          initializer=tf.random_uniform_initializer(minval=-0.05,maxval=0.05),
                          trainable = True)
    """

  def feed_foward_NN(self,x):
      if self.config.nb_layer == 1:
          return tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(x, self.W1), self.b1)), self.W2),self.b2)
      elif self.config.nb_layer == 0:
          return tf.add(tf.matmul(x, self.W1), self.b1)

  def feed_foward_NN_image(self,x):
      if self.config.nb_layer == 1:
          return tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(x, self.W1_image), self.b1_image)), self.W2_image),self.b2_image)
      elif self.config.nb_layer == 0:
          return tf.add(tf.matmul(x, self.W1_image), self.b1_image)

  def _initialize_gru_cell(self, num_units, trainable):

    return gru_cell.LayerNormGRUCell(
        num_units,
        w_initializer=self.uniform_initializer,
        u_initializer=random_orthonormal_initializer,
        b_initializer=tf.constant_initializer(0.0),
        trainable = trainable)


  def extract_axis_1(self,data, ind):

    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    return res


  def mask_ids(self, data, vocab):

    nb_s = len(data)
    ids = np.zeros([nb_s,70])
    mask = np.zeros([nb_s,70])

    for i in range(nb_s):

      sentence = [w.lower() for w in data[i].split(" ")]

      if self.config.textual == "SK":
        id_s = [vocab.get(w, 1) for w in sentence]
        len_s = len(sentence)
      else:
        id_s = []
        len_s = 0
        for w in sentence:
          if w in vocab:
            len_s += 1
            id_s.append(vocab.get(w, -1))
      ids[i][:len_s] = id_s
      mask[i][:len_s] = np.ones([len_s])

    return ids,mask

  def pearson(self, proj = False):

    devA, devB, devS = [], [], []
    with open(self.config.SICK_trial, 'rb') as f:
      for line in f:
          text = line.strip().split('\t')
          devA.append(text[1].decode('utf8'))
          devB.append(text[2].decode('utf8'))
          devS.append(text[3].decode('utf8'))
    devA, devB, devS = devA[1:], devB[1:], [float(s) for s in devS[1:]]

    # Load vocabulary
    vocab = collections.OrderedDict()
    with tf.gfile.GFile(os.path.join(self.config.bin_dir+"bookcorpus_data_"+self.config.textual,"vocab.txt"), mode="r") as f:
      for i, line in enumerate(f):
        word = line.decode("utf-8").strip()
        if word not in vocab:
          vocab[word] = i

    s1_ids, s1_mask = self.mask_ids(devA, vocab)
    s2_ids, s2_mask = self.mask_ids(devB, vocab)
    human_scores = devS

    s1_emb = tf.nn.embedding_lookup(self.word_emb, tf.cast(s1_ids,tf.int32))
    s2_emb = tf.nn.embedding_lookup(self.word_emb, tf.cast(s2_ids,tf.int32))
    encoded_s1 = self.text_encoder(s1_emb, s1_mask, reuse = True)
    encoded_s2 = self.text_encoder(s2_emb, s2_mask, reuse = True)

    if proj:
      encoded_s1 = self.feed_foward_NN(encoded_s1)
      encoded_s2 = self.feed_foward_NN(encoded_s2)

    s1 = tf.nn.l2_normalize(encoded_s1, 1)
    s2 = tf.nn.l2_normalize(encoded_s2, 1)
    scores = tf.reduce_sum(tf.multiply(s1,s2),1)
    scores = scores - tf.reduce_mean(scores)
    human_scores = tf.cast(human_scores - tf.reduce_mean(human_scores),tf.float32)
    sigma_scores = tf.sqrt(tf.reduce_mean(tf.multiply(scores,scores)))
    sigma_human_scores = tf.sqrt(tf.reduce_mean(tf.multiply(human_scores,human_scores)))
    rho =  tf.reduce_mean(tf.multiply(scores,human_scores))/(sigma_scores*sigma_human_scores)

    return rho

  def cosine(self,x,y):
      s1 = tf.nn.l2_normalize(x, 1)
      s2 = tf.nn.l2_normalize(y, 1)
      return tf.matmul(s1,tf.transpose(s2))

  def NN(self, x, k):
      distance = self.cosine(x, x)
      return  tf.nn.top_k(distance, k=k).indices


  def mNNO(self, x, y, K, depth, batch):

      nnx = self.NN(x, k=K+1)[:,1:]
      nny = self.NN(y, k=K+1)[:,1:]
      return tf.reduce_sum(tf.multiply(tf.reduce_sum(tf.one_hot(nnx,depth=depth,axis=-1),1), tf.reduce_sum(tf.one_hot(nny,depth=depth,axis=-1),1)))/(batch*K)

  def compute_mNNO_dev(self, K):

      
      sent = []
      with open(os.path.join(os.environ["CODE_DIR"],"coco_val.en"), "r") as f:
          for line in f:
              sent.append(line.replace("\n",""))
      sent = sent[:1000]

      # Load vocabulary
      vocab = collections.OrderedDict()
      with tf.gfile.GFile(os.path.join(self.config.bin_dir + "bookcorpus_data_"+self.config.textual,"vocab.txt"), mode="r") as f:
        for i, line in enumerate(f):
          word = line.decode("utf-8").strip()
          if word not in vocab:
            vocab[word] = i


      s_ids, s_mask = self.mask_ids(sent, vocab)
      s_emb = tf.nn.embedding_lookup(self.word_emb, tf.cast(s_ids,tf.int32))
      encoded_s = self.text_encoder(s_emb, s_mask, reuse = True)
      
      encoded_im = tf.convert_to_tensor(np.load(os.path.join(os.environ["CODE_DIR"],"coco_dev_ims.npy")))
      encoded_im = encoded_im[:1000,:]
      
      #encoded_s = tf.convert_to_tensor(np.load("/net/sister/bordes/coco_skipthought_eval.npy")[:1000,:])

      tf.summary.scalar("eval/mNNO_10_x_y" , self.mNNO(encoded_s, encoded_im, 10, 1000, 1000))
      tf.summary.scalar("eval/mNNO_10_fx_y" , self.mNNO(self.feed_foward_NN(encoded_s), encoded_im, 10, 1000, 1000))
      tf.summary.scalar("eval/mNNO_10_x_fx" , self.mNNO(encoded_s, self.feed_foward_NN(encoded_s), 10, 1000, 1000))
      tf.summary.scalar("eval/mNNO_10_x_x" , self.mNNO(encoded_s, encoded_s, 10, 1000, 1000))


  def build_bookcorpus_inputs(self):


    if self.mode == "encode":
      # Word embeddings are fed from an external vocabulary which has possibly
      # been expanded (see vocabulary_expansion.py).
      encode_ids = None
      decode_pre_ids = None
      decode_post_ids = None
      encode_mask = tf.placeholder(tf.int8, (None, None), name="encode_mask")
      decode_pre_mask = None
      decode_post_mask = None
    else:
        # Prefetch serialized tf.Example protos.
      input_queue = input_ops.prefetch_input_data(
          self.reader_bookcorpus,
          self.config.bookcorpus_pattern,
          shuffle=self.config.shuffle_input_data,
          capacity=self.config.bookcorpus_capacity,
          num_reader_threads=self.config.num_input_reader_threads,
          dataset="bookcorpus")

      # Deserialize a batch.
      serialized = input_queue.dequeue_many(self.config.bookcorpus_batch_size)
      encode, decode_pre, decode_post = input_ops.parse_text_batch(
          serialized)

      encode_ids = pad_up_to(encode.ids, [self.config.bookcorpus_batch_size, self.config.len_sentence], 0)
      decode_pre_ids = pad_up_to(decode_pre.ids, [self.config.bookcorpus_batch_size, self.config.len_sentence], 0)
      decode_post_ids = pad_up_to(decode_post.ids, [self.config.bookcorpus_batch_size, self.config.len_sentence], 0)

      encode_mask = pad_up_to(encode.mask, [self.config.bookcorpus_batch_size, self.config.len_sentence], 0)
      decode_pre_mask = pad_up_to(decode_pre.mask, [self.config.bookcorpus_batch_size, self.config.len_sentence], 0)
      decode_post_mask = pad_up_to(decode_pre.mask, [self.config.bookcorpus_batch_size, self.config.len_sentence], 0)

    self.encode_ids = encode_ids
    self.decode_pre_ids = decode_pre_ids
    self.decode_post_ids = decode_post_ids

    self.encode_mask = encode_mask
    self.decode_pre_mask = decode_pre_mask
    self.decode_post_mask = decode_post_mask

    if self.mode == "encode":
      self.encode_emb = tf.placeholder(tf.float32, (None, None, self.config.word_embedding_dim), "encode_emb")
      self.decode_pre_emb = None
      self.decode_post_emb = None
    else:
      self.decode_pre_emb = tf.nn.embedding_lookup(self.word_emb, self.decode_pre_ids)
      self.decode_post_emb = tf.nn.embedding_lookup(self.word_emb, self.decode_post_ids)
      self.encode_emb = tf.nn.embedding_lookup(self.word_emb, self.encode_ids)


    self.thought_vectors = tf.identity(self.text_encoder(self.encode_emb, self.encode_mask, reuse = (self.config.video_embedding != "T")), name = "thought_vectors")

  def text_encoder(self,encode_emb, encode_mask, reuse ):
    if self.config.textual == "SK":
      return self.SK_encoder(encode_emb, encode_mask, reuse = reuse)
    else:
      return self.FS_encoder(encode_emb, encode_mask)


  def build_MSVD_inputs(self, reader, pattern, batch_size, reuse):


    if self.mode == "encode":

      caption = None
      input_MSVD_vector = tf.placeholder(tf.float32, (None, None), name="video_vector")
      video = tf.placeholder(tf.float32, (None, None, self.config.dim_image), name="video")
      video_length = tf.placeholder(tf.float32, (None), name="video_length")
      MSVD_mask = tf.placeholder(tf.int32, (None, None), name="caption_mask")
      MSVD_ids = tf.placeholder(tf.int32, (None, None), name="caption_ids")
      nb_sentences = tf.placeholder(tf.int32, name="nb_sentences")
      MSVD_emb = tf.placeholder(tf.float32, (None, None, self.config.word_embedding_dim), "caption_emb")

      encoded_caption = self.text_encoder(MSVD_emb, MSVD_mask, reuse = reuse)
      encoded_visual = self.video_encoder(video, video_length, batch_size = batch_size)

    else:

      input_queue = input_ops.prefetch_input_data(
        reader,
        pattern,
        shuffle=self.config.shuffle_input_data,
        capacity=self.config.visual_capacity,
        num_reader_threads=self.config.num_input_reader_threads,
        dataset="MSVD")

      # Deserialize a batch.
      serialized = input_queue.dequeue_many(batch_size)
      caption, video, video_length = input_ops.parse_video_example_batch(serialized)
      video = tf.reshape(video,[batch_size,-1,self.config.dim_image])

      MSVD_ids = caption.ids
      MSVD_mask = caption.mask

      caption_word_emb = tf.nn.embedding_lookup(self.word_emb, MSVD_ids)
      encoded_caption = self.text_encoder(caption_word_emb, MSVD_mask, reuse = reuse)

    caption_W2V_emb = tf.reduce_sum(tf.nn.embedding_lookup(self.w2v_matrix, MSVD_ids),1)
    encoded_visual = self.video_encoder(video, video_length, h_t = caption_W2V_emb, batch_size = batch_size)

    if self.config.video_embedding == "TG":
      return MSVD_ids, encoded_caption, video, video_length, encoded_visual
    else:
      return encoded_caption, encoded_visual




  def build_ranking_text_inputs(self, reader, pattern, batch_size, reuse):
    # Prefetch serialized tf.Example protos.
    input_queue = input_ops.prefetch_input_data(
        reader,
        pattern,
        shuffle=self.config.shuffle_input_data,
        capacity=self.config.visual_capacity,
        num_reader_threads=self.config.num_input_reader_threads,
        dataset="COCO")

    # Deserialize a batch.
    serialized = input_queue.dequeue_many(batch_size)

    s1, s2 = input_ops.parse_pair_example_batch(serialized)

    s1_ids = s1.ids
    s1_mask = s1.mask
    s1_word_emb = tf.nn.embedding_lookup(self.word_emb, s1_ids)
    s2_ids = s2.ids
    s2_mask = s2.mask
    s2_word_emb = tf.nn.embedding_lookup(self.word_emb, s2_ids)

    encoded_s1 = self.text_encoder(s1_word_emb, s1_mask, reuse = reuse)
    encoded_s2 = self.text_encoder(s2_word_emb, s2_mask, reuse = True)


    return encoded_s1, encoded_s2, s2_word_emb, s2_ids, s2_mask




  def build_COCO_inputs(self, reader, pattern, batch_size, reuse):
    # Prefetch serialized tf.Example protos.
    input_queue = input_ops.prefetch_input_data(
        reader,
        pattern,
        shuffle=self.config.shuffle_input_data,
        capacity=self.config.visual_capacity,
        num_reader_threads=self.config.num_input_reader_threads,
        dataset="COCO")

    serialized = input_queue.dequeue_many(batch_size)

    if (self.config.video_embedding == "O"):
      caption, indices = input_ops.parse_HL_batch(serialized)
      indices = tf.cast(indices, tf.int32)
      input_COCO_vector = tf.one_hot(indices[:,0], 80)

    if (self.config.video_embedding == "R") or (self.config.video_embedding == "IF"):
      caption, input_COCO_vector = input_ops.parse_image_example_batch(serialized)

    encoded_visual = input_COCO_vector

    COCO_ids = caption.ids
    COCO_mask = caption.mask
    caption_word_emb = tf.nn.embedding_lookup(self.word_emb, COCO_ids)

    encoded_caption = self.text_encoder(caption_word_emb, COCO_mask, reuse = reuse)

    return encoded_caption, encoded_visual



  def SK_encoder(self, encode_emb, encode_mask, reuse):


    with tf.variable_scope("encoder", reuse = reuse) as scope:
      length = tf.to_int32(tf.reduce_sum(encode_mask, 1), name="length")
      if self.config.bidirectional_encoder:
        if self.config.encoder_dim % 2:
          raise ValueError(
              "encoder_dim must be even when using a bidirectional encoder.")
        num_units = self.config.encoder_dim // 2
        cell_fw = self._initialize_gru_cell(num_units, trainable = True)  # Forward encoder
        cell_bw = self._initialize_gru_cell(num_units, trainable = True)  # Backward encoder
        _, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=encode_emb,
            sequence_length=length,
            dtype=tf.float32,
            scope=scope)
        thought_vectors = tf.concat(states, 1)
      else:
        cell = self._initialize_gru_cell(self.config.encoder_dim, trainable = True)
        self.h_t, state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=encode_emb,
            sequence_length=length,
            dtype=tf.float32,
            scope=scope)
        thought_vectors = state

    return thought_vectors

  def video_encoder(self, video, video_length, batch_size, h_t = None):


    video_rep = tf.matmul(tf.diag(tf.cast(1/video_length,tf.float32)),tf.reduce_sum(video,1))

    video_rep = tf.identity(video_rep, name="vision_vectors")

    return video_rep


  def _build_decoder(self, name, embeddings, targets, mask, initial_state,
                     reuse_logits, reuse_decoder, setting):

    with tf.variable_scope(name , reuse = reuse_decoder) as scope:
    # Decoder RNN.
      cell = self._initialize_gru_cell(self.config.encoder_dim, trainable = True)

      # Add a padding word at the start of each sentence (to correspond to the
      # prediction of the first word) and remove the last word.
      decoder_input = tf.pad(embeddings[:, :-1, :], [[0, 0], [1, 0], [0, 0]], name="input")
      length = tf.reduce_sum(mask, 1, name="length")
      decoder_output, _ = tf.nn.dynamic_rnn(
          cell=cell,
          inputs=decoder_input,
          sequence_length=length,
          initial_state=initial_state,
          scope=scope)

    # Stack batch vertically.
    decoder_output = tf.reshape(decoder_output, [-1, self.config.encoder_dim])
    targets = tf.reshape(targets, [-1])
    weights = tf.to_float(tf.reshape(mask, [-1]))

    # Logits.
    with tf.variable_scope("logits", reuse=reuse_logits) as scope:
      logits = tf.contrib.layers.fully_connected(
          inputs=decoder_output,
          num_outputs=self.config.vocab_size,
          activation_fn=None,
          weights_initializer=self.uniform_initializer,
          scope=scope,
          trainable = True)

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
    batch_loss = tf.reduce_mean(losses * weights)
    tf.losses.add_loss(batch_loss)

    return batch_loss



  def FS_encoder(self, encode_emb, encode_mask):


    if self.mode == "encode":
      word_dim = self.config.word_encode_dim
    else:
      word_dim = self.config.word_embedding_dim

    encode_emb = tf.reshape(encode_emb, [-1, word_dim])
    weights = tf.to_float(tf.reshape(encode_mask, [-1, 1]))
    encode_emb = encode_emb * weights
    seq_len = tf.shape(encode_mask)[-1]

    encode_emb = tf.reshape(encode_emb, tf.stack([-1, seq_len, word_dim]))
    thought_vectors = tf.reduce_sum(encode_emb,axis=1)
    return thought_vectors


  def _build_FS_decoder(self, targets, mask):

    multiples = tf.stack([1, self.config.len_sentence, 1])

    neg_thought = tf.reshape(tf.tile(tf.expand_dims(self.thought_vectors, 1), multiples),[-1,self.config.encoder_dim])
    negative_word, _, _ = tf.nn.fixed_unigram_candidate_sampler(
        true_classes=tf.ones([self.config.bookcorpus_batch_size*self.config.len_sentence,1],tf.int64),
        num_true=1,
        num_sampled=self.config.text_negative_number * self.config.bookcorpus_batch_size * self.config.len_sentence,
        unique=True,
        range_max=self.config.vocab_size,
        distortion=0.75,
        unigrams=list(get_zipf_mat(self.config.vocab_size)))

    negatives = tf.reshape(negative_word, (self.config.bookcorpus_batch_size*self.config.len_sentence, self.config.text_negative_number))
    negative_context = tf.nn.embedding_lookup(self.word_target,negatives)
    negative_sim = tf.einsum("ij,ikj->ik",neg_thought,negative_context)
    tf.summary.histogram("debug/negative_sim", negative_sim)

    pos_thought = tf.reshape(tf.tile(tf.expand_dims(self.thought_vectors, 1), multiples),[-1,self.config.encoder_dim])
    positive_context = tf.nn.embedding_lookup(self.word_target,targets)
    positive_context = tf.reshape(positive_context,[-1,self.config.encoder_dim])

    positive_sim = tf.matmul(tf.expand_dims(positive_context, 1), tf.expand_dims(pos_thought, 2))[:, 0, 0]
    tf.summary.histogram("debug/positive_sim", positive_sim)

    sim_scores = tf.log(tf.sigmoid(positive_sim)) + tf.reduce_sum(tf.log(tf.sigmoid(-negative_sim)), axis=1)
    tf.summary.histogram("debug/sim_scores", sim_scores)

    weights = tf.to_float(tf.reshape(mask, [-1]))
    batch_loss = - tf.reduce_mean(tf.multiply(sim_scores, weights))
    return batch_loss


  def textual_loss(self):

      if self.config.textual == "SK":
        return self.skip_thought_loss()
      elif (self.config.textual == "FS"):
        return self.fast_sent_loss()


  def kiela_loss(self,s1,s2_word_emb, s2_ids, s2_mask,batch_size, reuse):

    loss = self._build_decoder("decoder_pre", s2_word_emb,
                        s2_ids, s2_mask,
                        s1, reuse_logits = reuse, reuse_decoder = reuse, setting = "skip_thought")

    return  loss


  def skip_thought_loss(self):


    loss_pre = self._build_decoder("decoder_pre", self.decode_pre_emb,
                        self.decode_pre_ids, self.decode_pre_mask,
                        self.thought_vectors, reuse_logits = False, reuse_decoder = False, setting = "skip_thought")
    loss_post = self._build_decoder("decoder_post", self.decode_post_emb,
                        self.decode_post_ids, self.decode_post_mask,
                        self.thought_vectors, reuse_logits = True, reuse_decoder = False, setting = "skip_thought")
    loss = loss_pre + loss_post
    return  loss


  def fast_sent_loss(self):

    loss_pre = self._build_FS_decoder(self.decode_pre_ids, self.decode_pre_mask)
    loss_post = self._build_FS_decoder(self.decode_post_ids, self.decode_post_mask)
    loss_auto = self._build_FS_decoder(self.encode_ids, self.encode_mask)
    loss = loss_pre + loss_post + loss_auto
    return loss

  def ranking_text_loss(self, s1, s2, batch_size, proj = False):

    if proj:
      s1 = self.feed_foward_NN(s1)
      s2 = self.feed_foward_NN(s2)

    return self.max_margin(s1, s2, batch_size)
    """
    V = tf.nn.l2_normalize(tf.cast(s1, tf.float32), 1)
    S = tf.nn.l2_normalize(tf.cast(s2, tf.float32), 1)
    A = tf.matmul(S,tf.transpose(V))
    A = tf.maximum(0.,A - tf.multiply(tf.diag_part(A),tf.ones([batch_size,batch_size],tf.float32)) + self.config.gamma_sentences)
    loss = tf.reduce_mean(A)
    return loss
    """

  def rho_vis(self, vision_vectors, caption_vectors, batch_size, proj = False):
    
    if proj:
      caption_vectors = self.feed_foward_NN(caption_vectors)
      #vision_vectors  = self.feed_foward_NN_image(vision_vectors)
    
    S = tf.nn.l2_normalize(tf.cast(caption_vectors, tf.float32), 1)
    V = tf.nn.l2_normalize(tf.cast(vision_vectors, tf.float32), 1)
    #import ipdb
    #ipdb.set_trace()
    scores_V = tf.reshape(tf.matmul(V,tf.transpose(V)),[-1])
    scores_S = tf.reshape(tf.matmul(S,tf.transpose(S)),[-1])
    scores_V = scores_V - tf.reduce_mean(scores_V)
    scores_S = scores_S - tf.reduce_mean(scores_S)
    std_V = tf.sqrt(tf.reduce_mean(tf.multiply(scores_V,scores_V)))
    std_S = tf.sqrt(tf.reduce_mean(tf.multiply(scores_S,scores_S)))
    if self.config.distance == "rho":
      rho = - tf.reduce_mean(tf.multiply(scores_V,scores_S))/(std_V*std_S)
    elif self.config.distance == "L2":
      rho = tf.reduce_mean(tf.squared_difference(scores_V ,scores_S))
    elif self.config.distance == "L1":
      rho = tf.losses.absolute_difference(scores_V,scores_S)
    return rho


  def max_margin(self, a, b, batch_size, nb_neg=10, margin=0.5):

      a = tf.cast(a, tf.float32)
      b = tf.cast(b, tf.float32)
      Aa = a[:-nb_neg, :]
      Ab = b[:-nb_neg, :]
      Ba = a[-nb_neg:, :]
      Bb = b[-nb_neg:, :]

      normalized_Aa = tf.nn.l2_normalize(Aa,1)
      normalized_Ab = tf.nn.l2_normalize(Ab,1)
      normalized_Ba = tf.nn.l2_normalize(Ba,1)
      normalized_Bb = tf.nn.l2_normalize(Bb,1)
      #normalized_Aa = Aa 
      #normalized_Ab = Ab
      #normalized_Ba = Ba
      #normalized_Bb = Bb

      #normalized_b = F.normalize(Ab)
      pos = tf.reduce_sum(normalized_Aa * normalized_Ab, 1) # batch - nb_neg
      neg1 = tf.transpose(tf.matmul(normalized_Ba, tf.transpose(normalized_Ab))) # (batch - nb_neg) X nb_neg
      neg2 = tf.matmul(normalized_Aa, tf.transpose(normalized_Bb)) 

      M1 = margin - pos[:, None] + neg1 # (batch - nb_neg) X nb_neg
      relu_M1 = tf.maximum(tf.zeros([batch_size-nb_neg,nb_neg],tf.float32), M1)

      M2 = margin - pos[:, None] + neg2 # (batch - nb_neg) X nb_neg
      relu_M2 = tf.maximum(tf.zeros([batch_size-nb_neg,nb_neg],tf.float32), M2)

      return tf.reduce_mean(0.5 * tf.reduce_sum(relu_M1, 1) + 0.5 * tf.reduce_sum(relu_M2, 1))

  def visual_loss(self, vision_vectors, caption_vectors, batch_size):

    #vision_vectors = tf.matmul(vision_vectors,self.W)
    #caption_vectors = tf.matmul(caption_vectors,self.W)
    caption_vectors = self.feed_foward_NN(caption_vectors)
    if self.config.rank_distance == "triplet":
      return self.max_margin(caption_vectors, vision_vectors, batch_size)
      #V = tf.nn.l2_normalize(tf.cast(vision_vectors, tf.float32), 1)
      #S = tf.nn.l2_normalize(tf.cast(caption_vectors, tf.float32), 1)
      #A = tf.matmul(S,tf.transpose(V))
      #A = tf.maximum(0.,A - tf.multiply(tf.diag_part(A),tf.ones([batch_size,batch_size],tf.float32)) + self.config.gamma_sentences)
      #return tf.reduce_mean(A)
    elif self.config.rank_distance == "L2":
      return tf.reduce_mean(tf.squared_difference(caption_vectors, vision_vectors))



  def infer_loss(self, A, B, batch_size, dim, reuse):

    with tf.variable_scope("infer", reuse = reuse) as scope:
      W1_infer = tf.get_variable(name='W1_infer',
                        shape=[dim,500],
                        initializer=tf.random_uniform_initializer(minval=-0.05,maxval=0.05),
                        trainable = True)
      b1_infer = tf.get_variable(name='b1_infer',
                        shape=[500],
                        initializer=tf.random_uniform_initializer(minval=-0.05,maxval=0.05),
                        trainable = True)
      W2_infer = tf.get_variable(name='W2_infer',
                        shape=[500,2],
                        initializer=tf.random_uniform_initializer(minval=-0.05,maxval=0.05),
                        trainable = True)
      b2_infer = tf.get_variable(name='b2_infer',
                        shape=[2],
                        initializer=tf.random_uniform_initializer(minval=-0.05,maxval=0.05),
                        trainable = True)

    b_2 = int(batch_size/2)
    A1 = A[:b_2]
    A2 = A[b_2:]
    B1 = B[:b_2]
    B2 = B[b_2:]
    A1B1 = tf.concat([A1,B1],1)
    A1B2 = tf.concat([A1,B2],1)
    A2B1 = tf.concat([A2,B1],1)
    A2B2 = tf.concat([A2,B2],1)
    X = tf.concat([A1B1,A2B2,A1B2,A2B1],0)
    Y = tf.concat([tf.ones(batch_size),tf.zeros(batch_size)],0)
    pred_X = tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(X, W1_infer), b1_infer)), W2_infer), b2_infer)
    Y = tf.one_hot(tf.cast(Y,tf.int32),2)

    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.expand_dims(Y,1), logits= pred_X))

  def quick_loss(self, A, B, batch_size, dim, reuse):

    """
    with tf.variable_scope("infer", reuse = reuse) as scope:
      W1_infer = tf.get_variable(name='W1_infer',
                        shape=[dim,500],
                        initializer=tf.random_uniform_initializer(minval=-0.05,maxval=0.05),
                        trainable = True)
      b1_infer = tf.get_variable(name='b1_infer',
                        shape=[500],
                        initializer=tf.random_uniform_initializer(minval=-0.05,maxval=0.05),
                        trainable = True)
      W2_infer = tf.get_variable(name='W2_infer',
                        shape=[500,2],
                        initializer=tf.random_uniform_initializer(minval=-0.05,maxval=0.05),
                        trainable = True)
      b2_infer = tf.get_variable(name='b2_infer',
                        shape=[2],
                        initializer=tf.random_uniform_initializer(minval=-0.05,maxval=0.05),
                        trainable = True)
    """

    b_2 = int(batch_size/2)
    B = self.feed_foward_NN(B)
    A1 = A[:b_2]
    A2 = A[b_2:]
    B1 = B[:b_2]
    B2 = B[b_2:]
    A1B1 = tf.expand_dims(tf.reduce_sum(tf.multiply(A1,B1),1),1)
    A1B2 = tf.expand_dims(tf.reduce_sum(tf.multiply(A1,B2),1),1)
    A2B1 = tf.expand_dims(tf.reduce_sum(tf.multiply(A2,B1),1),1)
    A2B2 = tf.expand_dims(tf.reduce_sum(tf.multiply(A2,B2),1),1)
    pred_X = tf.concat([tf.concat([A1B1,A1B2],1),tf.concat([A2B2,A2B1],1),tf.concat([A1B2,A1B1],1),tf.concat([A2B1,A2B2],1)],0)
    Y = tf.concat([tf.ones(batch_size),tf.zeros(batch_size)],0)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.expand_dims(Y,1), logits= pred_X))

    """
    A1A2B1 = tf.concat([A1,A2,B1],1)
    A2A1B2 = tf.concat([A2,A1,B2],1)
    A2A1B1 = tf.concat([A2,A1,B1],1)
    A1A2B2 = tf.concat([A1,A2,B2],1)
    X = tf.concat([A1A2B1, A2A1B2, A2A1B1, A1A2B2],0)
    Y = tf.concat([tf.ones(batch_size),tf.zeros(batch_size)],0)
    pred_X = tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(X, W1_infer), b1_infer)), W2_infer), b2_infer)
    Y = tf.one_hot(tf.cast(Y,tf.int32),2)
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.expand_dims(Y,1), logits= pred_X))
    """

  def build_multi_loss(self):

    multi_loss = 0.

    if self.config.lambda_rho > 0:
      multi_loss += self.config.lambda_rho * self.loss_rho
    if self.config.lambda_rho_f > 0:
      multi_loss += self.config.lambda_rho_f * self.loss_rho_f
    if self.config.lambda_text > 0:
      multi_loss += self.config.lambda_text * self.loss_textual
    if self.config.lambda_ranking > 0:
      multi_loss += self.config.lambda_ranking * self.loss_ranking #+ self.config.mu_reg*tf.norm(self.W2)*tf.norm(self.W2)
    if self.config.lambda_ranking_text > 0:
      multi_loss += self.config.lambda_ranking_text * self.loss_ranking_text
    if self.config.lambda_ranking_text_f > 0:
      multi_loss += self.config.lambda_ranking_text_f * self.loss_ranking_text_f
    if self.config.lambda_cap2cap > 0:
      multi_loss += self.config.lambda_cap2cap * self.loss_cap2cap
    if self.config.lambda_nn > 0:
      multi_loss += self.config.lambda_nn * self.loss_nn
    if self.config.lambda_infer > 0:
      multi_loss += self.config.lambda_infer * self.loss_infer
    if self.config.lambda_Iinfer > 0:
      multi_loss += self.config.lambda_Iinfer * self.loss_Iinfer
    if self.config.lambda_quick > 0:
      multi_loss += self.config.lambda_quick * self.loss_quick
    return multi_loss




  def build_global_step(self):

    with tf.variable_scope(tf.get_variable_scope()):
      self.global_step = tf.train.create_global_step()



  def build(self):

    if self.config.video_embedding in ["VG-A"]:
      caption, visual = self.build_MSVD_inputs(self.reader_visual, self.config.visual_pattern, self.config.visual_batch_size, reuse = False)
      caption_val, visual_val = self.build_MSVD_inputs(self.reader_val, self.config.validation_pattern, self.config.val_batch_size, reuse = True)
    elif self.config.video_embedding in ["R","IF"]:
      caption, visual = self.build_COCO_inputs(self.reader_visual, self.config.visual_pattern, self.config.visual_batch_size, reuse = False)
      caption_val, visual_val = self.build_COCO_inputs(self.reader_val, self.config.validation_pattern, self.config.val_batch_size, reuse = True)
    """
    if self.config.video_embedding == "R":
      visual = visual[:,:self.config.video_encoder_dim]
      visual_val = visual_val[:,:self.config.video_encoder_dim]
    """
    if self.config.video_embedding != "T":
      s1, s2, s2_word_emb, s2_ids, s2_mask = self.build_ranking_text_inputs(self.reader_sim, self.config.sim_pattern, self.config.sim_batch_size, reuse = True)
      s1_val, s2_val, s2_word_emb_val, s2_ids_val, s2_mask_val = self.build_ranking_text_inputs(self.reader_sim_val, self.config.val_sim_pattern, self.config.sim_val_batch_size, reuse = True)

      nn_s1, nn_s2, nn_s2_word_emb, nn_s2_ids, nn_s2_mask = self.build_ranking_text_inputs(self.reader_nn, self.config.nn_pattern, self.config.sim_batch_size, reuse = True)
      nn_s1_val, nn_s2_val, nn_s2_word_emb_val, nn_s2_ids_val, nn_s2_mask_val = self.build_ranking_text_inputs(self.reader_nn_val, self.config.val_nn_pattern, self.config.sim_val_batch_size, reuse = True)

    self.build_bookcorpus_inputs()

    if self.mode != "encode":

      if self.config.lambda_ranking_text > 0:
        self.loss_ranking_text_val = self.ranking_text_loss(s1_val, s2_val, self.config.val_batch_size)
        self.loss_ranking_text = self.ranking_text_loss(s1, s2, self.config.visual_batch_size)
        tf.summary.scalar("losses_val/ranking_text_val", self.loss_ranking_text_val)
        tf.summary.scalar("losses/ranking_text", self.loss_ranking_text)
      if self.config.lambda_ranking_text_f > 0:
        self.loss_ranking_text_val_f = self.ranking_text_loss(s1_val, s2_val, self.config.val_batch_size, proj = True)
        self.loss_ranking_text_f = self.ranking_text_loss(s1, s2, self.config.visual_batch_size, proj = True)
        tf.summary.scalar("losses_val/ranking_text_val", self.loss_ranking_text_val_f)
        tf.summary.scalar("losses/ranking_text", self.loss_ranking_text_f)
      if self.config.lambda_ranking > 0:
        self.loss_ranking_val = self.visual_loss(visual_val, caption_val, self.config.val_batch_size)
        self.loss_ranking = self.visual_loss(visual, caption, self.config.visual_batch_size)
        tf.summary.scalar("losses_val/ranking_val", self.loss_ranking_val)
        tf.summary.scalar("losses/ranking", self.loss_ranking)
      if self.config.lambda_rho > 0:
        self.loss_rho =  self.rho_vis(visual, caption, self.config.visual_batch_size)
        self.loss_rho_val =  self.rho_vis(visual_val, caption_val, self.config.val_batch_size)
        tf.summary.scalar("losses/rho", self.loss_rho)
        tf.summary.scalar("losses_val/rho_val", self.loss_rho_val)
      if self.config.lambda_rho_f > 0:
        self.loss_rho_f =  self.rho_vis(visual, caption, self.config.visual_batch_size, proj = True)
        self.loss_rho_f_val =  self.rho_vis(visual_val, caption_val, self.config.val_batch_size, proj = True)
        tf.summary.scalar("losses/rho_f", self.loss_rho_f)
        tf.summary.scalar("losses_val/rho_f_val", self.loss_rho_f_val)
      if self.config.lambda_text > 0:
        self.loss_textual = self.textual_loss()
        tf.summary.scalar("losses/textual_", self.loss_textual)
      if self.config.lambda_cap2cap > 0:
        self.loss_cap2cap = self.kiela_loss(s1,s2_word_emb, s2_ids, s2_mask,self.config.visual_batch_size, False)
        self.loss_cap2cap_val = self.kiela_loss(s1_val,s2_word_emb_val, s2_ids_val, s2_mask_val,self.config.val_batch_size, True)
        tf.summary.scalar("losses/cap2cap", self.loss_cap2cap)
        tf.summary.scalar("losses_val/cap2cap_val", self.loss_cap2cap_val)
      if self.config.lambda_nn > 0:
        self.loss_nn_val = self.ranking_text_loss(nn_s1_val, nn_s2_val, self.config.val_batch_size)
        self.loss_nn = self.ranking_text_loss(nn_s1, nn_s2, self.config.visual_batch_size)
        tf.summary.scalar("losses_val/nn_val", self.loss_nn_val)
        tf.summary.scalar("losses/nn", self.loss_nn)
      if self.config.lambda_infer > 0:
        self.loss_infer = self.infer_loss(s1, s2, self.config.visual_batch_size, 2*self.config.encoder_dim, False)
        self.loss_infer_val = self.infer_loss(s1_val, s2_val, self.config.val_batch_size, 2*self.config.encoder_dim, True)
        tf.summary.scalar("losses_val/infer_val", self.loss_infer_val)
        tf.summary.scalar("losses/infer", self.loss_infer)
      if self.config.lambda_Iinfer > 0:
        self.loss_Iinfer = self.infer_loss(caption, visual, self.config.visual_batch_size, self.config.video_encoder_dim+self.config.encoder_dim, False)
        self.loss_Iinfer_val = self.infer_loss(caption_val, visual_val, self.config.val_batch_size, self.config.video_encoder_dim+self.config.encoder_dim, True)
        tf.summary.scalar("losses_val/Iinfer_val", self.loss_Iinfer_val)
        tf.summary.scalar("losses/Iinfer", self.loss_Iinfer)
      if self.config.lambda_quick > 0:
        self.loss_quick = self.quick_loss(visual, caption, self.config.visual_batch_size, 2*self.config.video_encoder_dim+self.config.encoder_dim, False)
        self.loss_quick_val = self.quick_loss(visual_val, caption_val,  self.config.val_batch_size, 2*self.config.video_encoder_dim+self.config.encoder_dim, True)
        tf.summary.scalar("losses_val/quick_val", self.loss_quick_val)
        tf.summary.scalar("losses/quick", self.loss_quick)

      self.total_loss  = self.build_multi_loss()

      #rho_val = self.pearson()
      #tf.summary.scalar("eval/SICK_valid" , rho_val)
      #rho_f_val = self.pearson(proj = True)
      #tf.summary.scalar("eval/SICK_f_val" , rho_f_val)

      #self.compute_mNNO_dev(K=10)

      self.build_global_step()
    #self.f_thought_vectors = tf.identity(tf.concat([self.thought_vectors,tf.add(tf.matmul(tf.nn.relu(tf.add(tf.matmul(self.thought_vectors, self.W1), self.b1)), self.W2),self.b2)],1), name = "f_thought_vectors")
    #self.f_thought_vectors = tf.identity(tf.concat([self.thought_vectors,tf.add(tf.matmul(self.thought_vectors, self.W1), self.b1)],1), name = "f_thought_vectors")
    #self.f_thought_vectors = tf.identity(tf.add(tf.matmul(self.thought_vectors, self.W1), self.b1), name = "f_thought_vectors")
    #self.f_thought_vectors = tf.identity(self.thought_vectors, name = "f_thought_vectors")





