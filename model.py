# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sketch-RNN Model."""

import random

# internal imports

import numpy as np
import tensorflow as tf

import rnn


def copy_hparams(hparams):
  """Return a copy of an HParams instance."""
  return tf.contrib.training.HParams(**hparams.values())
  

def get_default_hparams():
  """Return default HParams for sketch-rnn."""
  hparams = tf.contrib.training.HParams(
      data_set=['airplane','alarm_clock','ambulance','ant','apple','backpack','basket','butterfly','cactus',
               'campfire','candle','coffee_cup','crab','duck','face','ice_cream','pig','pineapple','suitcase','calculator'], # SketchX Dataset.
      #teat_data_set = ['airplane','alarm-clock','ambulance','ant','apple'], # 1
      #teat_data_set = ['backpack','basket','butterfly','cactus','campfire'], #2
      # teat_data_set = ['candle','coffee_cup','crab','duck','face'], #3
      # teat_data_set = ['ice-cream','pig','pineapple','suitcase','calculator'], #4

      num_steps=100000,  # Total number of steps of training. Keep large.
      save_every=50,  # Number of batches per checkpoint creation.
      max_seq_len=300,  # Not used. Will be changed by model. [Eliminate?]
      dec_rnn_size=512,  # Size of decoder.
      dec_model='lstm',  # Decoder: lstm, layer_norm or hyper.
      enc_rnn_size=256,  # Size of encoder.
      enc_model='lstm',  # Encoder: lstm, layer_norm or hyper.
      z_size=128,  # Size of latent vector z. Recommend 32, 64 or 128.
      kl_weight=0.5,  # KL weight of loss equation. Recommend 0.5 or 1.0.
      kl_weight_start=0.01,  # KL start weight when annealing.
      kl_tolerance=0.2,  # Level of KL loss at which to stop optimizing for KL.
      batch_size=100,  # Minibatch size. Recommend leaving at 100.
      grad_clip=1.0,  # Gradient clipping. Recommend leaving at 1.0.
      num_mixture=20,  # Number of mixtures in Gaussian mixture model.
      learning_rate=0.0001,  # Learning rate.
      decay_rate=0.9995,  # Learning rate decay per minibatch.
      kl_decay_rate=0.99995,  # KL annealing decay rate per minibatch.
      min_learning_rate=0.00001,  # Minimum learning rate.
      use_recurrent_dropout=True,  # Dropout with memory loss. Recomended
      recurrent_dropout_prob=0.90,  # Probability of recurrent dropout keep.
      use_input_dropout=False,  # Input dropout. Recommend leaving False.
      input_dropout_prob=0.90,  # Probability of input dropout keep.
      use_output_dropout=False,  # Output droput. Recommend leaving False.
      output_dropout_prob=0.90,  # Probability of output dropout keep.
      random_scale_factor=0.15,  # Random scaling data augmention proportion.
      augment_stroke_prob=0.10,  # Point dropping augmentation proportion.
      conditional=True,  # When False, use unconditional decoder-only model.
      is_training=True  # Is model training? Recommend keeping true.
  )
  return hparams


class Model(object):
  """Define a SketchRNN model."""

  def __init__(self, hps, gpu_mode=True, reuse=False):

    self.hps = hps
    with tf.variable_scope('vector_rnn', reuse=reuse):
      if not gpu_mode:
        with tf.device('/cpu:0'):
          tf.logging.info('Model using cpu.')
          self.build_model(hps)
      else:
        tf.logging.info('Model using gpu.')
        self.build_model(hps)

  def encoder(self, batch, sequence_lengths):
    """Define the bi-directional encoder module of sketch-rnn."""
    unused_outputs, last_states = tf.nn.bidirectional_dynamic_rnn(
        self.enc_cell_fw,
        self.enc_cell_bw,
        batch,
        sequence_length=sequence_lengths,
        time_major=False,
        swap_memory=True,
        dtype=tf.float32,
        scope='ENC_RNN')

    last_state_fw, last_state_bw = last_states
    last_h_fw = self.enc_cell_fw.get_output(last_state_fw)
    last_h_bw = self.enc_cell_bw.get_output(last_state_bw)
    last_h = tf.concat([last_h_fw, last_h_bw], 1)
    mu = rnn.super_linear(
        last_h,
        self.hps.z_size,
        input_size=self.hps.enc_rnn_size * 2,  # bi-dir, so x2
        scope='ENC_RNN_mu',
        init_w='gaussian',
        weight_start=0.001)
    presig = rnn.super_linear(
        last_h,
        self.hps.z_size,
        input_size=self.hps.enc_rnn_size * 2,  # bi-dir, so x2
        scope='ENC_RNN_sigma',
        init_w='gaussian',
        weight_start=0.001)
    return mu, presig

  def build_model(self, hps):
    """Define model architecture."""
    if hps.is_training:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    if hps.dec_model == 'lstm':
      cell_fn = rnn.LSTMCell
    elif hps.dec_model == 'layer_norm':
      cell_fn = rnn.LayerNormLSTMCell
    elif hps.dec_model == 'hyper':
      cell_fn = rnn.HyperLSTMCell
    else:
      assert False, 'please choose a respectable cell'

    if hps.enc_model == 'lstm':
      enc_cell_fn = rnn.LSTMCell
    elif hps.enc_model == 'layer_norm':
      enc_cell_fn = rnn.LayerNormLSTMCell
    elif hps.enc_model == 'hyper':
      enc_cell_fn = rnn.HyperLSTMCell
    else:
      assert False, 'please choose a respectable cell'

    use_recurrent_dropout = self.hps.use_recurrent_dropout
    use_input_dropout = self.hps.use_input_dropout
    use_output_dropout = self.hps.use_output_dropout

    cell = cell_fn(
        hps.dec_rnn_size,
        use_recurrent_dropout=use_recurrent_dropout,
        dropout_keep_prob=self.hps.recurrent_dropout_prob)

    if hps.conditional:  # vae mode:
      if hps.enc_model == 'hyper':
        self.enc_cell_fw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
        self.enc_cell_bw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
      else:
        self.enc_cell_fw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
        self.enc_cell_bw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)

    # dropout:
    tf.logging.info('Input dropout mode = %s.', use_input_dropout)
    tf.logging.info('Output dropout mode = %s.', use_output_dropout)
    tf.logging.info('Recurrent dropout mode = %s.', use_recurrent_dropout)
    if use_input_dropout:
      tf.logging.info('Dropout to input w/ keep_prob = %4.4f.',
                      self.hps.input_dropout_prob)
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, input_keep_prob=self.hps.input_dropout_prob)
    if use_output_dropout:
      tf.logging.info('Dropout to output w/ keep_prob = %4.4f.',
                      self.hps.output_dropout_prob)
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, output_keep_prob=self.hps.output_dropout_prob)
    self.cell = cell

    self.sequence_lengths = tf.placeholder(
        dtype=tf.int32, shape=[self.hps.batch_size])
    self.input_data = tf.placeholder(
        dtype=tf.float32,
        shape=[self.hps.batch_size, self.hps.max_seq_len + 1, 5])
    self.labels = tf.placeholder(
        dtype=tf.int32,
        shape=[self.hps.batch_size,self.hps.max_seq_len, self.hps.max_seq_len])
    self.str_labels = tf.placeholder(
        dtype=tf.int32,
        shape=[self.hps.batch_size,self.hps.max_seq_len,self.hps.max_seq_len])
    # self.saliency = tf.placeholder(
    #     dtype=tf.float32,
    #     shape=[self.hps.batch_size, self.hps.max_seq_len, self.hps.max_seq_len])
    self.triplets = tf.placeholder(
        dtype=tf.int32,
        shape=[self.hps.batch_size,3,3000]
    )
    # The target/expected vectors of strokes
    self.output_x = self.input_data[:, 1:self.hps.max_seq_len + 1, :]
    self.input_x = self.input_data[:, 1:self.hps.max_seq_len + 1, :]

    # either do vae-bit and get z, or do unconditional, decoder-only
    if hps.conditional:  # vae mode:
      self.mean, self.presig = self.encoder(self.output_x,
                                            self.sequence_lengths)
      self.sigma = tf.exp(self.presig / 2.0)  # sigma > 0. div 2.0 -> sqrt.
      
      eps = tf.random_normal(
          (self.hps.batch_size, self.hps.z_size), 0.0, 1.0, dtype=tf.float32)

      if hps.is_training:
        self.batch_z = self.mean + tf.multiply(self.sigma, eps)
      else:
        self.batch_z = self.mean + self.sigma
      # KL cost
      self.kl_cost = -0.5 * tf.reduce_mean(
          (1 + self.presig - tf.square(self.mean) - tf.exp(self.presig)))
      self.kl_cost = tf.maximum(self.kl_cost, self.hps.kl_tolerance)
      pre_tile_y = tf.reshape(self.batch_z,
                              [self.hps.batch_size, 1, self.hps.z_size])
      overlay_x = tf.tile(pre_tile_y, [1, self.hps.max_seq_len, 1])
      actual_input_x = tf.concat([self.input_x, overlay_x], 2)
      self.initial_state = tf.nn.tanh(
          rnn.super_linear(
              self.batch_z,
              cell.state_size,
              init_w='gaussian',
              weight_start=0.001,
              input_size=self.hps.z_size))
    else:  # unconditional, decoder-only generation
      self.batch_z = tf.zeros(
          (self.hps.batch_size, self.hps.z_size), dtype=tf.float32)
      self.kl_cost = tf.zeros([], dtype=tf.float32)
      actual_input_x = self.input_x
      self.initial_state = cell.zero_state(
          batch_size=hps.batch_size, dtype=tf.float32)

    self.num_mixture = hps.num_mixture

    n_out = (3 + self.num_mixture * 6)
    feat_out_size = 128


    with tf.variable_scope('Feat'):
      feat_w = tf.get_variable('feat_w', [self.hps.dec_rnn_size, feat_out_size])
      feat_b = tf.get_variable('feat_b', [feat_out_size])

    with tf.variable_scope('RNN'):
      output_w = tf.get_variable('output_w', [self.hps.dec_rnn_size, n_out])
      output_b = tf.get_variable('output_b', [n_out])

    # decoder module of sketch-rnn is below
    output, last_state = tf.nn.dynamic_rnn(
        cell,
        actual_input_x,
        sequence_length=self.sequence_lengths,
        initial_state=self.initial_state,
        time_major=False,
        swap_memory=True,
        dtype=tf.float32,
        scope='RNN')

    output_reshape = tf.reshape(output, [-1, hps.dec_rnn_size])

    output = tf.nn.xw_plus_b(output_reshape, output_w, output_b)
    feat_out = tf.nn.xw_plus_b(output_reshape, feat_w, feat_b)
    self.final_state = last_state

    # NB: the below are inner functions, not methods of Model
    def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
        """Returns result of eq # 24 of http://arxiv.org/abs/1308.0850."""
        norm1 = tf.subtract(x1, mu1)
        norm2 = tf.subtract(x2, mu2)
        s1s2 = tf.multiply(s1, s2)
        # eq 25
        z = (tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) -
             2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
        neg_rho = 1 - tf.square(rho)
        result = tf.exp(tf.div(-z, 2 * neg_rho))
        denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(neg_rho))
        result = tf.div(result, denom)
        return result

    def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr,
                     z_pen_logits, x1_data, x2_data, pen_data):
        """Returns a loss fn based on eq #26 of http://arxiv.org/abs/1308.0850."""
        # This represents the L_R only (i.e. does not include the KL loss term).

        result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2,
                               z_corr)
        epsilon = 1e-6
        # result1 is the loss wrt pen offset (L_s in equation 9 of
        # https://arxiv.org/pdf/1704.03477.pdf)
        result1 = tf.multiply(result0, z_pi)
        result1 = tf.reduce_sum(result1, 1, keep_dims=True)
        result1 = -tf.log(result1 + epsilon)  # avoid log(0)

        fs = 1.0 - pen_data[:, 2]  # use training data for this
        fs = tf.reshape(fs, [-1, 1])
        # Zero out loss terms beyond N_s, the last actual stroke
        result1 = tf.multiply(result1, fs)

        # result2: loss wrt pen state, (L_p in equation 9)
        result2 = tf.nn.softmax_cross_entropy_with_logits(
            labels=pen_data, logits=z_pen_logits)
        result2 = tf.reshape(result2, [-1, 1])
        if not self.hps.is_training:  # eval mode, mask eos columns
            result2 = tf.multiply(result2, fs)

        result = result1 + result2
        return result

    # below is where we need to do MDN (Mixture Density Network) splitting of
    # distribution params
    def get_mixture_coef(output):
        """Returns the tf slices containing mdn dist params."""
        # This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850.
        z = output
        z_pen_logits = z[:, 0:3]  # pen states
        z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(z[:, 3:], 6, 1)

        # process output z's into MDN paramters

        # softmax all the pi's and pen states:
        z_pi = tf.nn.softmax(z_pi)
        z_pen = tf.nn.softmax(z_pen_logits)

        # exponentiate the sigmas and also make corr between -1 and 1.
        z_sigma1 = tf.exp(z_sigma1)
        z_sigma2 = tf.exp(z_sigma2)
        z_corr = tf.tanh(z_corr)

        r = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, z_pen_logits]
        return r

    # output: pred  
    def pre_label_com_loss(hps,sequence_lengths,output,ground_labels,str_labels,batch_triplets):
        batch_size = hps.batch_size
        # loss = 0
        accuracy =0
        # max_seq_len = 300
        output = tf.reshape(output,[batch_size,hps.max_seq_len,-1])
        # Output shape after:  (100, 300, 128)
        output_shape = output.shape

        soft_pre_labels =[]
        with tf.variable_scope('logic'):
            logic_w = tf.get_variable('logic_w', [output_shape[2], 2]) # (128, 2)
            logic_b = tf.get_variable('logic_b', [2]) # (128,)
        # str_labels shape (100, 300, 300)
        # tf.unstack:
        # Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
        for idx,seq_len in enumerate(tf.unstack(sequence_lengths)):
            # str_label is the label state. 0 Means start, 1 means drawing
            c_str_label = tf.cast(tf.squeeze(tf.slice(str_labels, begin=[idx, 0, 0], size=[1, seq_len, seq_len])), tf.float32)
            reshape_str_label = tf.reshape(c_str_label,[-1])
            # Flatten the stroke label
            # tf.tile: Constructs a tensor by tiling a given tensor. This will copy across columns
            tile_str_label = tf.tile(tf.expand_dims(reshape_str_label,1),[1,2])
            # tile_str_label shape (?, 2)

            # output_shape[2]: 128
            # Output shape after:  (100, 300, 128)
            # tf.slice:
            # This operation extracts a slice of size size from a tensor input_ starting at the location specified by begin
            # This is the f_vec from t=0 to t=300/seq_len (if lower)
            feats = tf.squeeze(tf.slice(output, begin=[idx, 0, 0], size=[1, seq_len, output_shape[2]]))
            # This is unknown as well..., but I think it would be [1, 300, 128], so just one batch of all sequences
            # c = i
            # r = j
            c_feats = tf.expand_dims(feats,1)
            r_feats = tf.expand_dims(feats,0)
            c_tile = tf.tile(c_feats,[1,seq_len,1])
            r_tile = tf.tile(r_feats,[seq_len,1,1])
            delta_feats = tf.abs(c_tile-r_tile)
            # delta_feats += saliency_difference()
            # Feature Difference Matrix D
            reshape_d_f_matrix = tf.reshape(delta_feats,[-1,output_shape[2]])
            # reshape_d_f_matrix shape (?, 128)

            c_ground_labels = tf.cast(tf.squeeze(tf.slice(ground_labels,begin=[idx,0,0],size=[1,seq_len,seq_len])),tf.float32)
            reshape_c_g_l = tf.reshape(c_ground_labels,[-1])
            # This basically takes the ground_truth values and assignms them to all the different classes, as there is multi
            reshape_c_g_l = tf.multiply(reshape_c_g_l,reshape_str_label)

            # logic_w shape (128, 2)
            # logic_b shape (2,)
            logic_wxb = tf.nn.xw_plus_b(reshape_d_f_matrix, logic_w, logic_b)
            # 0 out any stroke that is not in "Drawing" mode
            logic_wxb = tf.multiply(logic_wxb,tile_str_label)
            # Local Grouping Loss: sketch_loss
            # Find out how logic_wxb -> G^HAT
            # reshape_c_g_l is flattened
            # Logits: Unscaled log probabilities of shape [d_0, d_1, ...,, num_classes]
            # Labels: Tensor of shape [d_0, d_1, ..., d_{r-1}]. Each entry in labels must be an index in [0, num_classes)
            sketch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(reshape_c_g_l,dtype=tf.int32), logits=logic_wxb)
            # sketch_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.cast(reshape_c_g_l, dtype=tf.int32), soft_max_logic)
            soft_max_logic = tf.nn.softmax(logic_wxb)

            # tf.cast(reshape_c_g_l,dtype=tf.int32), soft_max_logic)
            # sketch_loss shape will be the same as labels (Each stroke will have a label)
            # saliency_loss = 
            # input seqs,logic_wx output: saliency_loss
            # sketch_loss += saliency_loss

            reshape_soft_max_logic = tf.reshape(soft_max_logic, [seq_len, seq_len, 2])
            soft_pre_label = tf.squeeze(tf.slice(reshape_soft_max_logic, begin=[0, 0, 1], size=[seq_len, seq_len, 1]))
            soft_pre_label =tf.multiply(soft_pre_label,c_str_label)
            pre_label = tf.argmax(soft_max_logic,1)

            correct_label = tf.equal(tf.cast(pre_label,dtype=tf.int32),tf.cast(reshape_c_g_l,dtype=tf.int32))
            #tf.reduce_sum:
            # Flattened sum
            accuracy += tf.div(tf.reduce_sum(tf.cast(correct_label,dtype=tf.float32)),tf.cast(seq_len*seq_len,dtype=tf.float32))

            # This may be G!! group_matrix
            soft_pre_labels.append([soft_pre_label])

            # Triplet Loss
            anc_idx = tf.squeeze(tf.slice(batch_triplets,begin=[idx,0,0],size=[1,1,3000]))
            pos_idx = tf.squeeze(tf.slice(batch_triplets, begin=[idx, 1, 0], size=[1, 1, 3000]))
            neg_idx = tf.squeeze(tf.slice(batch_triplets, begin=[idx, 2, 0], size=[1, 1, 3000]))
            anc = tf.gather(soft_pre_label,anc_idx)
            pos = tf.gather(soft_pre_label,pos_idx)
            neg = tf.gather(soft_pre_label,neg_idx)
            d_pos = tf.reduce_sum(tf.square(anc - pos), -1)
            d_neg = tf.reduce_sum(tf.square(anc - neg), -1)
            triplet_loss = tf.maximum(0., d_pos - d_neg+2.5)
            #triplet_loss = d_pos - d_neg
            if idx==0:
                sketch_cost = sketch_loss
                triplet_cost = triplet_loss
            else:
                sketch_cost = tf.concat([sketch_cost,sketch_loss],0)
                triplet_cost = tf.concat([triplet_cost,triplet_loss],0)
        return sketch_cost,triplet_cost,accuracy/batch_size,soft_pre_labels#,tf.cast(test_loss,dtype=tf.float32)

    out = get_mixture_coef(output)
    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits] = out

    self.pi = o_pi
    self.mu1 = o_mu1
    self.mu2 = o_mu2
    self.sigma1 = o_sigma1
    self.sigma2 = o_sigma2
    self.corr = o_corr
    self.pen_logits = o_pen_logits
    # pen state probabilities (result of applying softmax to self.pen_logits)
    self.pen = o_pen

    # reshape target data so that it is compatible with prediction shape
    target = tf.reshape(self.output_x, [-1, 5])
    [x1_data, x2_data, eos_data, eoc_data, cont_data] = tf.split(target, 5, 1)
    pen_data = tf.concat([eos_data, eoc_data, cont_data], 1)

    # Reconstrunction loss
    lossfunc = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr,
                            o_pen_logits, x1_data, x2_data, pen_data)

    self.r_cost = tf.reduce_mean(lossfunc)

    self.sketch_loss,self.triplets_loss, self.accuracy, self.out_pre_labels = pre_label_com_loss(hps, self.sequence_lengths,feat_out,self.labels, self.str_labels,self.triplets)

    # Grouping loss
    self.g_cost = tf.reduce_mean(self.sketch_loss)
    # Triplet Loss/Global Grouping
    self.t_cost = tf.reduce_mean(self.triplets_loss)
    self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
    optimizer = tf.train.AdamOptimizer(self.lr)

    self.r_weight = tf.Variable(self.hps.kl_weight, trainable=False)
    # Total Loss
    self.cost = self.g_cost+self.r_cost*self.r_weight+self.t_cost*0.6+self.kl_cost*0.02

    # Grad Clip
    gvs = optimizer.compute_gradients(self.cost)
    g = self.hps.grad_clip
    for grad,var in gvs:
      tf.clip_by_value(grad,-g,g)
      capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
    self.train_op = optimizer.apply_gradients(
          capped_gvs, global_step=self.global_step, name='train_step')
