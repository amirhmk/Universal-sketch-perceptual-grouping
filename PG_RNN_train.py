
import json
import os
import time
import urllib
import zipfile
import h5py
# internal imports
import tensorflow.contrib.slim as slim
import numpy as np
import requests
import six
from six.moves import cStringIO as StringIO
import tensorflow as tf

import model as sketch_rnn_model
import utils
import pdb
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string(
    'data_dir',
    'SketchX-PRIS-Dataset/Perceptual_Grouping/',

    'The directory in which to find the dataset specified in model hparams. '
    'If data_dir starts with "http://" or "https://", the file will be fetched '
    'remotely.')
tf.app.flags.DEFINE_string(
    'log_root', './models',
    'Directory to store model checkpoints, tensorboard.')
tf.app.flags.DEFINE_boolean(
    'resume_training', False,
    'Set to true to load previous checkpoint')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'Pass in comma-separated key=value pairs such as '
    '\'save_every=40,decay_rate=0.99\' '
    '(no whitespace) to be read into the HParams object defined in model.py')

PRETRAINED_MODELS_URL = ('http://download.magenta.tensorflow.org/models/'
                         'sketch_rnn.zip')


def reset_graph():
  """Closes the current default session and resets the graph."""
  sess = tf.get_default_session()
  if sess:
    sess.close()
  tf.reset_default_graph()


def load_env(data_dir, model_dir):
  """Loads environment for inference mode, used in jupyter notebook."""
  model_params = sketch_rnn_model.get_default_hparams()
  with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
    model_params.parse_json(f.read())
  return load_dataset(data_dir, model_params, inference_mode=True)


def load_model(model_dir):
  """Loads model for inference mode, used in jupyter notebook."""
  model_params = sketch_rnn_model.get_default_hparams()
  with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
    model_params.parse_json(f.read())

  model_params.batch_size = 1  # only sample one at a time
  eval_model_params = sketch_rnn_model.copy_hparams(model_params)
  eval_model_params.use_input_dropout = 0
  eval_model_params.use_recurrent_dropout = 0
  eval_model_params.use_output_dropout = 0
  eval_model_params.is_training = 0
  sample_model_params = sketch_rnn_model.copy_hparams(eval_model_params)
  sample_model_params.max_seq_len = 1  # sample one point at a time
  return [model_params, eval_model_params, sample_model_params]


def download_pretrained_models(
    models_root_dir='./models',
    pretrained_models_url=PRETRAINED_MODELS_URL):
  """Download pretrained models to a temporary directory."""
  tf.gfile.MakeDirs(models_root_dir)
  zip_path = os.path.join(
      models_root_dir, os.path.basename(pretrained_models_url))
  if os.path.isfile(zip_path):
    tf.logging.info('%s already exists, using cached copy', zip_path)
  else:
    tf.logging.info('Downloading pretrained models from %s...',
                    pretrained_models_url)
    urllib.urlretrieve(pretrained_models_url, zip_path)
    tf.logging.info('Download complete.')
  tf.logging.info('Unzipping %s...', zip_path)
  with zipfile.ZipFile(zip_path) as models_zip:
    models_zip.extractall(models_root_dir)
  tf.logging.info('Unzipping complete.')


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
                    z_pen_logits, x1_data, x2_data, pen_data, hps):
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
    if not hps.is_training:  # eval mode, mask eos columns
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


def pre_label_com_loss(hps, sequence_lengths, output, ground_labels, str_labels, batch_triplets):
    batch_size = hps.batch_size
    # loss = 0
    accuracy = 0
    # max_seq_len = 300
    output = tf.reshape(output, [batch_size, hps.max_seq_len, -1])
     # Output shape after:  (100, 300, 128)
    output_shape = output.shape

    soft_pre_labels = []
    with tf.variable_scope('logic'):
        logic_w = tf.get_variable(
            'logic_w', [output_shape[2], 2])  # (128, 2)
        logic_b = tf.get_variable('logic_b', [2])  # (128,)
        # str_labels shape (100, 300, 300)
        # tf.unstack:
        # Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
        for idx, seq_len in enumerate(tf.unstack(sequence_lengths)):
            # str_label is the label state. 0 Means start, 1 means drawing
            c_str_label = tf.cast(tf.squeeze(tf.slice(
                str_labels, begin=[idx, 0, 0], size=[1, seq_len, seq_len])), tf.float32)
            # print("fuck u tf", c_str_label, seq_len)
            reshape_str_label = tf.reshape(c_str_label, [-1])
            # Flatten the stroke label
            # tf.tile: Constructs a tensor by tiling a given tensor. This will copy across columns
            tile_str_label = tf.tile(
                tf.expand_dims(reshape_str_label, 1), [1, 2])
            # tile_str_label shape (?, 2)

            # output_shape[2]: 128
            # Output shape after:  (100, 300, 128)
            # tf.slice:
            # This operation extracts a slice of size size from a tensor input_ starting at the location specified by begin
            # This is the f_vec from t=0 to t=300/seq_len (if lower)
            feats = tf.squeeze(tf.slice(output, begin=[idx, 0, 0], size=[
                               1, seq_len, output_shape[2]]))
            # This is unknown as well..., but I think it would be [1, 300, 128], so just one batch of all sequences
            # c = i
            # r = j
            c_feats = tf.expand_dims(feats, 1)
            r_feats = tf.expand_dims(feats, 0)
            c_tile = tf.tile(c_feats, [1, seq_len, 1])
            r_tile = tf.tile(r_feats, [seq_len, 1, 1])
            delta_feats = tf.abs(c_tile-r_tile)
            # delta_feats += saliency_difference()
            # Feature Difference Matrix D
            reshape_d_f_matrix = tf.reshape(delta_feats, [-1, output_shape[2]])
            # reshape_d_f_matrix shape (?, 128)

            c_ground_labels = tf.cast(tf.squeeze(tf.slice(ground_labels, begin=[idx, 0, 0], size=[1, seq_len, seq_len])), tf.float32)
            reshape_c_g_l = tf.reshape(c_ground_labels, [-1])
            # This basically takes the ground_truth values and assignms them to all the different classes, as there is multi
            reshape_c_g_l = tf.multiply(reshape_c_g_l, reshape_str_label)

            # logic_w shape (128, 2)
            # logic_b shape (2,)
            logic_wxb = tf.nn.xw_plus_b(reshape_d_f_matrix, logic_w, logic_b)
            # 0 out any stroke that is not in "Drawing" mode
            logic_wxb = tf.multiply(logic_wxb, tile_str_label)
            # Local Grouping Loss: sketch_loss
            # Find out how logic_wxb -> G^HAT
            # reshape_c_g_l is flattened
            # Logits: Unscaled log probabilities of shape [d_0, d_1, ...,, num_classes]
            # Labels: Tensor of shape [d_0, d_1, ..., d_{r-1}]. Each entry in labels must be an index in [0, num_classes)
            soft_max_logic = tf.nn.softmax(logic_wxb)
            # sketch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(reshape_c_g_l,dtype=tf.int32), logits=logic_wxb)
            sketch_loss = tf.keras.losses.sparse_categorical_crossentropy(
                tf.cast(reshape_c_g_l, dtype=tf.int32), soft_max_logic)

            # tf.cast(reshape_c_g_l,dtype=tf.int32), soft_max_logic)
            # sketch_loss shape will be the same as labels (Each stroke will have a label)
            # saliency_loss =
            # input seqs,logic_wx output: saliency_loss
            # sketch_loss += saliency_loss

            reshape_soft_max_logic = tf.reshape(
                soft_max_logic, [seq_len, seq_len, 2])
            soft_pre_label = tf.squeeze(tf.slice(reshape_soft_max_logic, begin=[
                                        0, 0, 1], size=[seq_len, seq_len, 1]))
            soft_pre_label = tf.multiply(soft_pre_label, c_str_label)
            pre_label = tf.argmax(soft_max_logic, 1)

            correct_label = tf.equal(
                tf.cast(pre_label, dtype=tf.int32), tf.cast(reshape_c_g_l, dtype=tf.int32))
            #tf.reduce_sum:
            # Flattened sum
            accuracy += tf.div(tf.reduce_sum(tf.cast(correct_label, dtype=tf.float32)),
                               tf.cast(seq_len*seq_len, dtype=tf.float32))

            # This may be G!! group_matrix
            soft_pre_labels.append([soft_pre_label])

            # Triplet Loss
            anc_idx = tf.squeeze(tf.slice(batch_triplets, begin=[
                                 idx, 0, 0], size=[1, 1, 3000]))
            pos_idx = tf.squeeze(tf.slice(batch_triplets, begin=[
                                 idx, 1, 0], size=[1, 1, 3000]))
            neg_idx = tf.squeeze(tf.slice(batch_triplets, begin=[
                                 idx, 2, 0], size=[1, 1, 3000]))
            anc = tf.gather(soft_pre_label, anc_idx)
            pos = tf.gather(soft_pre_label, pos_idx)
            neg = tf.gather(soft_pre_label, neg_idx)
            d_pos = tf.reduce_sum(tf.square(anc - pos), -1)
            d_neg = tf.reduce_sum(tf.square(anc - neg), -1)
            triplet_loss = tf.maximum(0., d_pos - d_neg+2.5)
            #triplet_loss = d_pos - d_neg
            if idx == 0:
                sketch_cost = sketch_loss
                triplet_cost = triplet_loss
            else:
                sketch_cost = tf.concat([sketch_cost, sketch_loss], 0)
                triplet_cost = tf.concat([triplet_cost, triplet_loss], 0)
            # ,tf.cast(test_loss,dtype=tf.float32)
            return sketch_cost, triplet_cost, accuracy/batch_size, soft_pre_labels


def load_dataset(data_dir, model_params, inference_mode=False):
  # aug_data_dir ='/import/vision-datasets/kl303/PG_data/svg_fine_tuning/Aug_data/'


  datasets = model_params.data_set
  model_params.data_set = datasets
  train_strokes = None
  valid_strokes = None
  eval_strokes = None

  for dataset in datasets:

    with open(data_dir+dataset+'.ndjson','r') as f:
      ori_data = json.load(f)
      train_stroke = ori_data['train_data'][:650]
      valid_stroke = ori_data['train_data'][650:700]
      eval_stroke = ori_data['train_data'][700:]

    if train_strokes is None:
      train_strokes = train_stroke
    else:
      train_strokes = np.concatenate((train_strokes, train_stroke))
    if valid_strokes is None:
      valid_strokes = valid_stroke
    else:
      valid_strokes = np.concatenate((valid_strokes, valid_stroke))
    if eval_strokes is None:
      eval_strokes = eval_stroke
    else:
      eval_strokes = np.concatenate((eval_strokes, eval_stroke))


  all_strokes = np.concatenate((train_strokes, valid_strokes, eval_strokes))
  #all_strokes = train_strokes
  num_points = 0
  for stroke in all_strokes:
    num_points += len(stroke)

  # calculate the max strokes we need.
  max_seq_len = utils.get_max_len(all_strokes)
  # overwrite the hps with this calculation.
  model_params.max_seq_len = max_seq_len

  tf.logging.info('model_params.max_seq_len %i.', model_params.max_seq_len)

  eval_model_params = sketch_rnn_model.copy_hparams(model_params)

  eval_model_params.use_input_dropout = 0
  eval_model_params.use_recurrent_dropout = 0
  eval_model_params.use_output_dropout = 0
  eval_model_params.is_training = 1

  if inference_mode:
    eval_model_params.batch_size = 1
    eval_model_params.is_training = 0

  sample_model_params = sketch_rnn_model.copy_hparams(eval_model_params)
  sample_model_params.batch_size = 1  # only sample one at a time
  sample_model_params.max_seq_len = 1  # sample one point at a time

  #pdb.set_trace()
  train_set = utils.DataLoader(
      train_strokes,
      model_params.batch_size,
      max_seq_length=model_params.max_seq_len,
      random_scale_factor=model_params.random_scale_factor,
      augment_stroke_prob=model_params.augment_stroke_prob)

  normalizing_scale_factor = train_set.calculate_normalizing_scale_factor()
  train_set.normalize(normalizing_scale_factor)

  valid_set = utils.DataLoader(
      valid_strokes,
      eval_model_params.batch_size,
      max_seq_length=eval_model_params.max_seq_len,
      random_scale_factor=0.0,
      augment_stroke_prob=0.0)
  valid_set.normalize(normalizing_scale_factor)
  #
  test_set = utils.DataLoader(
      eval_strokes,
      eval_model_params.batch_size,
      max_seq_length=eval_model_params.max_seq_len,
      random_scale_factor=0.0,
      augment_stroke_prob=0.0)
  test_set.normalize(normalizing_scale_factor)

  tf.logging.info('normalizing_scale_factor %4.4f.', normalizing_scale_factor)


  result = [train_set,valid_set,test_set, model_params, eval_model_params,sample_model_params]
  return result


def evaluate_model(sess, model, data_set):
  """Returns the average weighted cost, reconstruction cost and KL cost."""
  total_g_cost=0.0
  test_ac=0.0
  for batch in range(data_set.num_batches):
    unused_orig_x, x,labels,str_labels, s, saliency = data_set.get_batch(batch)
    feed = {
      model.input_data: x,
      model.sequence_lengths: s,
      model.labels:labels,
      model.str_labels:str_labels,
      model.saliency: saliency
    }
    (g_cost,ac) = sess.run([g_loss,model.accuracy], feed)

    total_g_cost += g_cost
    test_ac +=ac

  total_g_cost /= (data_set.num_batches)
  test_ac /= (data_set.num_batches)
  return (total_g_cost,test_ac)


def load_checkpoint(checkpoint_path,checkpoint_exclude_scopes):
  ckpt = tf.train.get_checkpoint_state(checkpoint_path)
  pretrain_model = ckpt.model_checkpoint_path
  print("load pretrained model from %s" % pretrain_model)

  exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

  variables_to_restore = []

  for var in tf.trainable_variables():
    excluded = False
    for exclusion in exclusions:
      if var.op.name.startswith(exclusion):
        excluded = True
        break
    if not excluded:
      print(var.name)
      variables_to_restore.append(var)
  return slim.assign_from_checkpoint_fn(pretrain_model, variables_to_restore)


def save_model(sess, model_save_path, global_step,saver):
  checkpoint_path = os.path.join(model_save_path, 'vector')
  tf.logging.info('saving model %s.', checkpoint_path)
  tf.logging.info('global_step %i.', global_step)
  saver.save(sess, checkpoint_path, global_step=global_step)

def train(sess, model, eval_model, train_set, valid_set, test_set,saver):
  """Train a sketch-rnn model."""
  # Setup summary writer.
  summary_writer = tf.summary.FileWriter(FLAGS.log_root)
  # Calculate trainable params.
  t_vars = tf.trainable_variables()
  count_t_vars = 0
  for var in t_vars:
    num_param = np.prod(var.get_shape().as_list())
    count_t_vars += num_param
    tf.logging.info('%s %s %i', var.name, str(var.get_shape()), num_param)
  tf.logging.info('Total trainable variables %i.', count_t_vars)
  model_summ = tf.summary.Summary()
  model_summ.value.add(
      tag='Num_Trainable_Params', simple_value=float(count_t_vars))
  summary_writer.add_summary(model_summ, 0)
  summary_writer.flush()

  # setup eval stats
  best_valid_cost = 100000  # set a large init value
  valid_cost = 0.0

  # main train loop

  hps = model.hps
  start = time.time()
  
  sketch_loss,triplets_loss, accuracy, out_pre_labels = pre_label_com_loss(hps, model.sequence_lengths,model.feat_out,model.labels, model.str_labels,model.triplets)
  g_loss = tf.reduce_mean(sketch_loss)
  t_cost = tf.reduce_mean(triplets_loss)


  r_weight = tf.Variable(hps.kl_weight, trainable=False)
  lossfunc = get_lossfunc(model.pi, model.mu1, model.mu2, model.sigma1, model.sigma2,
                          model.corr, model.pen_logits, model.x1_data, model.x2_data, model.pen_data, hps)
  r_cost = tf.reduce_mean(lossfunc)
  cost = g_loss+r_cost*r_weight+t_cost*0.6+model.kl_cost*0.02
  optimizer = tf.train.AdamOptimizer()
  train_op = optimizer.minimize(cost)
  sess.run(tf.global_variables_initializer())
  for _ in range(hps.num_steps):

    step = sess.run(model.global_step)

    curr_learning_rate = ((hps.learning_rate - hps.min_learning_rate) *
                          (hps.decay_rate)**(step/3) + hps.min_learning_rate)
    curr_kl_weight = (hps.kl_weight - (hps.kl_weight - hps.kl_weight_start) *
                      (hps.kl_decay_rate)**(step/3))

    _, x,labels,seg_labels, s,triplet_label, saliency = train_set.random_batch()
  
    lr = tf.Variable(hps.learning_rate, trainable=False)


    # train_op = optimizer.minimize(cost)
    gvs = optimizer.compute_gradients(cost)
    g = hps.grad_clip
    for grad, var in gvs:
        tf.clip_by_value(grad, -g, g)
        capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]

    train_op = optimizer.apply_gradients(
        capped_gvs, global_step=model.global_step, name='train_step')
    feed = {
        model.input_data: x,
        model.sequence_lengths: s,
        # model.lr: curr_learning_rate,
        model.labels:labels,
        model.str_labels:seg_labels,
        model.triplets:triplet_label,
        # model.saliency: saliency
    }
        
    (triplet_loss,g_cost,train_accuracy, _, pre_labels,train_step, _) = sess.run([
        t_cost, g_loss, accuracy, model.final_state, out_pre_labels, model.global_step, train_op], feed)
    (triplet_loss,g_cost,train_accuracy, _, pre_labels,train_step, _) = sess.run([
        t_cost, g_loss, accuracy, model.final_state, out_pre_labels, model.global_step, train_op], feed)
    # (_, total_loss, g_cost) = sess.run([train_op, cost, g_cost], feed)
    # print(g_loss)
    # print("triplet_loss", triplet_loss)
    # print("g_cost", g_cost)
    # print("train_accuracy", train_accuracy)
    # print("train_step", train_step)
    if step % 10 == 0 and step > 0:
    #if step % 1 == 0 and step > 0:
      end = time.time()
      time_taken = end - start


      g_summ = tf.summary.Summary()
      g_summ.value.add(tag='Train_group_Cost', simple_value=float(g_cost))
      lr_summ = tf.summary.Summary()
      lr_summ.value.add(
          tag='Learning_Rate', simple_value=float(curr_learning_rate))
      kl_weight_summ = tf.summary.Summary()
      kl_weight_summ.value.add(
          tag='KL_Weight', simple_value=float(curr_kl_weight))
      time_summ = tf.summary.Summary()
      time_summ.value.add(
          tag='Time_Taken_Train', simple_value=float(time_taken))
      accuracy_summ = tf.summary.Summary()
      accuracy_summ.value.add(
      tag='train_accuracy', simple_value=float(train_accuracy))
      output_format = ('step: %d, lr: %.6f, cost: %.4f,'
                       'train_time_taken: %.4f,train_accuracy: %.4f')
      output_values = (step, curr_learning_rate,  g_cost, time_taken,train_accuracy)
      output_log = output_format % output_values

      tf.logging.info(output_log)

      summary_writer.add_summary(g_summ, train_step)
      summary_writer.add_summary(lr_summ, train_step)
      summary_writer.add_summary(kl_weight_summ, train_step)
      summary_writer.add_summary(time_summ, train_step)
      summary_writer.flush()
      start = time.time()

    if step % hps.save_every == 0 and step > 0:
      (valid_g_cost,valid_ac) = evaluate_model(sess, eval_model, valid_set)
      valid_cost=valid_g_cost
      end = time.time()
      time_taken_valid = end - start
      start = time.time()

      valid_g_summ = tf.summary.Summary()
      valid_g_summ.value.add(
          tag='Valid_group_Cost', simple_value=float(valid_g_cost))
      valid_time_summ = tf.summary.Summary()
      valid_time_summ.value.add(
          tag='Time_Taken_Valid', simple_value=float(time_taken_valid))

      output_format = ('best_valid_cost: %0.4f, valid_g_cost: %.4f, valid_time_taken: %.4f,valid_ac: %.4f')
      output_values = (min(best_valid_cost, valid_g_cost), valid_g_cost, time_taken_valid,valid_ac)
      output_log = output_format % output_values

      tf.logging.info(output_log)

      summary_writer.add_summary(valid_g_summ, train_step)
      summary_writer.add_summary(valid_time_summ, train_step)
      summary_writer.flush()

      if valid_cost < best_valid_cost:
        best_valid_cost = valid_cost

        save_model(sess, FLAGS.log_root, step,saver)

        end = time.time()
        time_taken_save = end - start
        start = time.time()

        tf.logging.info('time_taken_save %4.4f.', time_taken_save)

        best_valid_cost_summ = tf.summary.Summary()
        best_valid_cost_summ.value.add(
            tag='Best_Valid_Cost', simple_value=float(best_valid_cost))

        summary_writer.add_summary(best_valid_cost_summ, train_step)
        summary_writer.flush()

        (eval_g_cost,eval_ac) = evaluate_model(sess, eval_model, test_set)

        end = time.time()
        time_taken_eval = end - start
        start = time.time()

        eval_g_summ = tf.summary.Summary()
        eval_g_summ.value.add(
            tag='Eval_group_Cost', simple_value=float(eval_g_cost))
        eval_accuracy_summ = tf.summary.Summary()
        eval_accuracy_summ.value.add(
          tag='eval_accuracy', simple_value=float(eval_ac))
        eval_time_summ = tf.summary.Summary()
        eval_time_summ.value.add(
            tag='Time_Taken_Eval', simple_value=float(time_taken_eval))

        output_format = ('eval_g_cost: %.4f, '
                         'eval_time_taken: %.4f,eval_accuracy: %.4f')
        output_values = (eval_g_cost, time_taken_eval,eval_ac)
        output_log = output_format % output_values

        tf.logging.info(output_log)

        summary_writer.add_summary(eval_g_summ, train_step)
        summary_writer.add_summary(eval_time_summ, train_step)
        summary_writer.flush()


def trainer(model_params,sess):
  """Train a sketch-rnn model."""
  np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

  tf.logging.info('PG-rnn')
  tf.logging.info('Hyperparams:')
  for key, val in six.iteritems(model_params.values()):
    tf.logging.info('%s = %s', key, str(val))
  tf.logging.info('Loading data files.')
  datasets = load_dataset(FLAGS.data_dir, model_params)

  train_set = datasets[0]
  valid_set = datasets[1]
  test_set = datasets[2]
  model_params = datasets[3]
  eval_model_params = datasets[4]

  model = sketch_rnn_model.Model(model_params)
  eval_model = sketch_rnn_model.Model(eval_model_params, reuse=True)
  
  saver = tf.train.Saver(tf.global_variables())
  if FLAGS.resume_training:
    init_op = load_checkpoint(FLAGS.log_root,[])
    init_op(sess)

  # Write config file to json file.
  tf.gfile.MakeDirs(FLAGS.log_root)
  with tf.gfile.Open(
      os.path.join(FLAGS.log_root, 'model_config.json'), 'w') as f:
    json.dump(model_params.values(), f, indent=True)

  train(sess, model, eval_model, train_set,valid_set,test_set,saver)


def main(unused_argv):
  """Load model params, save config file and start trainer."""
  sess = tf.Session()
  default_model_params = sketch_rnn_model.get_default_hparams()
  trainer(default_model_params,sess)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
