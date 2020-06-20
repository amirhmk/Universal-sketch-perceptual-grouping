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
"""SketchRNN data loading and image manipulation utilities."""

import random
import numpy as np


def get_bounds(data, factor=10):
  """Return bounds of data."""
  min_x = 0
  max_x = 0
  min_y = 0
  max_y = 0

  abs_x = 0
  abs_y = 0
  for i in range(len(data)):
    x = float(data[i, 0]) / factor
    y = float(data[i, 1]) / factor
    abs_x += x
    abs_y += y
    min_x = min(min_x, abs_x)
    min_y = min(min_y, abs_y)
    max_x = max(max_x, abs_x)
    max_y = max(max_y, abs_y)

  return (min_x, max_x, min_y, max_y)


def slerp(p0, p1, t):
  """Spherical interpolation."""
  omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
  so = np.sin(omega)
  return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1


def lerp(p0, p1, t):
  """Linear interpolation."""
  return (1.0 - t) * p0 + t * p1


# A note on formats:
# Sketches are encoded as a sequence of strokes. stroke-3 and stroke-5 are
# different stroke encodings.
#   stroke-3 uses 3-tuples, consisting of x-offset, y-offset, and a binary
#       variable which is 1 if the pen is lifted between this position and
#       the next, and 0 otherwise.
#   stroke-5 consists of x-offset, y-offset, and p_1, p_2, p_3, a binary
#   one-hot vector of 3 possible pen states: pen down, pen up, end of sketch.
#   See section 3.1 of https://arxiv.org/abs/1704.03477 for more detail.
# Sketch-RNN takes input in stroke-5 format, with sketches padded to a common
# maximum length and prefixed by the special start token [0, 0, 1, 0, 0]
# The QuickDraw dataset is stored using stroke-3.
def strokes_to_lines(strokes):
  """Convert stroke-3 format to polyline format."""
  x = 0
  y = 0
  lines = []
  line = []
  for i in range(len(strokes)):
    if strokes[i, 2] == 1:
      x += float(strokes[i, 0])
      y += float(strokes[i, 1])
      line.append([x, y])
      lines.append(line)
      line = []
    else:
      x += float(strokes[i, 0])
      y += float(strokes[i, 1])
      line.append([x, y])
  return lines


def lines_to_strokes(lines):
  """Convert polyline format to stroke-3 format."""
  eos = 0
  strokes = [[0, 0, 0]]
  for line in lines:
    linelen = len(line)
    for i in range(linelen):
      eos = 0 if i < linelen - 1 else 1
      strokes.append([line[i][0], line[i][1], eos])
  strokes = np.array(strokes)
  strokes[1:, 0:2] -= strokes[:-1, 0:2]
  return strokes[1:, :]


def augment_strokes(strokes, prob=0.0):
  """Perform data augmentation by randomly dropping out strokes."""
  # drop each point within a line segments with a probability of prob
  # note that the logic in the loop prevents points at the ends to be dropped.
  result = []
  prev_stroke = [0, 0, 1]
  count = 0
  stroke = [0, 0, 1]  # Added to be safe.
  for i in range(len(strokes)):
    candidate = [strokes[i][0], strokes[i][1], strokes[i][2]]
    if candidate[2] == 1 or prev_stroke[2] == 1:
      count = 0
    else:
      count += 1
    urnd = np.random.rand()  # uniform random variable
    if candidate[2] == 0 and prev_stroke[2] == 0 and count > 2 and urnd < prob:
      stroke[0] += candidate[0]
      stroke[1] += candidate[1]
    else:
      stroke = candidate
      prev_stroke = stroke
      result.append(stroke)
  return np.array(result)


def scale_bound(stroke, average_dimension=10.0):
  """Scale an entire image to be less than a certain size."""
  # stroke is a numpy array of [dx, dy, pstate], average_dimension is a float.
  # modifies stroke directly.
  bounds = get_bounds(stroke, 1)
  max_dimension = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
  stroke[:, 0:2] /= (max_dimension / average_dimension)


def to_normal_strokes(big_stroke):
  """Convert from stroke-5 format (from sketch-rnn paper) back to stroke-3."""
  l = 0
  for i in range(len(big_stroke)):
    if big_stroke[i, 4] > 0:
      l = i
      break
  if l == 0:
    l = len(big_stroke)
  result = np.zeros((l, 3))
  result[:, 0:2] = big_stroke[0:l, 0:2]
  result[:, 2] = big_stroke[0:l, 3]
  return result


def clean_strokes(sample_strokes, factor=100):
  """Cut irrelevant end points, scale to pixel space and store as integer."""
  # Useful function for exporting data to .json format.
  copy_stroke = []
  added_final = False
  for j in range(len(sample_strokes)):
    finish_flag = int(sample_strokes[j][4])
    if finish_flag == 0:
      copy_stroke.append([
          int(round(sample_strokes[j][0] * factor)),
          int(round(sample_strokes[j][1] * factor)),
          int(sample_strokes[j][2]),
          int(sample_strokes[j][3]), finish_flag
      ])
    else:
      copy_stroke.append([0, 0, 0, 0, 1])
      added_final = True
      break
  if not added_final:
    copy_stroke.append([0, 0, 0, 0, 1])
  return copy_stroke


def to_big_strokes(stroke, max_len=250):
  """Converts from stroke-3 to stroke-5 format and pads to given length."""
  # (But does not insert special start token).

  result = np.zeros((max_len, 5), dtype=float)
  l = len(stroke)
  assert l <= max_len
  result[0:l, 0:2] = stroke[:, 0:2]
  result[0:l, 3] = stroke[:, 2]
  result[0:l, 2] = 1 - result[0:l, 3]
  result[l:, 4] = 1
  return result


def get_max_len(strokes):
  """Return the maximum length of an array of strokes."""
  max_len = 0
  for stroke in strokes:
    ml = len(stroke)
    if ml > max_len:
      max_len = ml
  return max_len


class DataLoader(object):
  """Class for loading data."""

  def __init__(self,
               strokes,
               batch_size=128,
               max_seq_length=250,
               scale_factor=1.0,
               random_scale_factor=0.0,
               augment_stroke_prob=0.0,
               limit=1000):
    self.batch_size = batch_size  # minibatch size
    self.max_seq_length = max_seq_length  # N_max in sketch-rnn paper
    self.scale_factor = scale_factor  # divide offsets by this factor
    self.random_scale_factor = random_scale_factor  # data augmentation method
    # Removes large gaps in the data. x and y offsets are clamped to have
    # absolute value no greater than this limit.
    self.limit = limit
    self.augment_stroke_prob = augment_stroke_prob  # data augmentation method
    self.start_stroke_token = [0, 0, 1, 0, 0]  # S_0 in sketch-rnn paper
    # sets self.strokes (list of ndarrays, one per sketch, in stroke-3 format,
    # sorted by size)
    self.preprocess(strokes)

  def preprocess(self, strokes):
    """Remove entries from strokes having > max_seq_length points."""
    raw_data = []
    seq_len = []
    count_data = 0

    for i in range(len(strokes)):
      data = strokes[i]
      if len(data) <= (self.max_seq_length):
        count_data += 1
        # removes large gaps from the data
        data = np.minimum(data, self.limit)
        data = np.maximum(data, -self.limit)
        data = np.array(data, dtype=np.float32)
        data[:, 0:2] /= self.scale_factor
        raw_data.append(data)
        seq_len.append(len(data))
    seq_len = np.array(seq_len)  # nstrokes for each sketch
    idx = np.argsort(seq_len)
    # idx = range(len(seq_len))
    self.strokes = []
    for i in range(len(seq_len)):
      self.strokes.append(raw_data[idx[i]])
    print("total images <= max_seq_len is %d" % count_data)
    self.num_batches = int(count_data / self.batch_size)

  def random_sample(self):
    """Return a random sample, in stroke-3 format as used by draw_strokes."""
    sample = np.copy(random.choice(self.strokes))
    return sample

  def random_scale(self, data):
    """Augment data by stretching x and y axis randomly [1-e, 1+e]."""
    x_scale_factor = (
        np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
    y_scale_factor = (
        np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
    result = np.copy(data)
    result[:, 0] *= x_scale_factor
    result[:, 1] *= y_scale_factor
    return result

  def calculate_normalizing_scale_factor(self):
    """Calculate the normalizing factor explained in appendix of sketch-rnn."""
    data = []
    for i in range(len(self.strokes)):
      if len(self.strokes[i]) > self.max_seq_length:
        continue
      for j in range(len(self.strokes[i])):
        data.append(self.strokes[i][j, 0])
        data.append(self.strokes[i][j, 1])
    data = np.array(data)
    return np.std(data)

  def normalize(self, scale_factor=None):
    """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
    if scale_factor is None:
      scale_factor = self.calculate_normalizing_scale_factor()
    self.scale_factor = scale_factor
    for i in range(len(self.strokes)):
      self.strokes[i][:, 0:2] /= self.scale_factor

  def train_get_batch_from_indices(self, indices):
    """Given a list of indices, return the potentially augmented batch."""
    x_batch = []
    seq_len = []
    for idx in range(len(indices)):
      i = indices[idx]
      data = self.random_scale(self.strokes[i])
      data_copy = np.copy(data)
      #if self.augment_stroke_prob > 0:
      #  data_copy = augment_strokes(data_copy, self.augment_stroke_prob)
      x_batch.append(data_copy)
      length = len(data_copy)
      seq_len.append(length)

    # We return three things: stroke-3 format, stroke-5 format, list of seq_len.
    pad_data,pad_labels,pad_str_labels,new_seq_len,batch_triplets, saliency = self.train_pad_batch(x_batch, self.max_seq_length,seq_len)
    #seq_len = np.array(seq_len, dtype=int)
    return x_batch, pad_data,pad_labels,pad_str_labels, new_seq_len,batch_triplets, saliency

  def test_get_batch_from_indices(self, indices):
    """Given a list of indices, return the potentially augmented batch."""
    x_batch = []
    seq_len = []
    for idx in range(len(indices)):
      i = indices[idx]
      data = self.random_scale(self.strokes[i])
      data_copy = np.copy(data)
      #if self.augment_stroke_prob > 0:
      #  data_copy = augment_strokes(data_copy, self.augment_stroke_prob)
      x_batch.append(data_copy)
      length = len(data_copy)
      seq_len.append(length)

    # We return three things: stroke-3 format, stroke-5 format, list of seq_len.
    pad_data,pad_labels,pad_str_labels, saliency = self.test_pad_batch(x_batch, self.max_seq_length)
    seq_len = np.array(seq_len, dtype=int)
    print("saliency", saliency)
    return x_batch, pad_data, pad_labels, pad_str_labels, seq_len, saliency

  def random_batch(self):
    """Return a randomised portion of the training data."""
    idx = np.random.permutation(range(0, len(self.strokes)))[0:self.batch_size*2]
    # idx = range(0, len(self.strokes))
    return self.train_get_batch_from_indices(idx)

  def get_batch(self, idx):
    """Get the idx'th batch from the dataset."""
    assert idx >= 0, "idx must be non negative"
    assert idx < self.num_batches, "idx must be less than the number of batches"
    start_idx = idx * self.batch_size
    indices = range(start_idx, start_idx + self.batch_size)
    return self.test_get_batch_from_indices(indices)

  def get_triplets(self,str_label,ori_pad_group_label):
    unique_labels = np.unique(ori_pad_group_label)
    C = []
    diff_C = []
    anc = []
    pos = []
    neg = []
    gap_index = np.where(str_label==0)[0]
    pad_group_label = np.copy(ori_pad_group_label)
    pad_group_label[gap_index]=100
    triplet_nums_C1 = 0
    triplet_nums_C2=0
    triplet_nums_C3 = 0
    for unique_label in unique_labels:
      same_group_idx = np.where(pad_group_label==unique_label)[0]
      C.append(same_group_idx)
      temp_diff_group_idx = np.where(pad_group_label != unique_label)[0]
      diff_group_idx = [ss for ss in temp_diff_group_idx if ss not in gap_index]
      diff_C.append(diff_group_idx)
      triplet_nums_C1 += len(same_group_idx) * (len(same_group_idx) - 1) / 2 * min(len(diff_group_idx), 5)
      triplet_nums_C2 += len(same_group_idx) * (len(same_group_idx) - 1) / 2 * min(len(diff_group_idx), 20)
      triplet_nums_C3 += len(same_group_idx) * (len(same_group_idx) - 1) / 2 * min(len(diff_group_idx), 30)
    neg_in_each_triplet=0
    if triplet_nums_C1<3000:
      if triplet_nums_C2<3000:
        if triplet_nums_C3<3000:
          neg_in_each_triplet=4
        else:
          neg_in_each_triplet = 3
      else:
        neg_in_each_triplet = 2
    else:
      neg_in_each_triplet=1
    for C_idx, same_group_idxs in enumerate(C):
      diff_group_idxs = diff_C[C_idx]
      if neg_in_each_triplet == 1:
        d_idxs = np.random.permutation(range(len(diff_group_idxs)))[0:5]
      elif neg_in_each_triplet == 2:
        d_idxs = np.random.permutation(range(len(diff_group_idxs)))[0:20]
      elif neg_in_each_triplet == 3:
        d_idxs = np.random.permutation(range(len(diff_group_idxs)))[0:30]
      else:
        d_idxs = range(len(diff_group_idxs))
      for i in range(len(same_group_idxs)):
        for j in range(i + 1, len(same_group_idxs)):
          for d_idx in d_idxs:
            anc.append(same_group_idxs[i])
            pos.append(same_group_idxs[j])
            neg.append(diff_group_idxs[d_idx])

    idx = np.random.permutation(range(0, len(anc)))[0:3000]
    triplets = np.zeros((3,3000))
    for i in range(3000):
      idx_idx = np.mod(i,len(anc))
      triplets[0, i] = anc[idx[idx_idx]]
      triplets[1, i] = pos[idx[idx_idx]]
      triplets[2, i] = neg[idx[idx_idx]]
    return triplets

  def train_pad_batch(self, batch, max_len,seq_len):
    """Pad the batch to be stroke-5 bigger format as described in paper."""
    cluster_num = np.zeros((self.batch_size),dtype='int32')
    result = np.zeros((self.batch_size, max_len + 1, 5), dtype=float)
    labels = np.zeros((self.batch_size,max_len, max_len), dtype='int32')
    gap_seg_labels = np.ones((self.batch_size,max_len,max_len),dtype='int32')
    new_seq_len = np.zeros((self.batch_size),dtype='int32')
    pad_stroke_nums = []
    batch_triplets = np.zeros((self.batch_size,3,3000),dtype='int32')
    saliency = np.zeros((self.batch_size, max_len, max_len), dtype='float32')
    ii =0
    for i in range(len(batch)):
      stroke_labels = batch[i][:, 3]
      temp_str_label = np.ones((max_len, 1), dtype='int32')
      temp_label_matrix = np.zeros((max_len, max_len), dtype='int32')
      l = len(batch[i])
      assert l <= max_len
      for line_indx in range(l):
        if line_indx>0:
          if batch[i][line_indx-1,2]==1:
            temp_str_label[line_indx,0] = 0
        current_label = batch[i][line_indx,3]
        same_label_index = np.where(stroke_labels==current_label)[0]
        temp_labels = np.zeros((1,max_len),dtype='int32')
        temp_labels[0,same_label_index]=1
        temp_label_matrix[line_indx,:]=temp_labels
      temp_str_label[0, 0] = 0

      sum_labels = sum(sum(temp_label_matrix))
      one_label_accuracy = float(sum_labels)/(l*l)
      if ii<self.batch_size:
        if (one_label_accuracy>0.1) and (one_label_accuracy<0.9):
          gap_seg_labels[ii,:,:]=np.dot(temp_str_label,temp_str_label.T)
          labels[ii,:,:]=temp_label_matrix
          result[ii, 0:l, 0:2] = batch[i][:, 0:2]
          result[ii, 1:, :] = result[ii, :-1, :]
          result[ii, 0, :] = 0
          result[ii, 0:l, 3] = batch[i][:, 2]
          result[ii, 0:l, 2] = 1 - result[ii, 0:l, 3]
          result[ii, l:, 4] = 1
        # put in the first token, as described in sketch-rnn methodology

          result[ii, 0, 2] = self.start_stroke_token[2]  # setting S_0 from paper.
          result[ii, 0, 3] = self.start_stroke_token[3]
          result[ii, 0, 4] = self.start_stroke_token[4]
          cluster_num[ii]= len(np.unique(batch[i][:,3]))
          pad_stroke_nums.append(batch[i][:,3])

          batch_triplets[ii,:,:]=self.get_triplets(temp_str_label,batch[i][:,3])
          new_seq_len[ii] = seq_len[i]
          saliency[ii, :, :] = get_saliency(batch[i][:, 0:1], max_len)
          ii +=1
      else:
        # labels: (100, 300, 300)
        # gap_seg_labels: (100, 300, 300)
        return result, labels, gap_seg_labels, new_seq_len,batch_triplets, saliency


    #return result,labels,gap_seg_labels,new_seq_len
  # def temp_fun(self,C,diff_C,gap_index,neg_in_each_triplet,anc,pos,neg):

  def test_pad_batch(self, batch, max_len):
    """Pad the batch to be stroke-5 bigger format as described in paper."""
    result = np.zeros((self.batch_size, max_len + 1, 5), dtype=float)
    labels = np.zeros((self.batch_size,max_len, max_len), dtype='int32')
    gap_seg_labels = np.ones((self.batch_size,max_len,max_len),dtype='int32')
    saliency = np.zeros((self.batch_size, max_len, max_len), dtype='float32')
    #pad_stroke_nums = []
    assert len(batch) == self.batch_size
    for i in range(self.batch_size):
      l = len(batch[i])
      assert l <= max_len
      #current_stroke_num=0
      #stroke_nums = np.zeros((l),dtype='int32')
      result[i, 0:l, 0:2] = batch[i][:, 0:2]
      result[i, 0:l, 2] = 1 - result[i, 0:l, 3]
      result[i, l:, 4] = 1
      # put in the first token, as described in sketch-rnn methodology
      result[i, 1:, :] = result[i, :-1, :]
      result[i, 0, :] = 0
      result[i, 0, 2] = self.start_stroke_token[2]  # setting S_0 from paper.
      result[i, 0, 3] = self.start_stroke_token[3]
      result[i, 0, 4] = self.start_stroke_token[4]
      stroke_labels = batch[i][:, 3]
      temp_sep_label = np.ones((max_len,1),dtype='int32')
      # Saliency 
      saliency[i, :, :] = get_saliency(batch[i][:, 0:1], max_len)
      for line_indx in range(l):
        if line_indx>0:
          if batch[i][line_indx-1,2]==1:
            temp_sep_label[line_indx,0] = 0
       # stroke_nums[line_indx] = current_stroke_num
        #if batch[i][line_indx,2]==1:
        #  current_stroke_num = current_stroke_num+1
        current_label = batch[i][line_indx,3]
        same_label_index = np.where(stroke_labels==current_label)[0]
        temp_labels = np.zeros((1,max_len),dtype='int32')
        temp_labels[0,same_label_index]=1
        labels[i,line_indx,:]=temp_labels
      #pad_stroke_nums.append([batch[i][:,3]])
      temp_sep_label[0, 0] = 0
      gap_seg_labels[i,:,:]=np.dot(temp_sep_label,temp_sep_label.T)
    return result, labels, gap_seg_labels, saliency # ,pad_stroke_nums


def get_saliency(batch_strokes, max_len):
  saliency = np.random.uniform(size=(max_len, max_len))
  return saliency
