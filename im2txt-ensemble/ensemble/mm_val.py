# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import numpy as np
import json

from ast import literal_eval
from tqdm import tqdm

import tensorflow as tf

from ensemble import configuration
from ensemble import inference_wrapper
from ensemble.inference_utils import mm_caption_generator as caption_generator
from ensemble.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")
tf.flags.DEFINE_string("output_files", "",
                       "validation set saving path.")
tf.flags.DEFINE_string("debug_mode", "",
                       "debug or not.")

tf.logging.set_verbosity(tf.logging.INFO)

def getValID(path):
  with open(path, 'r') as f:
    raw = json.load(f)
  return raw['images']

def getValAttr(path):
  with open(path, 'r') as f:
    attr_data = json.load(f)
  filename_to_attribute = {}
  for filename, attribute in attr_data.iteritems():
    p = [literal_eval(i.split(':')[1])[0] for i in attribute]
    filename_to_attribute[filename] = p 
  return filename_to_attribute

def main(_):
  debug_mode = True if FLAGS.debug_mode == 'debug' else False
  ensemble = [2, 4, 6, 7, 8]
  num = len(ensemble)
  models = []
  generators = []
  for en in ensemble:
    saver_def_file = FLAGS.checkpoint_path + str(en)
    if debug_mode:
      print('DEBUG:ensemble model:', saver_def_file)
    # Build the inference graph.
    g = tf.Graph()
    tf.reset_default_graph()
    d = {}

    with g.as_default():
      d['model'] = inference_wrapper.InferenceWrapper()
      model_ckpt = tf.train.get_checkpoint_state(saver_def_file)
      assert(model_ckpt != None)
      if debug_mode:
        print('DEBUG: model_ckpt', model_ckpt)
      d['restore_fn'] = d['model'].build_graph_from_config(
                                                configuration.ModelConfig(),
                                                model_ckpt.model_checkpoint_path)
      # Sessions created in this scope will run operations from `g`.
      d['sess'] = tf.Session()
    g.finalize()
    d['g'] = g
    models.append(d)

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
  tf.logging.info("Generating Beam Search Model")
  for ind, m in enumerate(models):
    if debug_mode:
      print('DEBUG:loading model %d:'%(ind+1))
    m['restore_fn'](m['sess'])
  generator = caption_generator.CaptionGenerator(models, vocab)

  if debug_mode:
    filenames = FLAGS.input_files
  else:
    img_path, annos_path, attr_data = FLAGS.input_files.split(",")
    save_path = FLAGS.output_files

    filenames = getValID(annos_path)
    # filename_to_attribute = getValAttr(attrs_path)

  epoch = 0
  savefreq = 200
  filenames = []
  results = []
  records = []
  if not debug_mode:
    record_path = save_path.replace('results', 'records')
    tf.logging.info("record_path : %s"%record_path)
    tf.logging.info("save_path : %s"%save_path)
    try:
      tf.logging.info("Loading records")
      with open(record_path, 'r') as f:
        record = json.load(f)
    except:
      tf.logging.info("no records to read")

  if debug_mode:
    print('DEBUG: FLAGS.input_files.split(",")', 
                  FLAGS.input_files.split(","))
    print('DEBUG: FLAGS.input_files.split(",")[0]',
                  FLAGS.input_files.split(",")[0])
    print('DEBUG: tf.gfile.Glob(FLAGS.input_files.split(",")[0])', 
                  tf.gfile.Glob(FLAGS.input_files.split(",")[0]))

  if debug_mode:
    for file_pattern in FLAGS.input_files.split(","):
      filenames.extend(tf.gfile.Glob(file_pattern))
  else:
    tf.logging.info("Loading filenames")
    filenames = getValID(annos_path)

  tf.logging.info("Running caption generation on %d files matching %s",
                    len(filenames), FLAGS.input_files)

  for item in tqdm(filenames):
    if debug_mode:
      filename = item
      print('DEBUG: filename', filename)
    elif item['id'] in records:
      continue
    else:
      filename = img_path + item['file_name']
    
    with tf.gfile.GFile(filename, "r") as f:
      image = f.read()
    if debug_mode:
      captions = generator.beam_search(image)
      ppl = [math.exp(x.logprob) for x in captions]
      caption = captions[ppl.index(max(ppl))]
      sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
      sentence = " ".join(sentence)
      print('DEBUG: sentence:', sentence)
    else:
      try:
        captions = generator.beam_search(image)
        ppl = [math.exp(x.logprob) for x in captions]
        caption = captions[ppl.index(max(ppl))]
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        if debug_mode:
          print('DEBUG: sentence:', sentence)
        else:
          results.append({"image_id":item['id'], "caption":sentence})
          records.append(item['id'])
      except:
        if not debug_mode:
          print('filename %s is broken'%item['file_name'])
        else:
          print('DEBUG: something wrong in beam_search')
      finally:
        if not debug_mode:
          if epoch % savefreq == 0:
              tf.logging.info('%d times temporally saving...'%(int(epoch/savefreq)))
              with open(save_path, 'w') as f:
                json.dump(results, f)
              with open(record_path, 'w') as f:
                json.dump(records, f)
          epoch += 1

  if not debug_mode:
    with open(save_path, 'w') as f:
      json.dump(results, f)
    with open(record_path, 'w') as f:
      json.dump(records, f)


if __name__ == "__main__":
  tf.app.run()
