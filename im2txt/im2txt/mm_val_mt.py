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
r"""Generate captions for images using default beam search parameters.
    With multi models ensemble & multithread
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import numpy as np
import json
import threading

from ast import literal_eval
from tqdm import tqdm
from datetime import datetime

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
tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to preprocess the images.")

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

def process(thread_index, filenames, ranges, img_path, generator, vocab, save_path):
  results = []
  for item in filenames[ranges[thread_index][0]:ranges[thread_index][1]]:
    filename = img_path + item['file_name'] 
    with tf.gfile.GFile(filename, "r") as f:
      image = f.read()
    try:
      captions = generator.beam_search(image)
      ppl = [math.exp(x.logprob) for x in captions]
      caption = captions[ppl.index(max(ppl))]
      sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
      sentence = " ".join(sentence)
      results.append({"image_id":item['id'], "caption":sentence})
    except:
      tf.logging.info('Thread %d filename %s is broken'%(thread_index, item['file_name']))
    finally:
      pass

  with open(save_path.replace('.json', str(thread_index)+'.json'), 'w') as f:
      json.dump(results, f)

  tf.logging.info("%s: Thread %d finished processing all %d image caption generation." %
          (datetime.now(), thread_index, len(filenames)))

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
        print('DEBUG:', model_ckpt)
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

  img_path, annos_path, attrs_path = FLAGS.input_files.split(",")
  save_path = FLAGS.output_files

  tf.logging.info("save_path : %s"%save_path)
  tf.logging.info("Loading filenames")
  filenames = getValID(annos_path)
  # filename_to_attribute = getValAttr(attrs_path)

  results = []
  num_threads = FLAGS.num_threads

  # supposed we have 20 images and 5 threads
  # np.linspace(0, 20, 6).astype(np.int) = array([ 0,  4,  8, 12, 16, 20])
  spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in xrange(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Launch a thread for each batch.
  tf.logging.info("Launching %d threads for spacings: %s" % (num_threads, ranges))
  for thread_index in xrange(len(ranges)):
    args = (thread_index, filenames, ranges, img_path, generator, vocab, save_path)
    t = threading.Thread(target=process, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)

  for thread_index in xrange(len(ranges)):
    tf.logging.info(
        "%s: Loading json in dataset '%d'." %
        (datetime.now(), thread_index))

    with open(save_path.replace('.json', str(thread_index)+'.json'), 'r') as f:
      results.extend(json.load(f))

  with open(save_path, 'w') as f:
      json.dump(results, f)

  tf.logging.info(
        "%s: Finished processing all %d image caption generation in data set '%s'." %
        (datetime.now(), len(filenames), img_path))


if __name__ == "__main__":
  tf.app.run()
