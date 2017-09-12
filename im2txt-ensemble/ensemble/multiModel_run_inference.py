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


import tensorflow as tf

from ensemble import configuration
from ensemble import inference_wrapper
from ensemble.inference_utils import caption_generator
from ensemble.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns "
                       "of image files.")

tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
  ensemble = [1, 2]
  num = len(ensemble)
  models = []
  generators = []
  
  for en in ensemble:
    print('ensemble model:', FLAGS.checkpoint_path + str(en))
    # graph_def_file = FLAGS.checkpoint_path + str(en) + '/model.ckpt-1000000.meta'
    saver_def_file = FLAGS.checkpoint_path + str(en) 
    g = tf.Graph()
    tf.reset_default_graph()
    d = {}
    # print(FLAGS.checkpoint_path)
    with g.as_default():
      d['model'] = inference_wrapper.InferenceWrapper()
      model_ckpt = tf.train.get_checkpoint_state(saver_def_file)
      assert(model_ckpt != None)
      d['restore_fn'] = d['model'].build_graph_from_config(
                                                configuration.ModelConfig(),
                                                model_ckpt.model_checkpoint_path)
      # d['model_saver']  = tf.train.import_meta_graph(graph_def_file)
      # d['model_ckpt'] = tf.train.get_checkpoint_state(saver_def_file)
      # assert(d['model_ckpt'] != None)
      # d['restore_fn'] = d['model_saver'].restore
      
    g.finalize()
    d['g'] = g
    models.append(d)

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  filenames = []
  print(FLAGS.input_files.split(","))
  print(FLAGS.input_files.split(",")[0])
  print(tf.gfile.Glob(FLAGS.input_files.split(",")[0]))

  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  for ind, model in enumerate(models):
    with tf.Session(graph=model['g']) as sess:
      # Load the model from checkpoint.
      # model['restore_fn'](sess)
      model['restore_fn'](sess)#, d['model_ckpt'].model_checkpoint_path)
      # Prepare the caption generator. Here we are implicitly using the default
      # beam search parameters. See caption_generator.py for a description of the
      # available beam search parameters.
      models[ind]['generator'] = caption_generator.CaptionGenerator(d['model'], vocab)
      print("INFO: model %d rebuild"%ind)
 
  tf.logging.info("Loading")
  for filename in filenames:
    with tf.gfile.GFile(filename, "r") as f:
      image = f.read()

    for ind, model in enumerate(models):
      with tf.Session(graph=model['g']) as sess:
        # Load the model from checkpoint.
        model['restore_fn'](sess)#, d['model_ckpt'].model_checkpoint_path)
        
        print("INFO: model %d beam search"%ind)
        captions = model['generator'].beam_search(sess, image)

        print("INFO: Model %d Captions for image %s:" % (ind, os.path.basename(filename)))
        for i, caption in enumerate(captions):
          # Ignore begin and end words.
          sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
          sentence = " ".join(sentence)
          print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

if __name__ == "__main__":
  tf.app.run()
