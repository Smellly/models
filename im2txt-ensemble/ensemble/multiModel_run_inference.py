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
  ensemble = [1]
  num = len(ensemble)
  models = []
  generators = []
  for en in ensemble:
    # Build the inference graph.
    g = tf.Graph()
    tf.reset_default_graph()
    d = {}
    with g.as_default():
      d['model'] = inference_wrapper.InferenceWrapper()
      d['restore_fn'] = model.build_graph_from_config(configuration.ModelConfig(),
                                                 FLAGS.checkpoint_path + str(i))
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
      model['restore_fn'](sess)
      # Prepare the caption generator. Here we are implicitly using the default
      # beam search parameters. See caption_generator.py for a description of the
      # available beam search parameters.
      models[ind]['generator'] = caption_generator.CaptionGenerator(model, vocab)

 
  for filename in filenames:
    with tf.gfile.GFile(filename, "r") as f:
      image = f.read()

    # generate the first word
    p0 = 0.

    for ind, model in enumerate(models):
      with tf.Session(graph=model['g']) as sess:
        # Load the model from checkpoint.
        model['restore_fn'](sess)
    
        ((softmax, new_states, metadata), 
          captions_tuple) = model['generator'].beam_search_first_word(sess, image)
      models[ind]['state'] = new_states
      models[ind]['metadata_set'] = metadata
      models[ind]['captions_tuple'] = captions_tuple
      maxy0 = np.amax(softmax[0])
      # for numerical stability shift into good numerical range
      e0 = np.exp(softmax[0] - maxy0) 
      p0 = p0 + e0 / np.sum(e0)

    p0 = p0/num

    for model in models:
      with tf.Session(graph=model['g']) as sess:
        # Load the model from checkpoint.
        model['restore_fn'](sess)
        captions = model['generator'].\
                          beam_search_prob2word(sess, 
                                                (p0, model['state'], model['metadata']), 
                                                model['captions_tuple'])
    # generate the rest n words
    max_caption_length = 20
    
    for _ in range(max_caption_length - 1):
      p1 = 0.
      for ind, model in enumerate(models):
        with tf.Session(graph=model['g']) as sess:
          model['restore_fn'](sess)
          
          if len(captions == 1):
            captions.extract(sort=True)
            break

          ((softmax, new_states, metadata), 
            captions_tuple) = model['generator'].beam_search_one_step(sess, captions)

          maxy1 = np.amax(softmax[0])
          # for numerical stability shift into good numerical range
          e1 = np.exp(softmax[0] - maxy1) 
          p1 = p1 + e1 / np.sum(e1)

          models[ind]['state'] = new_states
          models[ind]['metadata_set'] = metadata
          models[ind]['captions_tuple'] = captions_tuple

    p1 = p1/num
    for model in models:
      with tf.Session(graph=model['g']) as sess:
        # Load the model from checkpoint.
        model['restore_fn'](sess)
        captions = model['generator'].\
                beam_search_prob2word(sess, 
                                      (p1, model['state'], model['metadata']), 
                                      model['captions_tuple'])

    print("Captions for image %s:" % os.path.basename(filename))
    for i, caption in enumerate(captions):
      # Ignore begin and end words.
      sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
      sentence = " ".join(sentence)
      print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))


if __name__ == "__main__":
  tf.app.run()
