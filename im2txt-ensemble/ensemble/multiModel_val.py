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
  ensemble = [2]
  num = len(ensemble)
  models = []
  generators = []
  for en in ensemble:
    print('ensemble model:', FLAGS.checkpoint_path + str(en))
    saver_def_file = FLAGS.checkpoint_path + str(en)
    # Build the inference graph.
    g = tf.Graph()
    tf.reset_default_graph()
    d = {}

    with g.as_default():
      d['model'] = inference_wrapper.InferenceWrapper()
      model_ckpt = tf.train.get_checkpoint_state(saver_def_file)
      assert(model_ckpt != None)
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

  # wait for full version of val
  # img_path, annos_path, attrs_path = FLAGS.input_files.split(",")
  # save_path = FLAGS.output_files

  # filenames = getValID(annos_path)
  # filename_to_attribute = getValAttr(attrs_path)

  filenames = []
  print('DEBUG', FLAGS.input_files.split(","))
  print('DEBUG', FLAGS.input_files.split(",")[0])
  print('DEBUG', tf.gfile.Glob(FLAGS.input_files.split(",")[0]))

  for file_pattern in FLAGS.input_files.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  tf.logging.info("Loading filenames")
  for filename in filenames:
    with tf.gfile.GFile(filename, "r") as f:
      image = f.read()

    # generate the first word
    p0 = 0.

    print("DEBUG: Generate the first word") 
    for ind, model in enumerate(models):
      # Load the model from checkpoint.
      model['restore_fn'](model['sess'])
      # Prepare the caption generator. Here we are implicitly using the default
      # beam search parameters. See caption_generator.py for a description of the
      # available beam search parameters.
      models[ind]['generator'] = caption_generator.CaptionGenerator(model['model'], vocab)
      print("DEBUG: model %d rebuild"%ind) 
      print("DEBUG: in model %d generates the first word"%ind) 
      ((softmax, new_states, metadata), 
        captions_tuple) = model['generator'].beam_search_first_word(model['sess'], image)

      models[ind]['state'] = new_states
      models[ind]['metadata'] = metadata # None actually
      models[ind]['captions_tuple'] = captions_tuple
      # print(type(softmax), len(softmax[0]), softmax) 
      # <type 'numpy.ndarray'> 12000(word dict length)
      
      # print('DEBUG:new_states', new_states)
      # print('DEBUG:metadata', metadata)
      maxy0 = np.amax(softmax) 
      print(type(maxy0), maxy0)
      '''
      # <type 'numpy.float32'> 0.855502
      # for numerical stability shift into good numerical range
      e0 = np.exp(softmax - maxy0) 
      p0 = p0 + e0 / np.sum(e0)
      '''
      p0 = p0 + softmax
    p0 = p0/num
    print(np.amax(p0))
    # print(type(p0), len(p0), p0) 
    # <type 'numpy.ndarray'> 12000(word dict length)

    for ind, model in enumerate(models):
      models[ind]['captions'] = model['generator'].\
                        beam_search_prob2word((p0, model['state'], model['metadata']), 
                                              model['captions_tuple'])
      print('DEBUG: in model', ind, 'captions', model['captions'])

    # generate the rest n words
    max_caption_length = 20
    
    print("DEBUG: Generate the rest words") 
    for ii in range(max_caption_length - 1):
      p1 = 0.
      
      for ind, model in enumerate(models):
        print("DEBUG: in model %d generates the rest words %d times"%(ind, ii)) 
        # print(type(model['captions']))
        if len(model['captions']) == 1:
          captions = model['captions'][0].extract(sort=True)
          break # maybe with bugs

        ((softmax, new_states, metadata), 
          captions_tuple) = model['generator'].beam_search_one_step(model['sess'], model['captions'])

        # print(type(softmax), len(softmax[0]), softmax) 
        maxy1 = np.amax(softmax)
        # print(type(maxy1), maxy1)
        '''
        # for numerical stability shift into good numerical range
        e1 = np.exp(softmax - maxy1) 
        p1 = p1 + e1 / np.sum(e1)
        '''
        p1 = p1 + softmax
        models[ind]['state'] = new_states
        models[ind]['metadata'] = metadata
        models[ind]['captions_tuple'] = captions_tuple
      else:
        # after all the for then we will go on 
        p1 = p1/num
        # print(type(p1), len(p1), p1) 

        for model in models:
          models[ind]['captions'] = model['generator'].\
                  beam_search_prob2word((p1, model['state'], model['metadata']), 
                                        model['captions_tuple'])
    else:
      # with bugs
      partial_captions, partial_captions_list, complete_captions = model['captions_tuple']
      # If we have no complete captions then fall back to the partial captions.
      # But never output a mixture of complete and partial captions because a
      # partial caption could have a higher score than all the complete captions.
      if not complete_captions.size():
        captions = partial_captions.extract(sort=True)
      else:
        captions = complete_captions.extract(sort=True)

    print("Captions for image %s:" % os.path.basename(filename))
    for i, caption in enumerate(captions):
      # Ignore begin and end words.
      sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
      sentence = " ".join(sentence)
      print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

    break


if __name__ == "__main__":
  tf.app.run()
