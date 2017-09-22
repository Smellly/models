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
import json

from tqdm import tqdm
from ast import literal_eval

import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

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

tf.logging.set_verbosity(tf.logging.INFO)

def getValID(path):
  with open(path, 'r') as f:
    raw = json.load(f)
  return raw['images']

'''
def getValAttr(path):
  with open(path, 'r') as f:
    attr_data = json.load(f)
  filename_to_attribute = {}
  for filename, attribute in attr_data.iteritems():
    p = [literal_eval(i.split(':')[1])[0] for i in attribute]
    filename_to_attribute[filename] = p 
  return filename_to_attribute
'''
  
def main(_):
  # Build the inference graph.
  g = tf.Graph()
  tf.reset_default_graph()
  print(FLAGS.checkpoint_path)
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    model_ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
    assert(model_ckpt != None)
    print(model_ckpt.model_checkpoint_path)
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               model_ckpt.model_checkpoint_path.replace('train', 'raw_adam'))
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  img_path, annos_path = FLAGS.input_files.split(",")
  save_path = FLAGS.output_files
  results = []
  # record image which have been process
  records = []
  savefreq = 10000
  record_path = save_path.replace('results', 'records')
  print('record_path : %s'%record_path)
  print('save_path : %s'%save_path)

  filenames = getValID(annos_path)

  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)
    try:
      with open(record_path, 'r') as f:
        record = json.load(f)
    except:
      print('no record to read')

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)
    epoch = 0

    for item in tqdm(filenames):
      if item['id'] in records:
        continue
      filename = img_path + item['file_name']
      # print(filename)
      with tf.gfile.GFile(filename, "r") as f:
        image = f.read()
      try:
        captions = generator.beam_search(sess, image)
        ppl = [math.exp(x.logprob) for x in captions]
        caption = captions[ppl.index(max(ppl))]
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        results.append({"image_id":item['id'], "caption":sentence})
        records.append(item['id'])
      except:
        print('filename %s is broken'%item['file_name'])
      finally:
        if epoch % savefreq == 0:
            print('%d times temporally saving...'%(int(epoch/savefreq)))
            with open(save_path, 'w') as f:
              json.dump(results, f)
            with open(record_path, 'w') as f:
              json.dump(records, f)
        epoch += 1
    
    with open(save_path, 'w') as f:
      json.dump(results, f)
    with open(record_path, 'w') as f:
      json.dump(records, f)


if __name__ == "__main__":
  tf.app.run()
