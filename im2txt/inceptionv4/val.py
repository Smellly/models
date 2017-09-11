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

import tensorflow as tf

from inceptionv4 import configuration
from inceptionv4 import inference_wrapper
from inceptionv4.inference_utils import caption_generator
from inceptionv4.inference_utils import vocabulary

FLAGS = tf.app.flags.FLAGS

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

def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  img_path, annos_path = FLAGS.input_files.split(",")
  save_path = FLAGS.output_files
  record_path = save_path.replace('results', 'records')
  print('save_path : %s'%save_path)
  print('record_path : %s'%record_path)
  results = []
  records = []
  epoch = 0
  savefreq = 400
  # print(img_path)
  # print(annos_path)
  filenames = getValID(annos_path)
  # print(type(filenames)) # list
  # print(len(filenames))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  try:
    with open(record_path, 'r' ) as f:
      records = json.load(f)
  except:
    print('no records to read')

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

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
        print('filename %s is broken'%item['fine_name'])
      finally:
        epoch += 1
        if epoch % savefreq == 0:
          print('%d times saving'%(int(epoch/savefreq)))
          with open(save_path, 'w') as f:
            json.dump(results, f)
          with open(record_path, 'w') as f:
            json.dump(records, f)
    
    with open(save_path, 'w') as f:
      json.dump(results, f)
    with open(record_path, 'w') as f:
      json.dump(records, f)

if __name__ == "__main__":
  tf.app.run()
