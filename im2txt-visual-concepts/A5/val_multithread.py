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
import threading

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
    # print(filename)
    with tf.gfile.GFile(filename, "r") as f:
      image = f.read()
    attribute = filename_to_attribute[item['file_name']]
    captions = generator.beam_search(sess, image, attribute)
    ppl = [math.exp(x.logprob) for x in captions]
    caption = captions[ppl.index(min(ppl))]
    sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
    sentence = " ".join(sentence)
    results.append({"image_id":item['id'], "caption":sentence})

  with open(save_path.replace('.json', str(thread_index)+'.json'), 'w') as f:
      json.dump(results, f)

  print("%s: Thread %d finished processing all %d image caption generation." %
          (datetime.now(), thread_index, len(filenames)))

      
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

  img_path, annos_path, attrs_path = FLAGS.input_files.split(",")
  save_path = FLAGS.output_files

  filenames = getValID(annos_path)
  filename_to_attribute = getValAttr(attrs_path)
  
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), FLAGS.input_files)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)
    results = []

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    # Break the images into num_threads batches. Batch i is defined as
    # images[ranges[i][0]:ranges[i][1]].
    num_threads = FLAGS.num_threads
    # supposed we have 20 images and 5 threads
    # np.linspace(0, 20, 6).astype(np.int) = array([ 0,  4,  8, 12, 16, 20])
    spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in xrange(len(spacing) - 1):
      ranges.append([spacing[i], spacing[i + 1]])

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Launch a thread for each batch.
    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in xrange(len(ranges)):
      args = (thread_index, filenames, ranges, img_path, generator, vocab, save_path)
      t = threading.Thread(target=process, args=args)
      t.start()
      threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)

    for thread_index in xrange(len(ranges)):
      with open(save_path.replace('.json', str(thread_index)+'.json'), 'r') as f:
        results.extend(json.load(f))

    with open(save_path, 'r') as f:
        json.dump(results, f)

    print("%s: Finished processing all %d image caption generation in data set '%s'." %
          (datetime.now(), len(filenames), img_path))



if __name__ == "__main__":
  tf.app.run()