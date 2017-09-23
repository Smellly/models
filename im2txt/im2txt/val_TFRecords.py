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

import tensorflow as tf

from datetime import datetime

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary
from im2txt.ops import inputs as input_ops
from im2txt.ops import image_processing

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
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

def parse_sequence_example(serialized, image_id, image_feature, caption_feature):
  """Parses a tensorflow.SequenceExample into an image and caption.

  Args:
    serialized: A scalar string Tensor; a single serialized SequenceExample.
    image_feature: Name of SequenceExample context feature containing image
      data.
    caption_feature: Name of SequenceExample feature list containing integer
      captions.

  Returns:
    encoded_image: A scalar string Tensor containing a JPEG encoded image.
    caption: A 1-D uint64 Tensor with dynamically specified length.
  """
  context, sequence = tf.parse_single_sequence_example(
      serialized,
      context_features={
          image_id : tf.FixedLenFeature([], dtype=tf.int64),
          image_feature: tf.FixedLenFeature([], dtype=tf.string)
      },
      sequence_features={
          caption_feature: tf.FixedLenSequenceFeature([], dtype=tf.int64),
      })

  encoded_image_id = context[image_id]
  encoded_image = context[image_feature]
  caption = sequence[caption_feature]
  return encoded_image_id, encoded_image, caption

def process_image(encoded_image, config, thread_id=0):
  return image_processing.process_image(encoded_image,
                                        is_training=False,
                                        height=config.image_height,
                                        width=config.image_width,
                                        thread_id=thread_id,
                                          image_format=config.image_format)

def read(reader, config):
  data_files = []
  file_pattern = config.input_file_pattern
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                      len(data_files), file_pattern)

  filename_queue = tf.train.string_input_producer(
          data_files, shuffle=False, capacity=1)
  _, serialized_sequence_example = reader.read(filename_queue)
  encoded_image_id, encoded_image, caption = parse_sequence_example(
      serialized_sequence_example,
      image_id="image/image_id",
      image_feature=config.image_feature_name, #  "image/data"
      caption_feature=config.caption_feature_name) # "image/caption_ids"
  return encoded_image_id, encoded_image, caption

def main(_):
  assert FLAGS.input_file_pattern, "--input_file_pattern is required"
  model_config = configuration.ModelConfig()
  model_config.input_file_pattern = FLAGS.input_file_pattern
  # Build the inference graph.
  g = tf.Graph()
  tf.reset_default_graph()
  with g.as_default():
    reader = tf.TFRecordReader()
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(model_config,
                                               FLAGS.checkpoint_path)
  #g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
  save_path = FLAGS.output_files
  tf.logging.info('save_path : %s'%save_path)
  results = []
  records = []

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)
    
    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    images_and_captions = read(reader, model_config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    tf.logging.info(
                "%s Start processing image caption generation in dataset." %
                (datetime.now()))
    
    for step in xrange(2100):
      img_id, encoded_image, _ = sess.run(images_and_captions)
      if img_id in records:
        continue
      else:
        records.append(img_id)
      captions = generator.beam_search(sess, encoded_image)
      ppl = [math.exp(x.logprob) for x in captions]
      caption = captions[ppl.index(max(ppl))]
      sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
      sentence = " ".join(sentence)
      results.append({"image_id":img_id, "caption":sentence})
      if step % 100 == 0:
        tf.logging.info(
              "%d steps: %s: Finished processing %d image caption generation in dataset." %
              (step, datetime.now(), len(results)))

    coord.request_stop()
    coord.join(threads)

    tf.logging.info(
        "%s: Finished processing all %d image caption generation in dataset." %
        (datetime.now(), len(results)))
    with open(save_path, 'w') as f:
      json.dump(results, f)

if __name__ == "__main__":
  tf.app.run()
