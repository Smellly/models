from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from ensemble.ops import inputs as input_ops
from ensemble import configuration

def read(config):
    reader = tf.TFRecordReader()
    input_queue = input_ops.prefetch_input_data(
          reader,
          config.input_file_pattern,
          is_training=False,
          batch_size=config.batch_size)
    # Image processing and random distortion. Split across multiple threads
    # with each thread applying a slightly different distortion.
    assert self.config.num_preprocess_threads % 2 == 0
    images_and_captions = []
    for thread_id in range(self.config.num_preprocess_threads):
        serialized_sequence_example = input_queue.dequeue()
        encoded_image, caption = input_ops.parse_sequence_example(
            serialized_sequence_example,
            image_feature=self.config.image_feature_name,
            caption_feature=self.config.caption_feature_name)
        image = self.process_image(encoded_image, thread_id=thread_id)
        images_and_captions.append([image, caption])
    return images_and_captions

def main():
    with with tf.Session() as sess:
        sess.run(read(configuration.ModelConfig()))
