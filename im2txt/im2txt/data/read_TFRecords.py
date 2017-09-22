from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import sys

sys.path.append('/home/smelly/projects/ic_models/im2txt/im2txt')
sys.path.append('/home/smelly/projects/ic_models/im2txt/im2txt/ops')

import inputs as input_ops
import configuration
import image_processing

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
    """Decodes and processes an image string.

    Args:
      encoded_image: A scalar string Tensor; the encoded image.
      thread_id: Preprocessing thread id used to select the ordering of color
        distortions.

    Returns:
      A float32 Tensor of shape [height, width, 3]; the processed image.
    """
    return image_processing.process_image(encoded_image,
                                          is_training=False,
                                          height=config.image_height,
                                          width=config.image_width,
                                          thread_id=thread_id,
                                          image_format=config.image_format)

def read(config):
    reader = tf.TFRecordReader()
    input_queue = input_ops.prefetch_input_data(
          reader,
          # config.input_file_pattern,
          file_pattern='mscoco/val-?????-of-00004',
          values_per_shard=config.values_per_input_shard,
          is_training=False,
          batch_size=config.batch_size)
    # Image processing and random distortion. Split across multiple threads
    # with each thread applying a slightly different distortion.
    assert config.num_preprocess_threads % 2 == 0
    # 4
    images_and_captions = []
    serialized_sequence_example = input_queue.dequeue()
    '''
    context = tf.train.Features(feature={
      "image/image_id": _int64_feature(image.image_id),
      "image/data": _bytes_feature(encoded_image),
    })
    feature_lists = tf.train.FeatureLists(feature_list={
      "image/caption": _bytes_feature_list(caption),
      "image/caption_ids": _int64_feature_list(caption_ids)
    })
    '''
    # for thread_id in range(config.num_preprocess_threads):
    encoded_image_id, encoded_image, caption = parse_sequence_example(
        serialized_sequence_example,
        image_id="image/image_id",
        image_feature=config.image_feature_name, #  "image/data"
        caption_feature=config.caption_feature_name) # "image/caption_ids"
    #image = process_image(encoded_image, config, thread_id=0)
    images_and_captions.append([encoded_image_id, encoded_image, caption])
    return [encoded_image_id, encoded_image, caption]

    # Batch inputs.
    # queue_capacity = (2 * config.num_preprocess_threads *
    #                 config.batch_size)
    # images, input_seqs, target_seqs, input_mask = (
    #   input_ops.batch_with_dynamic_pad(images_and_captions,
    #                                    batch_size=config.batch_size,
    #                                    queue_capacity=queue_capacity))
    return images_and_captions

def main(_):
    config = configuration.ModelConfig()
    #image, caption = read(config)
    images_and_captions = read(config)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess)
        i = 0
        while True:
            print('i:', i),
            images_and_captions = sess.run([images_and_captions])
            print('len of images_and_captions', len(images_and_captions))
            for ii in images_and_captions:
                print(ii[1])
                print(ii[1].shape)
                print(ii[2])
            # print(img_id, val.shape, l.shape)
            # print(l.shape)
            i += 1
            break
        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    tf.app.run()