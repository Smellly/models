#!/bin/bash
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

# Script to download and preprocess the MSCOCO data set.
#
# The outputs of this script are sharded TFRecord files containing serialized
# SequenceExample protocol buffers. See build_mscoco_data.py for details of how
# the SequenceExample protocol buffers are constructed.
#
# usage:
#  ./download_and_preprocess_mscoco.sh
set -e

# Create the output directories.
OUTPUT_DIR="${HOME}/projects/ic_models/im2txt/im2txt/data/blend_mscoco_with_flickr30"
COCO_DATA_DIR="${HOME}/projects/ic_models/im2txt/im2txt/data/mscoco"
FLRKR_DATA_DIR="${HOME}/projects/ic_models/im2txt/im2txt/data/flickr30"
SCRATCH_DIR="${OUTPUT_DIR}/raw-data"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRATCH_DIR}"
CURRENT_DIR=$(pwd)
WORK_DIR="$0.runfiles/im2txt/im2txt"

cd ${SCRATCH_DIR}

COCO_TRAIN_IMAGE_DIR="${COCO_DATA_DIR}/raw-data/train2014"
COCO_VAL_IMAGE_DIR="${COCO_DATA_DIR}/raw-data/val2014"
FLRKR_IMAGE_DIR="${FLRKR_DATA_DIR}/raw-data/flickr30k-images"

COCO_TRAIN_CAPTIONS_FILE="${COCO_DATA_DIR}/raw-data/annotations/captions_train2014.json"
COCO_VAL_CAPTIONS_FILE="${COCO_DATA_DIR}/raw-data/annotations/captions_val2014.json"
FLRKR_CAPTIONS_FILE="${FLRKR_DATA_DIR}/raw-data/dataset_flickr30k.json"

# Build TFRecords of the image data.
cd "${CURRENT_DIR}"
BUILD_SCRIPT="${WORK_DIR}/build_mscoco_blend_flickr30_data"
"${BUILD_SCRIPT}" \
  --coco_train_image_dir="${COCO_TRAIN_IMAGE_DIR}" \
  --coco_val_image_dir="${COCO_VAL_IMAGE_DIR}" \
  --flickr_image_dir="${FLRKR_IMAGE_DIR}" \
  --coco_train_captions_file="${COCO_TRAIN_CAPTIONS_FILE}" \
  --coco_val_captions_file="${COCO_VAL_CAPTIONS_FILE}" \
  --flickr_captions_file="${FLRKR_CAPTIONS_FILE}" \
  --output_dir="${OUTPUT_DIR}" \
  --word_counts_output_file="${OUTPUT_DIR}/wordcounts.txt" \
