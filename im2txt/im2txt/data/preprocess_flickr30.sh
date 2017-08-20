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
OUTPUT_DIR="${HOME}/projects/ic_models/im2txt/im2txt/data/flickr30"
DATA_DIR="${HOME}/projects/ic_models/im2txt/im2txt/data"
SCRATCH_DIR="${OUTPUT_DIR}/raw-data"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SCRATCH_DIR}"
CURRENT_DIR=$(pwd)
WORK_DIR="$0.runfiles/im2txt/im2txt"

cd ${SCRATCH_DIR}

TRAIN_IMAGE_DIR="${DATA_DIR}/mscoco/raw-data/train2014"
VAL_IMAGE_DIR="${DATA_DIR}/mscoco/raw-data/val2014"
IMAGE_DIR="${SCRATCH_DIR}/flickr30k-images"

TRAIN_CAPTIONS_FILE="${DATA__DIR}/mscoco/raw-data/annotations/captions_train2014.json"
VAL_CAPTIONS_FILE="${DATA_DIR}/mscoco/raw-data/annotations/captions_val2014.json"
CAPTIONS_FILE="${SCRATCH_DIR}/dataset_flickr30k.json"

# Build TFRecords of the image data.
cd "${CURRENT_DIR}"
BUILD_SCRIPT="${WORK_DIR}/build_flickr30_data"
"${BUILD_SCRIPT}" \
  --train_image_dir="${IMAGE_DIR}" \
  --val_image_dir="${VAL_IMAGE_DIR}" \
  --train_captions_file="${CAPTIONS_FILE}" \
  --val_captions_file="${VAL_CAPTIONS_FILE}" \
  --output_dir="${OUTPUT_DIR}" \
  --word_counts_output_file="${OUTPUT_DIR}/word_counts.txt" \
