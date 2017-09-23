# Path to checkpoint file or a directory containing checkpoint files. Passing
# a directory will only work if there is also a file named 'checkpoint' which
# lists the available checkpoints in the directory. It will not work if you
# point to a directory with just a copy of a model checkpoint: in that case,
MSCOCO_DIR="${HOME}/projects/ic_models/im2txt-visual-concepts/im2txt/data/mscoco-visual-concepts"

# you will need to pass the checkpoint path explicitly.
CHECKPOINT_PATH="${HOME}/projects/ic_models/im2txt-visual-concepts/im2txt/model/train"

# Vocabulary file generated by the preprocessing script.
VOCAB_FILE="${HOME}/projects/ic_models/im2txt-visual-concepts/im2txt/data/mscoco-visual-concepts/word_counts.txt"

# JPEG image file to caption.
IMAGE_FILE="${HOME}/projects/ic_models/im2txt-visual-concepts/im2txt/data/mscoco-visual-concepts/raw-data/val2014/,${HOME}/projects/ic_models/im2txt-visual-concepts/im2txt/data/mscoco-visual-concepts/raw-data/annotations/captions_val2014.json,${HOME}/projects/ic_models/im2txt-visual-concepts/im2txt/data/mscoco-visual-concepts/raw-data/val_sorted.json,${HOME}/projects/ic_models/im2txt-visual-concepts/im2txt/data/mscoco-visual-concepts/raw-data/annotations/val2014_filename_to_imgid.json"

OUTPUT_FILE="${HOME}/projects/ic_models/im2txt-visual-concepts/A5/results/results_5k_a3_maxppl.json"
# Build the inference binary.
cd ${HOME}/projects/ic_models/im2txt-visual-concepts
# bazel build -c opt //A5:val
bazel build -c opt //im2txt:val_TFRecords

# Ignore GPU devices (only necessary if your GPU is currently memory
# constrained, for example, by running the training script).
export CUDA_VISIBLE_DEVICES=3

# Run inference to generate captions.
# bazel-bin/A5/val \
bazel-bin/im2txt/val_TFRecords \
  --input_file_pattern="${MSCOCO_DIR}/val-?????-of-00004" \
  --checkpoint_path=${CHECKPOINT_PATH} \
  --vocab_file=${VOCAB_FILE} \
  --input_files=${IMAGE_FILE} \
  --output_files=${OUTPUT_FILE}
