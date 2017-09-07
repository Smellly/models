# Directory containing preprocessed MSCOCO data.
MSCOCO_DIR="${HOME}/projects/ic_models/im2txt-visual-concepts/im2txt/data/mscoco-visual-concepts"

# Inception v3 checkpoint file.
INCEPTION_CHECKPOINT="${HOME}/projects/ic_models/im2txt-visual-concepts/im2txt/data/inception_v3.ckpt"

# Directory to save the model.
MODEL_DIR="${HOME}/projects/ic_models/im2txt-visual-concepts/im2txt/model"

# Build the model.
cd ${HOME}/projects/ic_models/im2txt-visual-concepts
bazel build -c opt //im2txt/...

# Ignore GPU devices (only necessary if your GPU is currently memory
# constrained, for example, by running the training script).
export CUDA_VISIBLE_DEVICES=0

# Run the training script.
bazel-bin/im2txt/train \
  --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --number_of_steps=1000000

# You can further improve the performance of the model 
# by running a second training phase to jointly fine-tune 
# the parameters of the Inception v3 image submodel and the LSTM.
# Restart the training script with --train_inception=true.
# bazel-bin/im2txt/train \
#   --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
#   --train_dir="${MODEL_DIR}/train" \
#   --train_inception=true \
#   --number_of_steps=3000000  # Additional 2M steps (assuming 1M in initial training).

