# Directory containing preprocessed MSCOCO data.
MSCOCO_DIR="${HOME}/projects/ic_models/im2txt/im2txt/data/mscoco"

# Inception v4 checkpoint file.
# INCEPTION_CHECKPOINT="${HOME}/projects/ic_models/im2txt/im2txt/data/inception/inception_v3.ckpt"
# INCEPTION_CHECKPOINT="${HOME}/projects/ic_models/im2txt/im2txt/data/inception/inception_v4.ckpt"
INCEPTION_CHECKPOINT="${HOME}/projects/ic_models/im2txt/im2txt/data/resnet152/resnet_v2_152.ckpt"

# Directory to save the model.
# MODEL_DIR="${HOME}/projects/ic_models/im2txt/im2txt/model"
# MODEL_DIR="${HOME}/projects/ic_models/im2txt/inceptionv4/model"
MODEL_DIR="${HOME}/projects/ic_models/im2txt/resnet152/model"

# Build the model.
cd ${HOME}/projects/ic_models/im2txt
# bazel build -c opt //im2txt/...
# bazel build -c opt //inceptionv4/...
# bazel build -c opt //resnet152/...

# Ignore GPU devices (only necessary if your GPU is currently memory
# constrained, for example, by running the training script).
export CUDA_VISIBLE_DEVICES=0

# Run the training script.
# bazel-bin/inceptionv4/train \
# bazel-bin/im2txt/train \
bazel-bin/resnet152/train \
  --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --number_of_steps=1000000

# You can further improve the performance of the model 
# by running a second training phase to jointly fine-tune 
# the parameters of the Inception v4 image submodel and the LSTM.
# Restart the training script with --train_inception=true.

# bazel-bin/im2txt/train \
#   --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
#   --train_dir="${MODEL_DIR}/train" \
#   --train_inception=true \
#   --number_of_steps=3000000  # Additional 2M steps (assuming 1M in initial training).

