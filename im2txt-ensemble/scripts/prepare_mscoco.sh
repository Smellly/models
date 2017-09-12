# Location to save the MSCOCO data.
MSCOCO_DIR="${HOME}/projects/ic_models/im2txt-visual-concepts/im2txt/data/mscoco-visual-concepts"
echo $MSCOCO_DIR

# Build the preprocessing script.
cd ${HOME}/projects/ic_models/im2txt-visual-concepts/
bazel build //im2txt:preprocess_mscoco

# Run the preprocessing script.
bazel-bin/im2txt/preprocess_mscoco $MSCOCO_DIR

