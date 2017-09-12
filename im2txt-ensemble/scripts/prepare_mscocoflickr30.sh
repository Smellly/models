# Build the preprocessing script.
cd ${HOME}/projects/ic_models/im2txt
bazel build //im2txt:preprocess_mscoco_blend_flickr30

# Run the preprocessing script.
bazel-bin/im2txt/preprocess_mscoco_blend_flickr30

