package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//adaboost/...",
    ],
)

py_binary(
    name = "build_mscoco_data",
    srcs = [
        "data/build_mscoco_data.py",
    ],
)

sh_binary(
    name = "preprocess_mscoco",
    srcs = ["data/preprocess_mscoco.sh"],
    data = [
        ":build_mscoco_data",
    ],
)

py_binary(
    name = "build_flickr30_data",
    srcs = [
        "data/build_flickr30_data.py",
    ],
)

sh_binary(
    name = "preprocess_flickr30",
    srcs = ["data/preprocess_flickr30.sh"],
    data = [
        ":build_flickr30_data",
    ],
)

py_library(
    name = "configuration",
    srcs = ["configuration.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "show_and_tell_model",
    srcs = ["show_and_tell_model.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//adaboost/ops:image_embedding",
        "//adaboost/ops:image_processing",
        "//adaboost/ops:inputs",
    ],
)

py_test(
    name = "show_and_tell_model_test",
    size = "large",
    srcs = ["show_and_tell_model_test.py"],
    deps = [
        ":configuration",
        ":show_and_tell_model",
    ],
)

py_library(
    name = "inference_wrapper",
    srcs = ["inference_wrapper.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":show_and_tell_model",
        "//adaboost/inference_utils:inference_wrapper_base",
    ],
)

py_binary(
    name = "evaluate",
    srcs = ["evaluate.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":configuration",
        ":show_and_tell_model",
    ],
)

py_binary(
    name = "run_inference",
    srcs = ["run_inference.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":configuration",
        ":inference_wrapper",
        "//adaboost/inference_utils:caption_generator",
        "//adaboost/inference_utils:vocabulary",
    ],
)

py_binary(
    name = "val",
    srcs = ["val.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":configuration",
        ":inference_wrapper",
        "//adaboost/inference_utils:caption_generator",
        "//adaboost/inference_utils:vocabulary",
    ],
)

py_binary(
    name = "mm_val",
    srcs = ["mm_val.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":configuration",
        ":inference_wrapper",
        "//adaboost/inference_utils:mm_caption_generator",
        "//adaboost/inference_utils:caption_generator",
        "//adaboost/inference_utils:vocabulary",
    ],
)

py_binary(
    name = "mm_val_mt",
    srcs = ["mm_val_mt.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":configuration",
        ":inference_wrapper",
        "//adaboost/inference_utils:mm_caption_generator",
        "//adaboost/inference_utils:caption_generator",
        "//adaboost/inference_utils:vocabulary",
    ],
)
