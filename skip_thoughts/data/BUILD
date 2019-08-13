package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "special_words",
    srcs = ["special_words.py"],
    srcs_version = "PY2AND3",
    deps = [],
)

py_binary(
    name = "preprocess_BookCorpus",
    srcs = [
        "preprocess_BookCorpus.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":special_words",
    ],
)

py_binary(
    name = "preprocess_MSVD",
    srcs = [
        "preprocess_MSVD.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":special_words",
    ],
)

py_binary(
    name = "preprocess_high_level",
    srcs = [
        "preprocess_high_level.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":special_words",
    ],
)

py_binary(
    name = "preprocess_MSR_VTT",
    srcs = [
        "preprocess_MSR_VTT.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":special_words",
    ],
)