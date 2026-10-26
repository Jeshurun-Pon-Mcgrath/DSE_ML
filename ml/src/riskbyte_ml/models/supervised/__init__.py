"""Offline, validation-selected supervised experiments; no public prediction service."""

MODEL_VERSION = "supervised-classification-v1"
REGISTRY_VERSION = "model-registry-v1"
FROZEN_MANIFEST_HASH = "sha256:2668cf738689f229b9da566f20c3ad69918889dd946e72452f757f07146acd67"
FROZEN_SPLIT_HASH = "sha256:d42785e4bbfb267454131b45591c182981037a617aa4744af59c23ca7512e7ee"
FROZEN_PREPROCESSOR_HASHES = {
    "linear_distance": "sha256:6b84c7fee0ec728e561f7faea269d8f8d2cb5f32165289be68f6d9ab62914d99",
    "tree_boosting": "sha256:9d4b74baa210d08f98f714b469dbcd4bdeceedbd659983bb73c2947171ca801c",
}
FROZEN_RANDOM_ID = "sha256:af2f3bf533ba7516d199ec470ae56df9a58e8fdba21e3dc92ab4ba35cc351fa2"
FROZEN_RANDOM_HASH = "sha256:48f35f0acb183a0e9aa22e98cf3f905aa21a0658b0e778196a6dc585a9f93dfc"
