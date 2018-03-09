"""
The `dataset` dir contains the train/val.tfrecords files, along with scripts
* Generating the dataset from original .mat files
* Explaining how Examples are organized
"""

from dataset.example_spec import train_example_spec
from dataset.input_fn import input_fn, duo_input_fn


class INFO:
  # dataset size
  NUM_TRAIN = 17120
  NUM_EVAL = 592

  # projection info
  PRJ_WIDTH = 360
  PRJ_HEIGHT = 216
  PRJ_SPARSE_WIDTH = 72
  PRJ_SPARSE_NUM = 5
  PRJ_DEPTH = 1

  # image info
  IMG_WIDTH = 200
  IMG_HEIGHT = 200
  IMG_DEPTH = 1


__all__ = ['train_example_spec', 'input_fn', 'duo_input_fn', 'INFO']
