"""
ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices]
                                            [, required][, help][, metavar][, dest])
"""
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--learning_rate', type=float, default=1e-3, help='the base learning rate')
parser.add_argument('--gpus', type=str, default='0', help='(by id) the GPUs to use')
parser.add_argument('--model', type=str, choices=['v0', 'v1', 'v2'], default='v0', help='the model used')
FLAGS = parser.parse_args()

parser.print_help()
