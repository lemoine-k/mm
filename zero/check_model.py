import argparse

import torch
from mmengine import Config

from mmseg.apis import init_model

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('--config', help='train config file path', default='../configs/segformer/segformer_mit-b5_2xb4-20k_ds_dagm-512x512.py')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model = init_model(args.config)
    print(model.backbone.layers[-1][-1])


if __name__ == '__main__':
    main()