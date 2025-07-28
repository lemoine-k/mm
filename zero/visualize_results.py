import argparse
import os
import os.path as osp

import mmcv
from mmengine.config import Config
from mmengine.structures import PixelData

from mmseg.apis import init_model, inference_model, show_result_pyplot


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', default='cuda:1', help='Device used for inference')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    model = init_model(args.config, args.checkpoint, args.device)

    data_root = cfg.data_root
    test_img_prefix = cfg.test_dataloader.dataset.data_prefix.get('img_path', '')
    test_seg_prefix = cfg.test_dataloader.dataset.data_prefix.get('seg_map_path', '')
    test_img_dir = osp.join(data_root, test_img_prefix)
    test_seg_dir = osp.join(data_root, test_seg_prefix)

    for i in os.listdir(test_img_dir):
        img_path = osp.join(test_img_dir, i)
        result = inference_model(model, img_path)

        seg_path = osp.join(test_seg_dir, i)
        seg = mmcv.imread(seg_path, flag='unchanged')
        result.gt_sem_seg = PixelData(data=seg)

        # show_result_pyplot(model, img_path, result, show=True, draw_gt=False, with_labels=False)
        show_result_pyplot(model, img_path, result, show=True, draw_pred=False, with_labels=False)


if __name__ == '__main__':
    main()
