# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class DAGMDataset(BaseSegDataset):
    """DS_DAGM dataset.

    In segmentation map annotation for DS_DAGM, 0 stands for background, which
    is included in 6 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=('background', 'scratch', 'texture', 'crush', 'color', 'dirty'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                     reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
