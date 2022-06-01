import numpy as np

from mmcv.runner import auto_fp16
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.yolox import YOLOX

@DETECTORS.register_module()
class MyYOLOX(YOLOX):
    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 input_size=(640, 640),
                 size_multiplier=32,
                 random_size_range=(15, 25),
                 random_size_interval=10,
                 init_cfg=None,
                 img_metas=[{'scale_factor': np.array([0.6801275, 0.680776 , 0.6801275, 0.680776 ],
                                                      dtype=np.float32)}],
                 rescale=True):
        super(MyYOLOX, self).__init__(backbone=backbone,
                                      neck=neck,
                                      bbox_head=bbox_head,
                                      train_cfg=train_cfg,
                                      test_cfg=test_cfg,
                                      pretrained=pretrained,
                                      input_size=input_size,
                                      size_multiplier=size_multiplier,
                                      random_size_range=random_size_range,
                                      random_size_interval=random_size_interval,
                                      init_cfg=init_cfg)
        self._img_metas = img_metas
        self._rescale = rescale

    def forward_dummy(self, img):
        outs = super(MyYOLOX, self).forward_dummy(img=img)
        return outs