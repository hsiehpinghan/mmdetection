_base_ = ['../yolox/yolox_tiny_8x8_300e_coco.py']

# Model config
model = dict(bbox_head=dict(num_classes=14))
        
# Dataset config
dataset_type = 'CocoDataset'
data_root = '/tmp/back_id_card_detect_augmented'
#img_scale = (941, 567)
img_scale = (640, 640)

train_pipeline = [#dict(type='Mosaic',
                  #     img_scale=img_scale,
                  #     pad_val=114.0),
                  #dict(type='RandomAffine',
                  #     scaling_ratio_range=(0.1, 2),
                  #     border=(-img_scale[0] // 2,
                  #             -img_scale[1] // 2)),
                  #dict(type='MixUp',
                  #     img_scale=img_scale,
                  #     ratio_range=(0.8, 1.6),
                  #     pad_val=114.0),
                  dict(type='YOLOXHSVRandomAug'),
                  dict(type='RandomFlip',
                       flip_ratio=0.0),
                  # According to the official implementation, multi-scale
                  # training is not considered here but in the
                  # 'mmdet/models/detectors/yolox.py'.
                  dict(type='Resize',
                       img_scale=img_scale,
                       keep_ratio=True),
                  dict(type='Pad',
                       pad_to_square=True,
                       # If the image is three-channel, the pad value needs
                       # to be set separately for each channel.
                       pad_val=dict(img=(114.0, 114.0, 114.0))),
                  dict(type='FilterAnnotations',
                       min_gt_bbox_wh=(1, 1),
                       keep_empty=False),
                  dict(type='DefaultFormatBundle'),
                  dict(type='Collect',
                       keys=['img', 'gt_bboxes', 'gt_labels'])]

test_pipeline = [dict(type='LoadImageFromFile'),
                 dict(type='MultiScaleFlipAug',
                      img_scale=img_scale,
                      flip=False,
                      transforms=[dict(type='Resize',
                                       keep_ratio=True),
                                  dict(type='RandomFlip',
                                       flip_ratio=0.0),
                                  dict(type='Pad',
                                       pad_to_square=True,
                                       pad_val=dict(img=(114.0, 114.0, 114.0))),
                                  dict(type='DefaultFormatBundle'),
                                  dict(type='Collect',
                                       keys=['img'])])]

train_dataset = dict(type='MultiImageMixDataset',
                     dataset=dict(type=dataset_type,
                                  data_root=data_root,
                                  ann_file='coco_train.json',
                                  img_prefix='image/train',
                                  pipeline=[dict(type='LoadImageFromFile'),
                                            dict(type='LoadAnnotations',
                                                 with_bbox=True)],
                                  filter_empty_gt=False,
                                  classes=['address',
                                           'address-title',
                                           'barcode',
                                           'father',
                                           'father-title',
                                           'hometown',
                                           'hometown-title',
                                           'mother',
                                           'mother-title',
                                           'seq',
                                           'serve',
                                           'serve-title',
                                           'spouse',
                                           'spouse-title']),
                     pipeline=train_pipeline)

data = dict(samples_per_gpu=16,
            workers_per_gpu=16,
            persistent_workers=True,
            train=train_dataset,
            val=dict(type=dataset_type,
                     data_root=data_root,
                     ann_file='coco_val.json',
                     img_prefix='image/val',
                     pipeline=test_pipeline,
                     classes=['address',
                              'address-title',
                              'barcode',
                              'father',
                              'father-title',
                              'hometown',
                              'hometown-title',
                              'mother',
                              'mother-title',
                              'seq',
                              'serve',
                              'serve-title',
                              'spouse',
                              'spouse-title']),
            test=dict(type=dataset_type,
                      data_root=data_root,
                      ann_file=None,
                      img_prefix=None,
                      pipeline=test_pipeline,
                      classes=['address',
                               'address-title',
                               'barcode',
                               'father',
                               'father-title',
                               'hometown',
                               'hometown-title',
                               'mother',
                               'mother-title',
                               'seq',
                               'serve',
                               'serve-title',
                               'spouse',
                               'spouse-title']))

# Schedule config
## The original learning rate (LR) is set for 8-GPU training.
## We divide it by 8 since we only use one GPU.
optimizer = dict(lr=0.01/8)

## learning policy
max_epochs = 300
num_last_epochs = 15
lr_config = dict(_delete_=True,
                 policy='YOLOX',
                 warmup='exp',
                 by_epoch=False,
                 warmup_by_epoch=True,
                 warmup_ratio=1,
                 warmup_iters=5,  # 5 epoch
                 num_last_epochs=num_last_epochs,
                 min_lr_ratio=0.05)
runner = dict(type='EpochBasedRunner',
              max_epochs=max_epochs)

# hook
## CheckpointHook
checkpoint_config = dict(interval=1,
                         max_keep_ckpts=1)
## LoggerHooks
log_config = dict(interval=1,
                  hooks=[dict(type='TextLoggerHook')])

## EvalHook
## Change the evaluation metric since we use customized dataset.
#evaluation = dict(metric='mAP',
#                  interval=1)

evaluation = dict(save_best='auto',
                  # The evaluation interval is 'interval' when running epoch is
                  # less than ‘max_epochs - num_last_epochs’.
                  # The evaluation interval is 1 when running epoch is greater than
                  # or equal to ‘max_epochs - num_last_epochs’.
                  interval=1,
                  dynamic_intervals=[(max_epochs-num_last_epochs, 1)],
                  metric='bbox')