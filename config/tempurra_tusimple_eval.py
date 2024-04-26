from tempurranet.util.clogging import GeneralLogger

logger = GeneralLogger("TempurraNET")

net = dict(type='Detector',
           classes=2,
           bilinear=True)

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet34',
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
)

num_points = 72
max_lanes = 5
sample_y = range(710, 150, -10)

heads = dict(type='CLRHead',
             num_priors=192,
             refine_layers=3,
             fc_hidden_dim=64,
             sample_points=36)

iou_loss_weight = 2.
cls_loss_weight = 6.
xyt_loss_weight = 0.5
seg_loss_weight = 1.0

work_dirs = "work_dirs/clr/r34_tusimple"

neck = dict(type='FPN',
            in_channels=[128, 256, 512],
            out_channels=64,
            num_outs=3,
            attention=False)

test_parameters = dict(conf_threshold=0.40, nms_thres=50, nms_topk=max_lanes)

epochs = 30
batch_size = 1

optimizer = dict(type='AdamW', lr=0.8e-3)  # 3e-4 for batchsize 8
total_iter = (3616 // batch_size + 1) * epochs
scheduler = dict(type='CosineAnnealingLR', T_max=total_iter)

eval_ep = 3
save_ep = epochs

img_norm = dict(mean=[103.939, 116.779, 123.68], std=[1., 1., 1.])

ori_img_w = 1280
ori_img_h = 720

img_h = 128
img_w = 256

# img_h = 360
# img_w = 720

cut_height = 160

train_process = [
    dict(
        type='GenerateLaneLine',
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w), name='Resize'),
                 p=1.0),
            # dict(name='HorizontalFlip', parameters=dict(p=1.0, name="HorizontalFlip"), p=0.5),
            dict(name='ChannelShuffle', parameters=dict(p=1.0, name="ChannelShuffle"), p=0.1),
            dict(name='MultiplyAndAddToBrightness',
                 parameters=dict(mul=(0.85, 1.15), add=(-10, 10), name="MultiplyAndAddToBrightness"),
                 p=0.6),
            dict(name='AddToHueAndSaturation',
                 parameters=dict(value=(-10, 10), name="AddToHueAndSaturation"),
                 p=0.7),
            dict(name='OneOf',
                 transforms=[
                     dict(name='MotionBlur', parameters=dict(k=(3, 5),name="MotionBlur")),
                     dict(name='MedianBlur', parameters=dict(k=(3, 5),name="MotionBlur"))
                 ],
                 p=0.2),
            dict(name='Affine',
                 parameters=dict(translate_percent=dict(x=(-0.1, 0.1),
                                                        y=(-0.1, 0.1)),
                                 rotate=(-10, 10),
                                 scale=(0.8, 1.2), name="Affine"),
                 p=0.7),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w), name="Resize"),
                 p=1.0),
        ],
    ),
    dict(type='ToTensor', keys=['img', 'prev_frames', 'lanes']),
]

val_process = [
    dict(
        type='GenerateLaneLine',
        transforms=[
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w), name='Resize'),
                 p=1.0),
            dict(name='Resize',
                 parameters=dict(size=dict(height=img_h, width=img_w), name="Resize"),
                 p=1.0),
        ],
    ),
    dict(type='ToTensor', keys=['img', 'prev_frames', 'lanes']),
]

dataset_path = '../tempurranet/data/TuSimple'
dataset_type = 'TuSimple'
test_json_file = 'data/tusimple/test_label.json'
dataset = dict(train=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='trainval',
    processes=train_process,
),
val=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='test',
    processes=val_process,
),
test=dict(
    type=dataset_type,
    data_root=dataset_path,
    split='test',
    processes=val_process,
))

workers = 10
log_interval = 100
# seed = 0
num_classes = 6 + 1
ignore_label = 255
bg_weight = 0.4
lr_update_by_epoch = False