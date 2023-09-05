import torch
import numpy as np
import timm

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_fmow_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit


model = models_vit.__dict__["vit_large_patch16"](
    patch_size=16, 
    img_size=224, 
    in_chans=3,
    num_classes=-1, 
    drop_path_rate=0.1, 
    global_pool=False,
    )


checkpoint = torch.load("fmow_pretrain.pth", map_location='cpu')
checkpoint_model = checkpoint['model']
state_dict = model.state_dict()


msg = model.load_state_dict(checkpoint_model, strict=False)
print(msg)


dummy_x = torch.randn(4,3,224,224)
y = model(dummy_x)
Y_no_token = y[:, 1:, :]

print(Y_no_token.shape)