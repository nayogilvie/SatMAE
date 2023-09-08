import torch
from torch import nn
import numpy as np
import transforms_seg as T_seg
import models_vit
from folder import SegmentationDataset
from einops import rearrange
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn

# Main segmentation model
class MaskTransformer(nn.Module):
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))
        masks = torch.nn.functional.interpolate(masks, size=(H, W), mode="bilinear")

        return masks

    def get_attention_map(self, x, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MergeSegmentor(nn.Module):
    def __init__(self, embed_dim = 512, num_heads = 4, n_cls = 5):
        super().__init__()
        self.infrared_branch = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,32,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512,1024,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024)
        )

        self.q_matrix = nn.Linear(1024,embed_dim)
        self.k_matrix = nn.Linear(1024,embed_dim)
        self.v_matrix = nn.Linear(1024,embed_dim)

        self.merger = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)

        self.segmentor = MaskTransformer(n_cls, 16, embed_dim, 4, num_heads, embed_dim, embed_dim,0.1,0.1)

    def forward(self, x, mae_features):
        x = self.infrared_branch(x)
        B,C,H,W = x.shape
        x = x.view(B,C,H*W)
        infrared_features = x.permute(0,2,1)

        query = self.q_matrix(mae_features)
        key = self.k_matrix(infrared_features)
        value = self.v_matrix(mae_features)

        merge_feature, merge_feature_weights = self.merger(query=query, key=key, value=value)
        mask = self.segmentor(merge_feature, (256,256))

        return mask


Batch_size = 8
img_size = 256
embed_dim = 512
num_heads = 4
n_cls = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on {device}")


model = models_vit.__dict__["vit_large_patch16"](
    patch_size=16, 
    img_size=img_size, 
    in_chans=3,
    num_classes=-1, 
    drop_path_rate=0.1, 
    global_pool=False,
    )

# dataset is here
# transform = T_seg.Compose([
#     T_seg.ToTensor()
# ])
# train_dataset = SegmentationDataset(root = "/users/n/o/nogilvie/scratch/pytorch_2/cdata_overlap/train", extentions = ("tif"), transforms=transform, size=img_size)
# data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)


checkpoint = torch.load("fmow_pretrain.pth", map_location='cpu')
checkpoint_model = checkpoint['model']
state_dict = model.state_dict()

interpolate_pos_embed(model, checkpoint_model)

msg = model.load_state_dict(checkpoint_model, strict=False)
print(msg)

model = model.to(device)
model = model.eval()

megSeg = MergeSegmentor(embed_dim, num_heads,n_cls=n_cls).to(device)

# a fake input
dummy_x = torch.randn(8,4,img_size,img_size).to(device)

# split the data into infrared and RGB
# I assume infrared is the first channel infrared+RGB
infrared = dummy_x[:, 0, :, :]
RGB = dummy_x[:, 1:, :, :]
infrared = infrared.unsqueeze(1)

# MAE extract raw feature from RGB
with torch.no_grad():
    mae_output = model(RGB)
mae_output_no_token = mae_output[:, 1:, :]

# segment model takes input of infrared data and raw feature 
# Then predict the segmentation map
seg_output = megSeg(infrared, mae_output_no_token)

print("seg model output shape:", seg_output.shape)


# TODO: 
# 1. use argmax to obtain the 1 hot encoding result from seg_output
# 2. finish the training loop with a loss function from Unet baseline
# 3. experiment with different parameters.