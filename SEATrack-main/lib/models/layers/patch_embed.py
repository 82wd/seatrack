import torch.nn as nn

from timm.models.layers import to_2tuple


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # 【图 2｜图片→token】kernel=stride=16，用一个卷积同时映射所有不重叠图块。
        # 每块 3×16×16 像素相当于先展平，再乘投影矩阵并加偏置。
        # 输出 768 维是特征维度；恰好等于该图块的像素数，并不表示保留原像素。
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # allow different input size
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        # 搜索例子：[B,3,256,256] → [B,768,16,16]。
        # flatten(2) 合并两个空间维，transpose 后变 [B,256,768]；token 顺序保持网格展开顺序。
        # 模板同理，只是最后为 [B,64,768]。后面恢复二维特征图依赖这个顺序。
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
