import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
from lib.models.layers.attn import Attention, LoRP, HMoE
from lib.models.seatrack.utils import combine_tokens, recover_tokens, token2feature, feature2token


def candidate_elimination(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_s = attn.shape[-1] - lens_t
    bs, hn, _, _ = attn.shape

    '''
    math.ceil() 是一个Python标准库中的函数
    返回大于或等于所传参数的最小整数,它将参数向上取整到最接近的整数,如果参数已经是整数，则返回该整数。
    '''
    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None

    '''
    (B, 12, 64, 256)
    取的是templates关于search region的attention权重
    '''
    attn_t = attn[:, :, :lens_t, lens_t:]

    if box_mask_z is not None:
        '''
        对于central token的情况：
        unsqueeze(1): (B, 64) -> (B, 1, 64)
        unsqueeze(-1): (B, 1, 64) -> (B, 1, 64, 1)
        expand:(B, 1, 64, 1) -> (B, 12, 64, 256)
        扩充仍然从最右侧dim3开始，(64,1)->(64,256)，然后是dim1，将(64,256)看作整体进行复制
        其结果是：生成了关于attn_t的掩码矩阵，对于每个头的central token行，其掩码全为true
        '''
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1]) # (B, 1 64, 1) -> (B, 12, 64, 256)
        # attn_t = attn_t[:, :, box_mask_z, :]
        '''
        进行bool掩码索引，只返回attn_t中对应位置在box_mask_z中值为true的元素
        返回每个头的central token关于search region的attention权重
        (B, 12, 1, 256)
        '''
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        '''
        对于mean操作，返回一个降维tensor，指定的dim被消除
        ***对于指定了dim的操作，其操作单位是指定dim右边dim构成的整体张量***的attention权重
        (B, 12, 1, 256)
        对于mean操作，返回一个降维tensor，指定的dim被消除
        ***对于指定了dim的操作，其操作单位是指定dim右边dim构成的整体张量***
        mean(dim=2):(1, 12, 1, 256) -> (1, 12, 256)
        mean(dim=1):(1, 12, 256) -> (1, 256)
        将所有头中central token的attention求均值作为central token的最终相似度
        '''
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

        # attn_t = [attn_t[i, :, box_mask_z[i, :], :] for i in range(attn_t.size(0))]
        # attn_t = [attn_t[i].mean(dim=1).mean(dim=0) for i in range(len(attn_t))]
        # attn_t = torch.stack(attn_t, dim=0)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    '''
    将searchs与central token的相似度按降序排列，返回排序结果sorted_attn及其原索引indices
    '''
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    '''
    分别按顺序记录保留的token索引和删除的token索引
    '''
    keep_index = global_index.gather(dim=1, index=topk_idx)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t]
    tokens_s = tokens[:, lens_t:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)
    '''
    top_idx -> (1, 180)
    unsqueeze(-1) -> (1, 180, 1)，索引由行向量变为列向量
    expand(B, -1, C) -> (1, 180, 768)=index.shanpe=output.shape，每行的元素都相同（同一个索引）
    output[i][j][k] = tokens[i][index[i][j][k]][k]
    通过gather来获取tokens中index对应的那些token值
    '''
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # inattentive_tokens = tokens_s.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))

    # compute the weighted combination of inattentive tokens
    # fused_token = non_topk_attn @ inattentive_tokens

    # concatenate these tokens
    # tokens_new = torch.cat([tokens_t, attentive_tokens, fused_token], dim=0)
    '''
    tokens_new就是要送到下一层encoder的新tokens
    '''
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    return tokens_new, keep_index, removed_index

class CEBlock_AP(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_ratio_search=1.0, layer=None, lora_layers=[], moe_layers=[], amglora_rank=None, hmoe_rank=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, layer=layer, lora_layers=lora_layers, amglora_rank=8)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.keep_ratio_search = keep_ratio_search
        self.layer = layer
        self.lora_layers = lora_layers
        self.moe_layers = moe_layers
        
        if layer in lora_layers:
            self.r2dte_scaling = nn.Parameter(torch.zeros(1) + 1)
            self.dte2r_scaling = nn.Parameter(torch.zeros(1) + 1)
        if layer in moe_layers:
            self.attn_moe = HMoE(dim, 4, 2, 4)
            self.ffn_moe = HMoE(dim, 8, 2, 4)

    def cal_qkv(self, x, layer=None):
        B, N, C = x.shape
        qkv = self.attn.qkv(x) 
        qkv = qkv.reshape(B, N, 3, self.attn.num_heads, C // self.attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        return (q @ k.transpose(-2, -1)) * self.attn.scale, v
    
    def amglora_attn(self, attn, v, shape, guidance=None, mode=None, cls_token=None, layer=None):
        B, N, C = shape
        if mode == 'r2dte':
            attn = attn + self.r2dte_scaling*(guidance - attn)

        elif mode == 'dte2r':
            attn = attn + self.dte2r_scaling*(guidance - attn)

        attn = attn.softmax(dim=-1)
        attn = self.attn.attn_drop(attn)
        output = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.attn.proj_drop(self.attn.proj(output)), attn

    def forward(self, x, global_index_template, global_search_idx, mask=None, ce_template_mask=None, keep_ratio_search=None):
        lens_t = global_index_template.shape[1]
        lens_x = global_search_idx[0].shape[1]

        # AMG-LoRA for Attention maps alignment
        if self.layer in self.lora_layers:
            brgb_attn, rgb_v = self.cal_qkv(self.norm1(x[0]), self.layer)
            bdte_attn, dte_v = self.cal_qkv(self.norm1(x[1]), self.layer)
            xrgb_attn, _ = self.amglora_attn(brgb_attn, rgb_v, x[0].shape, guidance=bdte_attn, mode='dte2r', layer=self.layer)
            xdte_attn, _ = self.amglora_attn(bdte_attn, dte_v, x[1].shape, guidance=brgb_attn, mode='r2dte', layer=self.layer)
        else:
            brgb_attn, rgb_v = self.cal_qkv(self.norm1(x[0]))
            bdte_attn, dte_v = self.cal_qkv(self.norm1(x[1]))
            xrgb_attn, _ = self.amglora_attn(brgb_attn, rgb_v, x[0].shape)
            xdte_attn, _ = self.amglora_attn(bdte_attn, dte_v, x[1].shape)

        x[0] = x[0] + self.drop_path(xrgb_attn) 
        x[1] = x[1] + self.drop_path(xdte_attn)

        # HMoE for cross template and search region fusion
        if self.layer in self.moe_layers:
            mix_z = self.attn_moe(torch.cat([x[0][:, :lens_t], x[1][:, :lens_t]], dim=1), mode = 'template')
            mix_x = self.attn_moe(torch.cat([x[0][:, lens_t:], x[1][:, lens_t:]], dim=1), mode = 'search')
            x[0] = x[0] + self.drop_path(torch.cat([mix_z[:, :lens_t], mix_x[:, :lens_x]], dim=1))
            x[1] = x[1] + self.drop_path(torch.cat([mix_z[:, lens_t:], mix_x[:, lens_x:]], dim=1))
        
        removed_rgbsearch_idx = None
        removed_dtesearch_idx = None

        if self.keep_ratio_search < 1 and (keep_ratio_search is None or keep_ratio_search < 1):
            keep_ratio_search = self.keep_ratio_search if keep_ratio_search is None else keep_ratio_search
            x[0], global_search_idx[0], removed_rgbsearch_idx = candidate_elimination(x[0], x[0], lens_t, keep_ratio_search, global_search_idx[0], ce_template_mask)
            x[1], global_search_idx[1], removed_dtesearch_idx = candidate_elimination(x[1], x[1], lens_t, keep_ratio_search, global_search_idx[1], ce_template_mask)
            lens_x = global_search_idx[0].shape[1]

        x[0] = x[0] + self.drop_path(self.mlp(self.norm2(x[0]))) 
        x[1] = x[1] + self.drop_path(self.mlp(self.norm2(x[1]))) 

        # HMoE for cross template and search region fusion
        if self.layer in self.moe_layers:
            mix_z = self.ffn_moe(torch.cat([x[0][:, :lens_t], x[1][:, :lens_t]], dim=1), mode = 'template')
            mix_x = self.ffn_moe(torch.cat([x[0][:, lens_t:], x[1][:, lens_t:]], dim=1), mode = 'search')
            x[0] = x[0] + self.drop_path(torch.cat([mix_z[:, :lens_t], mix_x[:, :lens_x]], dim=1))
            x[1] = x[1] + self.drop_path(torch.cat([mix_z[:, lens_t:], mix_x[:, lens_x:]], dim=1))

        return x, global_index_template, global_search_idx, [removed_rgbsearch_idx, removed_dtesearch_idx]

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
