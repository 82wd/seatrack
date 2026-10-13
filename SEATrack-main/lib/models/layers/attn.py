import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, Mlp

from lib.models.layers.rpe import generate_2d_concatenated_self_attention_relative_positional_encoding_index
from timm.models.layers import Mlp, DropPath
import math
from typing import Optional, List

# Low-Rank Projection
# 【HMoE 的低秩投影｜code.md §7】这不是“冻结矩阵+旁路增量”的 LoRA，
# 而是独立的两层窄瓶颈映射，默认 768→4→768，带 bias 和训练态 dropout。
# 默认 if_act=False，所以中间 act 为 Identity，不要误读为必有 GELU。
class LoRP(nn.Module):
    def __init__(self, in_dim=768, hid_dim=4, out_dim=768, drp1=0.1, drp2=0.1, use_bias=True, if_act=False):
        super().__init__()
        self.lora_a = nn.Parameter(torch.empty(in_dim, hid_dim))
        self.lora_b = nn.Parameter(torch.empty(hid_dim, out_dim))
        self.use_bias = use_bias
        self.if_act = if_act
        if use_bias == True:
            self.bias_a = nn.Parameter(torch.empty(hid_dim))
            self.bias_b = nn.Parameter(torch.empty(out_dim))
        self.act = nn.GELU() if if_act else nn.Identity()
        for n, p in self.named_parameters():
            if 'lora' in n:
                nn.init.xavier_normal_(p, gain=math.sqrt(2))          
            else:
                bound = 1 / math.sqrt(in_dim) if 'a' in n else 1 / math.sqrt(hid_dim)
                nn.init.uniform_(p, -bound, bound)
            
        self.drop1 = nn.Dropout(drp1)
        self.drop2 = nn.Dropout(drp2)

    def forward(self, x):
        if self.use_bias:
            x = torch.matmul(x, self.lora_a) + self.bias_a.view(1, 1, -1)
            x = self.drop1(self.act(x))
            return self.drop2(torch.matmul(x, self.lora_b) + self.bias_b.view(1, 1, -1))
        else:
            return self.drop2(self.drop1(self.act(x @ self.lora_a)) @ self.lora_b)

# 【论文式(7)】专家并行计算，输入 [B,e,h,D/h]。
# 每个专家有独立 A_i、B_i，但该专家的 h 个槽位共享其 A_i、B_i。
# 默认 h=2、D/h=384、r=4：384→4→384，保持输入/输出形状。
# 不是 top-k 路由，所有专家均参与运算。
class MultiExpertLinear(nn.Module):
    def __init__(self, num_experts, in_dim, hid_dim, drp1, drp2, use_bias=True, if_act=False):
        super(MultiExpertLinear, self).__init__()
        self.lora_a = nn.Parameter(torch.empty(num_experts, in_dim, hid_dim))
        self.lora_b = nn.Parameter(torch.empty(num_experts, hid_dim, in_dim))
        self.drop1 = nn.Dropout(drp1)
        self.drop2 = nn.Dropout(drp2)
        self.act = nn.GELU() if if_act else nn.Identity()
        self.use_bias = use_bias
        if self.use_bias:
            self.bias_a = nn.Parameter(torch.empty(num_experts, hid_dim))
            self.bias_b = nn.Parameter(torch.empty(num_experts, in_dim))
        else:
            self.register_parameter("bias", None)

        for n, p in self.named_parameters():
            nn.init.xavier_normal_(p, gain=math.sqrt(2))             

    def forward(self, x):
        if self.use_bias:
            x = torch.matmul(x, self.lora_a) + self.bias_a.unsqueeze(1).unsqueeze(0) # (num_experts, output_dim) -> (1, num_experts, output_dim) -> (1, num_experts, 1, output_dim)
            x = torch.matmul(self.drop1(self.act(x)), self.lora_b) + self.bias_b.unsqueeze(1).unsqueeze(0) # (num_experts, output_dim) -> (1, num_experts, output_dim) -> (1, num_experts, 1, output_dim)
            return self.drop2(x)
        else:
            return self.drop2(self.drop1(self.act((x @ self.lora_a))) @ self.lora_b)

# 【图 3｜论文式(5)～(10)】先由所有子 token 汇聚到专家，再将专家结果分配回 token。
# 搜索调用 N=512（RGB256+X256），模板调用 N=128（RGB64+X64）。
# 本模块输入和输出均为 [B,N,D]；B=1 仅指一个样本，不代表只有一个 token。
# size_slots 对应论文 h，不是 Transformer 的 12 个注意力 head。
class HMoE(nn.Module):
    def __init__(self, dim, experts, slots, hid_dim, if_norm=True):
        super().__init__()
        self.size_slots = slots
        self.size_experts = experts
        thi = torch.empty(dim // slots, experts*slots)         
        std = math.sqrt(1.0 / thi.size(0))         
        thi = nn.init.normal_(thi, mean=0, std=std)
        self.norm = nn.LayerNorm(dim) if if_norm else nn.Identity()
        self.gate_thi = nn.Parameter(thi)
        # 【改骨干维度时注意】此调用未传 in_dim/out_dim，沿用 LoRP 的 768 默认值。
        # linear2 同样如此；将 backbone dim 改成其他值时不能只改 HMoE 的 dim。
        # D_temp/C_temp 是可学习温度，初始 1，当前没有正值约束；研究稳定性时应检查其值。
        self.linear1 = LoRP(hid_dim=hid_dim)
        # self.experts = nn.ModuleList([CAdapter(in_dim= dim // slots, out_dim=dim // slots, hid_dim=hid_dim) for _ in range(experts)])
        self.experts = MultiExpertLinear(experts, dim // slots, hid_dim, 0.1, 0.1, use_bias=True)
        self.D_temp = nn.Parameter(torch.zeros(1)+1.0)
        self.C_temp = nn.Parameter(torch.zeros(1)+1.0)
        self.linear2 = LoRP(hid_dim=hid_dim)

    def forward(self, x, mode=None):
        B, N, D = x.shape
        
        thi = self.gate_thi.unsqueeze(0).expand(B, -1, -1)
        
        # 【式(5)】LN+低秩映射不改变 [B,N,768]，reshape 将每个 token 的通道分成 h 份。
        # 搜索示例：[B,512,768]→[B,1024,384]，拆的是特征通道，不是再次裁图。
        # reshape 的连续顺序为 token0的两份、token1的两份……，后面 Combine 依赖这个顺序。
        x = self.linear1(self.norm(x)).reshape(B, N*self.size_slots, D//self.size_slots)

        # 【式(6)前半】gate_thi=Phi，shape=[D/h,e*h]，每列对应一个专家槽位。
        # 搜索、4专家示例：[B,1024,384] @ [B,384,8] → logits=[B,1024,8]。
        # Dispatch 沿输入子token维 softmax：每个槽位从全部位置收集的信息权重和为1。
        # 论文省略 batch 写 dim=0，代码带 batch 写 dim=1，含义一致。
        logits = torch.bmm(x, thi) # (B, N*slots, D//slots) x (B, D//slots, E*S) -> (B, N*slots, E*S)
        Dispatch = F.softmax(logits/self.D_temp, dim=1) # (B, N*slots, E*S)列向量是一个sequence的所有token关于一个slot的得分，所有token根据列向量的值做加权和，作为该slot的输入
        # 【式(9)】同一 logits 的另一条分支，不是对 Dispatch 再做 softmax。
        # 将 logits 看成 [B,N,h,e,h]，对两个 h 维求和，得到 token→expert 分数 [B,N,e]。
        # 这就是论文 F_p 的 h×h 非重叠块聚合；这里用 sum 而不是 mean。
        # 沿专家维 softmax 后，每个输出 token 读取各专家的权重和为1。
        Combine = logits.reshape(B, N, self.size_slots, self.size_slots*self.size_experts).sum(dim=2).reshape(B, N, self.size_experts, self.size_slots).sum(dim=-1) # (B, N, E)
        Combine = F.softmax(Combine/self.C_temp, dim=-1) # (B, N, E)行向量是一个token对一个expert的贡献度，所有expert的输出根据贡献度加权和得到该token的最终输出

        # 【式(6)后半】Dispatch^T @ x 汇聚全局信息：
        # [B,8,1024] @ [B,1024,384] → [B,8,384] → [B,4,2,384]。
        # 每个槽位拿到所有输入子 token 的加权和，已经包含两种模态、各空间位置的信息。
        experts_inputs = torch.bmm(Dispatch.transpose(1, 2), x).reshape(B, self.size_experts, self.size_slots, D//self.size_slots) # (B, E*S, D) -> (B, E, S, D)
        # experts_outputs = torch.stack([self.experts[i](experts_inputs[:, i]) for i in range(self.size_experts)], dim=1).reshape(B, self.size_experts, -1) # (B, E, D)
        # 【式(7)(8)】先独立专家变换，再把每专家的两个384维槽位拼回768维。
        # [B,4,2,384]→[B,4,768]；linear2 再做共享低秩映射，图3中的上方 Low-Rank Layer。
        # 论文式(8)简写为拼接，实际此处还有 linear2，读代码不能漏掉。
        experts_outputs = self.experts(experts_inputs).reshape(B,self.size_experts, -1)
        experts_outputs = self.linear2(experts_outputs)
        # 【式(10)】[B,N,e] @ [B,e,D] → [B,N,D]，回到每个原始 token 的位置。
        # 各 token 的 Combine 不同，输出也不同；调用方再拆回 RGB/X 并加残差。
        # 与 attention 的 N×N 不同，这里的路由围绕固定数量的专家槽位进行。
        moe_out = torch.bmm(Combine, experts_outputs) # (B, N, D)
        return moe_out


class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

# 【论文式(1)｜K/V LoRA】PyTorch weight 存成 [out,in]，与论文 HW 写法转置。
# 当前 Q/K/V 联合投影 out=2304，仅 K/V 启用；r=8 时 A=[16,768]、B=[1536,8]。
# 两组低秩矩阵分别服务 K 与 V，最后为 Q 补零形成完整投影增量。
class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features))) # (r*n, dim)
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r)) # ()
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        # 初始化 A 为 Xavier、B 为零，因此初始增量 BA=0，保留预训练投影输出。
        # 不能仅因论文笼统写 Xavier 就认为这里 A/B 初始均为随机非零。
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            # nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.xavier_normal_(self.lora_A, gain=math.sqrt(2))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    # 【实现技巧】conv1d 的 groups 对应启用的 K/V 两组；此处不是图像空间卷积。
    # 计算各组 B_i A_i 得到 [1536,768]，zero_pad 插入 Q 的全零增量，成为 [2304,768]。
    # 这里 self.scaling=alpha/r 的乘法在 forward/train 中被注释，实际没有应用该缩放。
    def merge_AB(self, mode=None):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        '''
        input (batch, in_channels, seq_len)
        weight (total_out_channels, filter_channels_per_group=in_channels // groups, kernel_size)
        groups
        requires：
        input.shape[1] % groups == 0 && weight.shape[0] == weight.shape[1] * groups
        '''
        if mode is not None:
            with torch.no_grad():
                delta_w = F.conv1d(
                                    self.lora_A.unsqueeze(0), # (1, 16, 768)
                                    self.lora_B.unsqueeze(-1), # (1536, 8, 1)
                                    groups=sum(self.enable_lora) # 2
                                ).squeeze(0)
        else:
            delta_w = F.conv1d(
                                    self.lora_A.unsqueeze(0), # (1, 16, 768)
                                    self.lora_B.unsqueeze(-1), # (1536, 8, 1)
                                    groups=sum(self.enable_lora) # 2
                                ).squeeze(0)
        return T(self.zero_pad(delta_w))

    # 【保存/加载注意】eval() 最终会调用 train(False)，将 LoRA 增量加到基础 weight；
    # 再 train(True) 时减掉，以恢复训练态的“基础权重+单独增量”计算。
    # merged 是普通 Python 标记，不会自动写入 state_dict；checkpoint 必须明确合并状态。
    # 冻结 qkv.weight 在保存时发生数值变化，可能源于合并，不等于优化器更新了它。
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB(mode) #* self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB(mode) #* self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x @ T(self.merge_AB().T)) #* self.scaling
            return result

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=False, z_size=7, x_size=14, layer=None, lora_layers=[], amglora_rank=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        # 【Q/K/V 的开关顺序】False/True/True = Q 不加 LoRA，K 和 V 加。
        # scale 使用每头 head_dim=768/12=64 的平方根倒数；它是注意力缩放，
        # 不是 LoRA alpha/r，也不是 AMG 的方向引导系数，这三类 scaling 不要混淆。
        if layer in lora_layers:
            self.qkv = MergedLinear(dim, dim * 3, r=amglora_rank, lora_alpha=1, lora_dropout=0.1,
                                bias=qkv_bias, enable_lora=[False, True, True])
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rpe =rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) 
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) (B, H, N, D//H=64)
        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, N, N)
        unnorm_attn = attn

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'),) # mask (B, N) -> (B, 1, 1, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))

        if return_attention:
            return x, attn, unnorm_attn
        else:
            return x

class Attention_talking_head(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 rpe=True, z_size=7, x_size=14):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe = rpe
        if self.rpe:
            relative_position_index = \
                generate_2d_concatenated_self_attention_relative_positional_encoding_index([z_size, z_size],
                                                                                           [x_size, x_size])
            self.register_buffer("relative_position_index", relative_position_index)
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(torch.empty((num_heads,
                                                                          relative_position_index.max() + 1)))
            trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index].unsqueeze(0)
            attn += relative_position_bias

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2),
                                    float('-inf'),)

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x   