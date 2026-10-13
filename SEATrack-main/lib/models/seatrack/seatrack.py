"""
Basic seatrack model.
"""
import math
import os
from typing import List
from timm.models.layers import to_2tuple
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from lib.models.layers.head import build_box_head
from lib.models.seatrack.vit_ci import vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh


class SEATrack(nn.Module):
    """ This is the base class for seatrackrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        # 【阅读入口｜code.md §4～8】这里是模型的一次 forward，不负责视频帧循环。
        # 默认 template=[B,6,128,128]、search=[B,6,256,256]；B 是样本组数，
        # 6 通道是 RGB 的 3 通道加第二模态的 3 通道，不是 6 张独立图片。
        # backbone 内部拆成两路，完成注意力与跨模态融合；默认返回 [B,256,768]。
        # 框坐标映射、Hann window 和上一帧状态在 lib/test/tracker/seatrack.py。
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # 【预测头输入】当前骨干已经只返回搜索 token，因此这里取末 256 个相当于保留全部。
        # 下面先构造 [B,1,768,256]，再还原为 [B,768,16,16]；Nq 默认等于 1。
        # 不要被上面的旧 docstring 误导：当前接口是 batch 在前的 [B,N,C]。
        # 若修改搜索分辨率，需要同步 token 数、位置编码和 box_head 的 feat_sz。
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            # 【论文 §3.4】CENTER 分支返回分数图 [B,1,16,16]、尺寸/偏移图 [B,2,16,16]。
            # bbox 是搜索区域内归一化的 (cx,cy,w,h)，随后整理成 [B,1,4]。
            # 推理 Tracker 会对分数图加 Hann window 再解码，未直接采用这里的 bbox。
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4) #(1, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


# 【模型工厂】真正使用的骨干从 vit_ci.py 导入，而不是同目录的 vit_ce.py。
# training=True 时按 PRETRAIN_FILE 加载初始化权重；测试模型权重在 Tracker 中加载。
# “冻结哪些参数”由训练的 get_optimizer_scheduler 决定，不是由这个参数单独决定。
def build_seatrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')  # use pretrained OSTrack as initialization
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           search_size=to_2tuple(cfg.DATA.SEARCH.SIZE),
                                           template_size=to_2tuple(cfg.DATA.TEMPLATE.SIZE),
                                           new_patch_size=cfg.MODEL.BACKBONE.STRIDE,
                                           amglora_rank = cfg.MODEL.AMGLORA_RANK,
                                           hmoe_rank = cfg.MODEL.HMOE_RANK
                                           )
        hidden_dim = backbone.embed_dim

    else:
        raise NotImplementedError
    """For prompt no need, because we have OSTrack as initialization"""
    # backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = SEATrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        # import ipdb; ipdb.set_trace()
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print(f"missing_keys: {missing_keys}")
        print(f"unexpected_keys: {unexpected_keys}")

    return model
