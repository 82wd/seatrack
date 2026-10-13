import math
from lib.models.seatrack import build_seatrack
from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
from lib.test.tracker.data_utils import PreprocessorMM
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
import matplotlib.pyplot as plt
import numpy as np
from lib.models.layers.attn import MergedLinear

class SEATrack(BaseTracker):
    def __init__(self, params, mode=None):
        super(SEATrack, self).__init__(params)
        network = build_seatrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        # 【LoRA 权重约定】RGB-T/D 当前加载流程假设 checkpoint 在验证后保存，已合并 LoRA。
        # 先标记 merged，再 train 拆分，最后 eval 重新合并；RGB-E 按未合并情况处理。
        # 换验证周期、保存策略或 checkpoint 来源时要核验该假设，不能仅看文件名判断。
        if self.params.task in ['rgbt', 'rgbd']:
            with torch.no_grad():
                for name, module in self.network.named_modules():
                    if isinstance(module, MergedLinear):  # 对于有val的数据集,lora 权重已经合并，因此先要进行分离
                        module.merged = True
                self.network.train()
        self.network.eval()
        self.preprocessor = PreprocessorMM()
        self.state = None
        self.mode = mode
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()
        # for debug
        if getattr(params, 'debug', None) is None:
            setattr(params, 'debug', 0)
        self.use_visdom = False #params.debug
        self.debug = params.debug
        self.frame_id = 0
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes

    def initialize(self, image, info: dict):
        # forward the template once
        # 【首帧初始化｜code.md §3】image 已包含RGB和X的6通道，按给定目标框一起裁剪。
        # 这里缓存的是预处理后的模板图像 tensor [1,6,128,128]，不是 backbone 特征。
        # 上面的旧注释“forward the template once”不能据此理解为只在首帧计算网络模板特征。
        z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        with torch.no_grad():
            self.z_tensor = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, dataset_name=None, save_name=None, seq_name=None, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        # 【当前帧】self.state 为上一帧原图 xywh；以它为中心取更大的搜索区域。
        # 预测偏移会影响后续裁剪位置，因此错误可能逐帧累积。
        # 默认搜索 factor=4、size=256；两模态一起裁剪，保持对应空间位置。
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            x_tensor = search
            # merge the template and the search
            # run the transformer
            # 每帧将固定模板图像和当前搜索图一起送入网络，重新计算联合注意力。
            # 测试不更新参数，因此这里用 no_grad；训练时不可照搬到可训练计算路径。
            out_dict = self.network.forward(
                template=self.z_tensor, search=x_tensor, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        # 【后处理】Hann窗口偏好搜索中心，可能改变最终赢家位置。
        # 重新调用 cal_bbox，因此最终跟踪结果可能不同于 out_dict['pred_boxes']。
        # 修改窗口也是改变跟踪算法的条件，评测对照时要记录。
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        # 归一化 cxcywh 乘 search_size/resize_factor=原图裁剪边长，恢复原图像素尺度。
        # 默认一个预测框，mean 不改变它；随后 map_box_back 平移并转换为左上角 xywh。
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        
        # for debug
        self.debug = 0
        if self.debug == 1:
            # (B=1, H=12, 320, 320) -> (H=12, 320, 320)
            rgb2_weight = out_dict['attns'][11][0].squeeze()[:, 64:, 64:].cpu().numpy().mean(axis=0).mean(axis=0).reshape(16, 16)
            tir2_weight = out_dict['attns'][11][2].squeeze()[:, 64:, 64:].cpu().numpy().mean(axis=0).mean(axis=0).reshape(16, 16)

            '''
            visualization of attention / cross-attention map
            '''
            # plt.imshow(rgb_weight, cmap='viridis', interpolation='nearest')
            # plt.axis('off')
            # # plt.savefig(r"C:\Users\Sue\Desktop\heatmap.png", bbox_inches='tight', pad_inches=0)
            # plt.show()

            '''
            visualization of overlay        
            '''
            rgb2_weight_resized = cv2.resize(cv2.normalize(rgb2_weight, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), (256, 256), interpolation=cv2.INTER_LINEAR_EXACT)
            rgb2_colored = cv2.applyColorMap(rgb2_weight_resized, cv2.COLORMAP_JET)
            rgb2_BGR = cv2.cvtColor(rgb2_colored, cv2.COLOR_RGB2BGR)

            tir2_weight_resized = cv2.resize(cv2.normalize(tir2_weight, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), (256, 256), interpolation=cv2.INTER_LINEAR_EXACT)
            tir2_colored = cv2.applyColorMap(tir2_weight_resized, cv2.COLORMAP_JET)
            tir2_BGR = cv2.cvtColor(tir2_colored, cv2.COLOR_RGB2BGR)    

            overlay_inputr2 = cv2.addWeighted(x_patch_arr[:,:,:3], 0.5, rgb2_BGR, 0.5, 0)
            if self.mode is not None:
                overlay_inputt2 = cv2.addWeighted(x_patch_arr[:,:,3:], 0.4, tir2_BGR, 0.6, 0)
            else:
                overlay_inputt2 = cv2.addWeighted(x_patch_arr[:,:,3:], 0.5, tir2_BGR, 0.5, 0)

            '''
            visualization of multiple images by subplots
            '''
            fig, axes = plt.subplots(1, 4, figsize=(12, 8)) # 创建一个1行3列的子图窗口
            # 设置窗口左上角标题栏文本
            fig.canvas.manager.set_window_title(f"Frame ID: {self.frame_id} - amglora")
            # column 0
            axes[0].imshow(x_patch_arr[:,:,:3])
            axes[0].axis('off')  # 移除坐标轴
            # column 1
            axes[1].imshow(x_patch_arr[:,:,3:])#, cmap='viridis', interpolation='nearest')
            axes[1].axis('off')
            # column 2
            axes[2].imshow(overlay_inputr2)
            axes[2].axis('off')
            # column 3
            axes[3].imshow(overlay_inputt2)#, cmap='viridis', interpolation='nearest')
            axes[3].axis('off')
            plt.tight_layout()  # 自动调整子图间的间距
            # print(self.frame_id)
            if self.frame_id>=40:
                plt.show()  
            # plt.show()
            plt.close()

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": max_score}
        else:
            return {"target_bbox": self.state,
                    "best_score": max_score}

    # 【code.md §9】输入已经乘过裁剪边长，是裁剪区域内的像素中心与宽高。
    # 原图中心 = 裁剪内中心 + 上帧框中心 - 裁剪边长/2；再减半宽高得到左上角。
    # 不要在这里再乘一次 resize_factor；调用方最后 clip_box 并写回 self.state。
    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return SEATrack
