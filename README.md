<div align="center">

# SEATrack 复现指南

**在 Linux GPU 服务器上从零复现 CVPR 2026 Oral —— *SEATrack: Simple, Efficient, and Adaptive Multimodal Tracker***

[![arXiv](https://img.shields.io/badge/arXiv-2604.12502-b31b1b)](https://arxiv.org/abs/2604.12502)
[![Python](https://img.shields.io/badge/Python-3.8-3776ab?logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76b900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Tasks](https://img.shields.io/badge/Tasks-RGB--T%20%7C%20RGB--D%20%7C%20RGB--E-blueviolet)](#复现目标)
[![Params](https://img.shields.io/badge/Trainable-0.6M-success)](#复现目标)

</div>

---

## 这是什么

本仓库是 [SEATrack](https://github.com/AutoLab-SAI-SJTU/SEATrack) 的**完整复现记录与操作手册**：
从装环境、下权重、摆数据集、改硬编码路径，到跑通 RGB-T / RGB-D / RGB-E 三路评测并算出论文指标。

- ✅ **命令都跑通过**：全部在真实服务器（`liziyan103`，7×GPU）上执行过，路径已写死，按顺序复制粘贴即可
- ✅ 每一步都配了 ✅ 预期输出 和 ❌ 失败怎么办
- ✅ 记录了实际踩到的坑：网络不通、家目录、LasHeR 帧名错位、LoRA checkpoint 反序列化等，见 [附录 A](#附录-a--本机liziyan103实机踩坑记录)
- ✅ 不需要编译任何 CUDA/C++ 算子（仓库里无 `setup.py`、无 `.cu`）

> 代码实现版权归 SEATrack 原作者所有（见 [`SEATrack-main/LICENSE`](./SEATrack-main/LICENSE)），
> 本仓库只包含**复现笔记**、路径配置与少量补丁脚本。

---

## 复现目标

跑出下面这组指标即为复现成功（论文表 1，允许 ±0.5 波动）：

| 基准 | 模态 | 指标 | 论文值 |
|------|------|------|--------|
| **LasHeR** | RGB-T | PR / NPR / SR | **71.6 / 67.5 / 57.3** |
| RGBT234 | RGB-T | MPR / MSR | 87.8 / 63.9 |
| DepthTrack | RGB-D | PR / RE / F | 62.9 / 63.5 / 63.2 |
| VOT-RGBD2022 | RGB-D | EAO / Acc / Rob | 73.6 / 82.1 / 88.4 |
| VisEvent | RGB-E | PR / SR | 77.1 / 60.3 |
| 效率 | — | 可学习参数 / FPS | 0.6M / 63.5（RTX 4090，约 1GB 显存） |

对比参考：ViPT（0.8M）LasHeR 65.1/52.5；SDSTrack（14.8M，20.8 FPS）66.5/53.1；XTrack（5.4M，10.3 FPS）69.1/55.7。

## 三条复现路线

| 路线 | 目标 | 需要下载 | 预计时间（不含数据集下载） |
|------|------|----------|--------------------------|
| **L1 最小验证** | 官方权重跑 LasHeR，验证 PR ≈ 71.6 | 代码 + `SEATrack_RGBT.pth.tar` + LasHeR 测试集 | 30 min 配置 + 1~2 h 评测 |
| **L2 完整评测** | 论文表 1 全部 5 个基准 | L1 + 另外 2 个权重 + 全部数据集（VisEvent 约 216GB） | + 半天到一天 |
| **L3 训练 + 评测** | 从 OSTrack 自训练再评测 | L2 + 2 个 OSTrack 预训练 + 全部训练集 | + 每任务约 1~2 天 |

## 仓库内容

| 路径 | 说明 |
|------|------|
| [`README.md`](./README.md) | 本文件：完整复现流程 |
| [`SEATrack-main/`](./SEATrack-main) | 官方代码（未修改，评测前需按 Step 6 改硬编码路径） |
| [`SEATrack_中文译文.md`](./SEATrack_中文译文.md) | 论文全文中文翻译（含表 1~8 全部实验数据） |
| `SEATrack： Simple, Efficient, and Adaptive Multimodal Tracker.pdf` | 论文原文 |
| `seatrack_extracted.txt` | 论文 PDF 抽取的纯文本 |

> 数据集、权重、训练产物默认不入库（见 [`.gitignore`](./.gitignore)），需要按本文 Step 4 / Step 5 自行下载。

---

## 目录

- [变量速查（全文统一，不要改）](#变量速查全文统一不要改)
- [Step 0 · 体检（先看清楚再动手）](#step-0--体检先看清楚再动手)
- [Step 1 · 放好代码](#step-1--放好代码)
- [Step 2 · 建 conda 环境](#step-2--建-conda-环境)
- [Step 3 · 生成路径配置（30 秒，别偷懒）](#step-3--生成路径配置30-秒别偷懒)
- [Step 4 · 下载权重并改名](#step-4--下载权重并改名)
- [Step 5 · 数据集](#step-5--数据集)
- [Step 6 · 改硬编码路径（一行 sed 搞定）](#step-6--改硬编码路径一行-sed-搞定)
- [Step 7 · 冒烟测试（先跑一个序列，1 分钟）](#step-7--冒烟测试先跑一个序列1-分钟)
- [Step 8 · 完整评测](#step-8--完整评测)
- [Step 9 · 自己训练（可选，L3）](#step-9--自己训练可选l3)
- [Step 10 · 算指标](#step-10--算指标)
- [附录 A · 本机（liziyan103）实机踩坑记录](#附录-a--本机liziyan103实机踩坑记录)
- [排错 FAQ](#排错-faq)
- [一键体检（任何时候怀疑哪步没做对就跑）](#一键体检任何时候怀疑哪步没做对就跑)
- [最短路线（L1：只跑官方权重 + LasHeR，约 3 小时）](#最短路线l1只跑官方权重--lasher约-3-小时)

---

# 复现流程

> 默认服务器：用户 `liziyan103`，家目录 `/nas/liziyan103`，项目根 `/nas/liziyan103/seatrack`。
> **路径已写死，直接复制粘贴执行即可**；如果你换机器，把 `/nas/liziyan103` 全局替换成你的家目录。

### 变量速查（全文统一，不要改）

| 变量 | 实际路径 | 说明 |
|------|----------|------|
| `$HOME` | `/nas/liziyan103` | 家目录 |
| `ROOT` | `/nas/liziyan103/seatrack` | 你的项目根（已存在） |
| `CODE` | `/nas/liziyan103/seatrack/SEATrack` | 代码仓 |
| `DATA` | `/nas/liziyan103/seatrack/datasets` | 数据集 |
| `DL` | `/nas/liziyan103/seatrack/downloads` | 下载缓存 |
| `CKPT` | `/nas/liziyan103/seatrack/SEATrack/models/checkpoints` | 权重最终位置 |

> ⚠️ 如果你已经把解压好的文件夹 `SEATrack-main` 传到了 `~/seatrack` 下，**不要重新 clone**，
> 只把下面的 `CODE` 改成 `/nas/liziyan103/seatrack/SEATrack-main` 即可，其余命令全部不变。

---

## Step 0 · 体检（先看清楚再动手）

```bash
# 0-1 确认你在家目录，确认 conda 能用
cd ~
pwd
ls                      # 应该看到 anaconda3  cuda-11.8  seatrack  zero-shot
source ~/anaconda3/etc/profile.d/conda.sh     # 关键点：这个服务器的 conda 通常没自动进 PATH
conda --version
```

✅ 期望：打印 `/nas/liziyan103`、看到 4 个目录、`conda 24.x.x`。
❌ 若 `conda: command not found`：确认 `ls ~/anaconda3/bin/conda` 存在；不存在就先装 anaconda，或换成 `source ~/miniconda3/etc/profile.d/conda.sh`。

```bash
# 0-2 看 GPU（决定后面所有 CUDA_VISIBLE_DEVICES 怎么填）
nvidia-smi
```

请把输出记住：**GPU 编号、数量、每张卡显存、右上角 CUDA Version（驱动支持的最高 CUDA）**。

```bash
# 0-3 看磁盘还剩多少（全量数据集约 420GB）
df -h ~
du -sh ~/seatrack 2>/dev/null

# 0-4 看 CPU 核数（评测默认 30 进程）
nproc

# 0-5 看能不能连外网（决定走 XX 官方 bulk 下载还是用本子传）
curl -sI https://huggingface.co --max-time 8 | head -1
curl -sI https://github.com   --max-time 8 | head -1
```

✅ `HTTP/2 200` 表示 `curl` 能连（**只代表 curl**，不代表 Python）。

> ⚠️ **本机实测陷阱**：这台机器 `curl` 返回 200，但 Python 访问 HuggingFace 报
> `[Errno 101] Network is unreachable`。所以还要补一个 Python 侧的探测：
>
> ```bash
> python - <<'PY'
> from huggingface_hub import HfApi
> try:
>     fs = HfApi().list_repo_files('jbs99/SEATrack', repo_type='model')
>     print("HF 可达，文件数:", len(fs))
> except Exception as e:
>     print("HF 不通（走 Mac 下载+scp）:", type(e).__name__)
> PY
> ```
>
> 打印 `HF 不通` → 所有 HF / Google Drive 文件都走「Mac 下载 + `scp`」（见附录 A-2）。

> **决策点** — 根据 `nvidia-smi` 右上角的 CUDA Version：
> - ≥ 12.1 → 直接用仓库的 `environment.yaml`（内置 CUDA 12.1），走 Step 2 的 A 方案。
> - = 11.8 或更低 → 驱动跑不了 cu121，**走 Step 2 的 B 方案**（pytorch-cuda=11.8）。

---

## Step 1 · 放好代码

```bash
cd /nas/liziyan103/seatrack
pwd
ls -a
```

**情况 A：目录是空的 → clone**

```bash
cd /nas/liziyan103/seatrack
git clone https://github.com/AutoLab-SAI-SJTU/SEATrack.git SEATrack
cd SEATrack && ls
```

**情况 B：已经把 `SEATrack-main` 传上来了 → 直接用它（跳过 clone）**

```bash
cd /nas/liziyan103/seatrack
ls
mv SEATrack-main SEATrack 2>/dev/null     # 统一名字，后面命令就不用改了
cd SEATrack && ls
```

**情况 C：从你的 Mac 传上去**（在本子的终端执行，不是服务器）

```bash
rsync -avP /Users/lzy/Desktop/GitHub/seatrack/SEATrack-main/ liziyan103@<服务器IP>:/nas/liziyan103/seatrack/SEATrack/
```

✅ 期望看到这些文件：

```text
assets  Depthtrack_workspace  experiments  lib  models  pretrained
RGBE_workspace  RGBT_workspace  tracking  VOT22RGBD_workspace
environment.yaml  eval_rgbd.sh  eval_rgbe.sh  eval_rgbt.sh  train.sh  README.md
```

❌ 缺文件就重新传，别往下走。

---

## Step 2 · 建 conda 环境

```bash
source ~/anaconda3/etc/profile.d/conda.sh
cd /nas/liziyan103/seatrack/SEATrack
```

#### 方案 A（推荐，驱动 ≥ 12.1）

```bash
# 解决 shared server 上 conda 依赖求解慢的问题
conda install -y -n base -c conda-forge mamba
mamba env create -f environment.yaml
```

> `mamba` 装不上或不好使，就用原命令（慢一些，可能 20~40 分钟）：
> `conda env create -f environment.yaml`

#### 方案 B（驱动只有 11.8）

```bash
# 先照旧建环境，但跳过 pytorch-cuda=12.1 那几个包
mamba create -y -n seatrack python=3.8.20 numpy=1.24.3 pandas=2.0.3 matplotlib=3.7.5 \
    scipy=1.10.1 opencv-python=4.11.0.86 pyyaml=6.0.2 tqdm=4.66.5 cython -c conda-forge
conda activate seatrack
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install timm==0.5.4 easydict jpeg4py lmdb setproctitle nvitop visdom tensorboard thop==0.1.1-2209072238 \
            vot-toolkit==0.5.3 vot-trax==3.0.3 gdown pycocotools numba==0.58.1 exifread ipdb
```

#### 激活并验证

```bash
conda activate seatrack
python - <<'PY'
import torch, importlib
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available(), "| nGPU:", torch.cuda.device_count())
for m in ["cv2","pandas","matplotlib","numpy","yaml","timm","cv2","thop"]:
    try: importlib.import_module(m); print("ok  ", m)
    except Exception as e: print("FAIL", m, e)
PY
```

✅ 期望：

```text
torch: 2.2.2
cuda available: True | nGPU: <你的卡数>
ok   cv2 / pandas / matplotlib / numpy / yaml / timm / thop
```

❌ `cuda available: False` → GPU 驱动和 PyTorch 的 CUDA 不匹配，换另一个方案重做。
❌ `ImportError: libGL.so.1` → `pip install opencv-python-headless` 或让管理员安装依赖：
`sudo apt install -y libgl1-mesa-glx libglib2.0-0`

---

## Step 3 · 生成路径配置（30 秒，别偷懒）

```bash
conda activate seatrack
cd /nas/liziyan103/seatrack/SEATrack

python tracking/create_default_local_file.py \
  --workspace_dir /nas/liziyan103/seatrack/SEATrack \
  --data_dir      /nas/liziyan103/seatrack/datasets \
  --save_dir      /nas/liziyan103/seatrack/SEATrack/models
```

> ⚠️ `--save_dir` **必须是 `.../SEATrack/models`**。写成 `./output` 会导致后面评测找不到权重
> （测试脚本硬编码读 `<save_dir>/checkpoints/<task>/SEATrack_epXXXX.pth.tar`，见 `lib/test/parameter/seatrack.py:27/32/37`）。

核对必须长这样：

```bash
grep -E "lasher_dir|depthtrack_dir|visevent_dir" lib/train/admin/local.py
grep -E "prj_dir|save_dir"                        lib/test/evaluation/local.py
```

✅ 期望输出：

```text
self.lasher_dir = '/nas/liziyan103/seatrack/datasets/lasher/trainingset'
self.depthtrack_dir = '/nas/liziyan103/seatrack/datasets/depthtrack/trainingset'
self.visevent_dir = '/nas/liziyan103/seatrack/datasets/visevent/trainingset'

settings.prj_dir = '/nas/liziyan103/seatrack/SEATrack'
settings.save_dir = '/nas/liziyan103/seatrack/SEATrack/models'
```

❌ 还是 `/data/...` 说明脚本没写成功 → 重跑上面的命令。

---

## Step 4 · 下载权重并改名

#### 4.1 官方 SEATrack 权重（评测必须）

> **本服务器实测：`curl https://huggingface.co` 返回 200，但 Python（`huggingface_hub` / `requests`）访问报
> `Network is unreachable`（Errno 101）。所以服务器上不能直接 `huggingface-cli download`，
> 一律走「Mac 下载 + `scp` 上传」。**（见附录 A-2）

**① 在【你的 Mac】下载**（已确认仓库真实文件名）：

```bash
export PATH="/Users/lzy/Library/Python/3.9/bin:$PATH"     # pip 装的脚本默认不进 PATH
mkdir -p ~/Desktop/seatrack_ckpt && cd ~/Desktop/seatrack_ckpt

# 先确认仓库里有什么（可选，13 个文件）
python3 - <<'PY'
from huggingface_hub import HfApi
for f in HfApi().list_repo_files('jbs99/SEATrack', repo_type='model'):
    print(f)
PY

# 下载三个权重（约 1.1 GB）
python3 - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["SEATrack_RGBT.pth.tar","SEATrack_RGBD.pth.tar","SEATrack_RGBE.pth.tar"]:
    print("OK", hf_hub_download("jbs99/SEATrack", f, local_dir="."))
PY
ls -lh
```

已确认的仓库文件清单（13 个）：

```text
.gitattributes  README.md
SEATrack_RGBT.pth.tar  SEATrack_RGBD.pth.tar  SEATrack_RGBE.pth.tar
SEATrack_LasHeR.zip  SEATrack_RGBT234.zip  SEATrack_DepthTrack.zip
SEATrack_VOT22.zip   SEATrack_VisEvent.zip          ← 官方 raw 结果，对照用，可选
seatrack-rgbt.log  seatrack-rgbd.log  seatrack-rgbe.log      ← 官方训练日志
```

**② 传到服务器（Mac 端）：**

```bash
scp ~/Desktop/seatrack_ckpt/SEATrack_RGB*.pth.tar \
    liziyan103@<服务器IP>:/nas/liziyan103/seatrack/downloads/hf/
```

**③ 服务器端改名放到「测试脚本唯一认的路径」：**

```bash
mkdir -p /nas/liziyan103/seatrack/SEATrack/models/checkpoints/{rgbt,rgbd,rgbe}

cp /nas/liziyan103/seatrack/downloads/hf/SEATrack_RGBT.pth.tar \
   /nas/liziyan103/seatrack/SEATrack/models/checkpoints/rgbt/SEATrack_ep0060.pth.tar

cp /nas/liziyan103/seatrack/downloads/hf/SEATrack_RGBD.pth.tar \
   /nas/liziyan103/seatrack/SEATrack/models/checkpoints/rgbd/SEATrack_ep0025.pth.tar

cp /nas/liziyan103/seatrack/downloads/hf/SEATrack_RGBE.pth.tar \
   /nas/liziyan103/seatrack/SEATrack/models/checkpoints/rgbe/SEATrack_ep0045.pth.tar
```

校验：

```bash
ls -lh /nas/liziyan103/seatrack/SEATrack/models/checkpoints/*/*.pth.tar

cd /nas/liziyan103/seatrack/SEATrack          # ★ 必须！否则 torch.load 会反序列化失败
python - <<'PY'
import torch
ck = torch.load('/nas/liziyan103/seatrack/SEATrack/models/checkpoints/rgbt/SEATrack_ep0060.pth.tar', map_location='cpu')
print("keys:", list(ck.keys()))
print("参数量:", sum(p.numel() for p in ck['net'].values())/1e6, "M")
PY
```

✅ 期望：三个文件各约 361 MB；`keys` 里有 `'net'`；参数量约 90~100 M（骨干 + head）。

❌ 若报 `ModuleNotFoundError: No module named 'lib.train'`：
**不是权重坏了**，是 `torch.load` 反序列化时会 import checkpoint 里记录的 `lib.train...` 类。
必须在 `/nas/liziyan103/seatrack/SEATrack` 目录下执行（让 `lib` 包可导入）。
**绝对不要 `pip install lib.train`**（附录 A-3）。

#### 4.2 OSTrack 预训练（**只有自己训练才需要**，L1/L2 评测可跳过）

```bash
mkdir -p /nas/liziyan103/seatrack/downloads/ostrack
cd /nas/liziyan103/seatrack/downloads/ostrack
pip install -U gdown
gdown --folder "https://drive.google.com/drive/folders/1ttafo0O5S9DXK2PX0YqPvPrQ-HWJjhSy?usp=sharing"
ls -lh
```

放到指定位置（**文件名必须叫 `OSTrack_ep0300.pth.tar`**）：

```bash
mkdir -p /nas/liziyan103/seatrack/SEATrack/pretrained/vitb_256_mae_ce_32x4_ep300
mkdir -p /nas/liziyan103/seatrack/SEATrack/pretrained/vitb_256_mae_32x4_ep300

# 把 Google Drive 下来的 ce 版（文件名可能叫 vitb_256_mae_ce_32x4_ep300.pth.tar 之类）拷过去：
cp /nas/liziyan103/seatrack/downloads/ostrack/<ce版权重文件名> \
   /nas/liziyan103/seatrack/SEATrack/pretrained/vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar

cp /nas/liziyan103/seatrack/downloads/ostrack/<非ce版权重文件名> \
   /nas/liziyan103/seatrack/SEATrack/pretrained/vitb_256_mae_32x4_ep300/OSTrack_ep0300.pth.tar

ls -lh /nas/liziyan103/seatrack/SEATrack/pretrained/*/*.pth.tar
```

对应关系（yaml 写死了，**放错 = 训练报 shape 不匹配**）：

| 训练任务 | yaml 行 | 必须用哪个 |
|----------|---------|-----------|
| RGB-T | `experiments/seatrack/rgbt.yaml:35` | `vitb_256_mae_ce_32x4_ep300/`（**ce 版**） |
| RGB-E | `experiments/seatrack/rgbe.yaml:35` | `vitb_256_mae_ce_32x4_ep300/`（**ce 版**） |
| RGB-D | `experiments/seatrack/rgbd.yaml:35` | `vitb_256_mae_32x4_ep300/`（**非 ce 版**） |

---

## Step 5 · 数据集

先建目录：

```bash
mkdir -p /nas/liziyan103/seatrack/datasets
mkdir -p /nas/liziyan103/seatrack/downloads/{lasher,rgbt234,depthtrack,visevent}
```

> **先只做 5.1 的 LasHeR 测试集，跑通 Step 6~7 的评测，再回来补其余数据集。**
> 别一口气下 400GB，下完了才发现路径错。

### 5.1 LasHeR（RGB-T，先下这个）

来源（百度盘）：<https://pan.baidu.com/s/1hZgK_OMHNp0fN20SJNNm9w> 密码 `mmic`
备选：TeraBox <https://terabox.com/s/1GgKDG3wXVNYZiX97sUzJZQ> 密码 `yfi0`

服务器基本没法直连百度盘 → **在你 Mac 上下好再传**：

```bash
# 【Mac 端】
rsync -avP ./LasHeR/trainingset/ liziyan103@<服务器IP>:/nas/liziyan103/seatrack/datasets/lasher/trainingset/
rsync -avP ./LasHeR/testingset/  liziyan103@<服务器IP>:/nas/liziyan103/seatrack/datasets/lasher/testingset/
```

传完在服务器上整理成这个样子：

```text
/nas/liziyan103/seatrack/datasets/lasher/testingset/<序列名>/
├── visible/*.jpg        # 可见光帧
├── infrared/*.jpg       # 红外帧
├── visible.txt          # GT，逗号分隔 xywh
└── infrared.txt
```

核对：

```bash
ls /nas/liziyan103/seatrack/datasets/lasher/trainingset | wc -l   # 应为 975（≥881 也能训练）
ls /nas/liziyan103/seatrack/datasets/lasher/testingset  | wc -l   # 应为 245
ls /nas/liziyan103/seatrack/datasets/lasher/testingset | head -3
ls /nas/liziyan103/seatrack/datasets/lasher/testingset/$(ls /nas/liziyan103/seatrack/datasets/lasher/testingset | head -1)
du -sh /nas/liziyan103/seatrack/datasets/lasher
```

❌ train/test 数量反了或混在一起 → 现在是 trainingset/testingset,务必分开，评测脚本只认 `testingset`。

### 5.2 RGBT234（RGB-T 第二个测试集，234 段）

来源：<https://pan.baidu.com/share/init?surl=weaiBh0_yH2BQni5eTxHgg> 提取码 `qvsq`

```text
/nas/liziyan103/seatrack/datasets/rgbt234/<序列名>/{visible,infrared}/*.jpg + visible.txt + infrared.txt
```

```bash
ls /nas/liziyan103/seatrack/datasets/rgbt234 | wc -l    # 应为 234
```

### 5.3 DepthTrack（RGB-D，Zenodo 可以 wget）

```bash
cd /nas/liziyan103/seatrack/downloads/depthtrack
wget -c "https://zenodo.org/records/5792146/files/DepthTrack_Test.zip?download=1"      -O DepthTrack_Test.zip
wget -c "https://zenodo.org/records/5794115/files/DepthTrack_Train_100.zip?download=1" -O DepthTrack_Train_100.zip
wget -c "https://zenodo.org/records/5837926/files/DepthTrack_Train_52.zip?download=1"  -O DepthTrack_Train_52.zip

cd /nas/liziyan103/seatrack/datasets
mkdir -p depthtrack/trainingset depthtrack/testingset
unzip -q /nas/liziyan103/seatrack/downloads/depthtrack/DepthTrack_Test.zip      -d depthtrack/testingset
unzip -q /nas/liziyan103/seatrack/downloads/depthtrack/DepthTrack_Train_100.zip -d depthtrack/trainingset
unzip -q /nas/liziyan103/seatrack/downloads/depthtrack/DepthTrack_Train_52.zip  -d depthtrack/trainingset
```

> ⚠️ Zenodo 的 zip 可能自带一层外层目录（`-d` 后变成 `testingset/DepthTrack_Test/<序列>`）。务必把它压平，
> 最终必须直接是 `testingset/<序列名>/`：

```bash
cd /nas/liziyan103/seatrack/datasets/depthtrack
# 如果看到多了一层，用这条压平：
find . -maxdepth 3 -mindepth 2 -type d -name "DepthTrack*" -exec sh -c 'mv "$1"/* "$(dirname "$1")"/ && rmdir "$1"' _ {} \;
ls trainingset | head -3; ls testingset | head -3
```

结构要求（**帧名 8 位补零、从 1 开始**，硬要求，见 `lib/train/dataset/depthtrack.py:110`）：

```text
<序列名>/color/00000001.jpg
<序列名>/depth/00000001.png
<序列名>/groundtruth.txt
```

```bash
ls /nas/liziyan103/seatrack/datasets/depthtrack/trainingset | wc -l   # 146（训练清单为准）
ls /nas/liziyan103/seatrack/datasets/depthtrack/testingset  | wc -l   # 应为 50
```

#### VOT 评测还要再拷一份进代码仓

```bash
mkdir -p /nas/liziyan103/seatrack/SEATrack/Depthtrack_workspace/sequences
cp -a /nas/liziyan103/seatrack/datasets/depthtrack/testingset/. \
      /nas/liziyan103/seatrack/SEATrack/Depthtrack_workspace/sequences/

wc -l /nas/liziyan103/seatrack/SEATrack/Depthtrack_workspace/list.txt      # 必须为 50
ls   /nas/liziyan103/seatrack/SEATrack/Depthtrack_workspace/sequences | wc -l   # 必须为 50
# 两个必须一一对应（名字相同）
head -3 /nas/liziyan103/seatrack/SEATrack/Depthtrack_workspace/list.txt
```

❌ 数量对不上说明多层目录没压平，重回上面那条 `find` 命令。

### 5.4 VOT22-RGBD（127 段）

来源：<https://www.votchallenge.net/vot2022/dataset.html>（需要注册）

```bash
mkdir -p /nas/liziyan103/seatrack/SEATrack/VOT22RGBD_workspace/sequences
cp -a <你解压出来的 VOT22-RGBD 序列目录>/. \
      /nas/liziyan103/seatrack/SEATrack/VOT22RGBD_workspace/sequences/

wc -l /nas/liziyan103/seatrack/SEATrack/VOT22RGBD_workspace/list.txt          # 必须为 127
ls   /nas/liziyan103/seatrack/SEATrack/VOT22RGBD_workspace/sequences | wc -l  # 必须为 127
```

### 5.5 VisEvent（RGB-E，约 216GB，最后再下）

来源：Dropbox <https://www.dropbox.com/scl/fo/r406wsgll56fy0hhhwu62/AFo3cjXjSI4Dzjn5nlnXNW0?rlkey=ecgyd26j1ycfl1jbm4pwc3vbn>
或百度 <https://pan.baidu.com/s/1VhdORXT4OvG8TUESfDZHfw> 密码 `AHUE`

```text
/nas/liziyan103/seatrack/datasets/visevent/trainingset/<序列名>/{vis_imgs,event_imgs}/*.bmp
                                                       + groundtruth.txt + absent_label.txt
/nas/liziyan103/seatrack/datasets/visevent/testingset/<序列名>/...
/nas/liziyan103/seatrack/datasets/visevent/testingset/testlist.txt      # ★ 缺了评测直接崩
```

```bash
ls /nas/liziyan103/seatrack/datasets/visevent/trainingset | wc -l   # 500
wc -l /nas/liziyan103/seatrack/datasets/visevent/testingset/testlist.txt   # 320
```

❌ 没有 `testlist.txt` 就自己生成：

```bash
cd /nas/liziyan103/seatrack/datasets/visevent/testingset
ls -1 | grep -v '\.txt$' > testlist.txt
wc -l testlist.txt
```

---

## Step 6 · 改硬编码路径（一行 sed 搞定）

```bash
conda activate seatrack
cd /nas/liziyan103/seatrack/SEATrack

# 6-1 RGB-T 测试集路径
sed -i "s#seq_home = '/data/rgbt234'#seq_home = '/nas/liziyan103/seatrack/datasets/rgbt234'#" \
    RGBT_workspace/test_rgbt_mgpus.py
sed -i "s#seq_home = '/data/lasher/testingset'#seq_home = '/nas/liziyan103/seatrack/datasets/lasher/testingset'#" \
    RGBT_workspace/test_rgbt_mgpus.py

# 6-2 RGB-E 测试集路径
sed -i "s#seq_home = '/data/visevent/testingset'#seq_home = '/nas/liziyan103/seatrack/datasets/visevent/testingset'#" \
    RGBE_workspace/test_rgbe_mgpus.py

# 6-3 RGB-D 的 trax 路径
sed -i "s#paths = /data/seatrack/lib/test/vot#paths = /nas/liziyan103/seatrack/SEATrack/lib/test/vot#" \
    Depthtrack_workspace/trackers.ini
sed -i "s#paths = /data/seatrack/lib/test/vot#paths = /nas/liziyan103/seatrack/SEATrack/lib/test/vot#" \
    VOT22RGBD_workspace/trackers.ini

# 6-4 检查：下面应该【一条都不输出】
grep -rn "/data/" RGBT_workspace/test_rgbt_mgpus.py \
                  RGBE_workspace/test_rgbe_mgpus.py \
                  Depthtrack_workspace/trackers.ini \
                  VOT22RGBD_workspace/trackers.ini
```

查看改完之后的样子（确认指向你的目录）：

```bash
grep -n "seq_home" RGBT_workspace/test_rgbt_mgpus.py | head -5
grep -n "seq_home" RGBE_workspace/test_rgbe_mgpus.py
cat Depthtrack_workspace/trackers.ini
```

✅ 期望：`/data/` 全没了，`trackers.ini` 里是 `paths = /nas/liziyan103/seatrack/SEATrack/lib/test/vot`。

> ⚠️ 额外一行：`lib/test/vot/seatrack_baseline.py:7` 硬编码了 `os.environ['CUDA_VISIBLE_DEVICES'] = '1'`。
> **如果你只有 1 张卡（或想用 0 号卡）**，改成 `'0'`：

```bash
sed -i "s#os.environ\['CUDA_VISIBLE_DEVICES'\] = '1'#os.environ['CUDA_VISIBLE_DEVICES'] = '0'#" \
    lib/test/vot/seatrack_baseline.py
grep -n "CUDA_VISIBLE_DEVICES" lib/test/vot/seatrack_baseline.py
```

---

## Step 7 · 冒烟测试（先跑一个序列，1 分钟）

```bash
conda activate seatrack
cd /nas/liziyan103/seatrack/SEATrack
export CUDA_VISIBLE_DEVICES=0

SEQ=$(ls /nas/liziyan103/seatrack/datasets/lasher/testingset | head -1)
echo "测试序列：$SEQ"

python ./RGBT_workspace/test_rgbt_mgpus.py \
  --script_name seatrack --dataset_name LasHeR --yaml_name rgbt \
  --mode sequential --video "$SEQ"
```

✅ 期望：

- 打印 `test config: ...` → 说明 `prj_dir` 和 yaml 都对了
- 打印 `————————Process sequence: <SEQ>————————`
- 打印 `fps: 50~70`
- 生成 `RGBT_workspace/LasHeR/<SEQ>.txt`

```bash
head -3 "RGBT_workspace/LasHeR/$SEQ.txt"
wc -l   "RGBT_workspace/LasHeR/$SEQ.txt"
```

❌ `ModuleNotFoundError: No module named 'lib'` → 你没在 `/nas/liziyan103/seatrack/SEATrack` 目录下。
❌ `FileNotFoundError ... checkpoints/rgbt/SEATrack_ep0060.pth.tar` → Step 4.1 没做对，或 Step 3 的 `--save_dir` 填错，回去重做 Step 3。
❌ `FileNotFoundError ... experiments/seatrack/rgbt.yaml` → Step 3 的 `--workspace_dir` 填错。
❌ 结果文件行数远小于帧数 → 序列被 `&&` 截断或 GPU OOM，看终端报错。

---

## Step 8 · 完整评测

> 用 **tmux** 跑，断开 SSH 也不会停。

```bash
tmux new -s eval_t          # 进入 tmux 窗口后再执行下面的命令
# Ctrl-b 然后按 d  →  脱离 tmux（任务继续跑）
# tmux a -t eval_t  →  回来看
```

#### 8.1 RGB-T：LasHeR + RGBT234

```bash
source ~/anaconda3/etc/profile.d/conda.sh; conda activate seatrack
cd /nas/liziyan103/seatrack/SEATrack
export CUDA_VISIBLE_DEVICES=0,1       # ← 按你 Step 0 看到的卡号改

python ./RGBT_workspace/test_rgbt_mgpus.py \
  --script_name seatrack --dataset_name LasHeR --yaml_name rgbt \
  2>&1 | tee /nas/liziyan103/seatrack/eval_lasher.log
```

跑完后：

```bash
ls /nas/liziyan103/seatrack/SEATrack/RGBT_workspace/LasHeR | wc -l    # 应为 245
```

再跑 RGBT234（建议先把 `--threads` 降下来，避免多进程爆显存）：

```bash
python ./RGBT_workspace/test_rgbt_mgpus.py \
  --script_name seatrack --dataset_name RGBT234 --yaml_name rgbt --threads 8 \
  2>&1 | tee /nas/liziyan103/seatrack/eval_rgbt234.log

ls /nas/liziyan103/seatrack/SEATrack/RGBT_workspace/RGBT234 | wc -l   # 应为 234
```

> 想一次跑完就用仓库脚本，但 **先改里面的 `CUDA_VISIBLE_DEVICES`**（默认是 `0,1` 和 `0,2,3`，你可能没 4 张卡）：
>
> ```bash
> cd /nas/liziyan103/seatrack/SEATrack
> sed -i "s#CUDA_VISIBLE_DEVICES=0,2,3#CUDA_VISIBLE_DEVICES=0,1#" eval_rgbt.sh
> cat eval_rgbt.sh
> bash eval_rgbt.sh
> ```

**断点续跑**：评测脚本自带——结果文件已存在就跳过该序列（`test_rgbt_mgpus.py:77-79`），中断后直接重跑同一条命令即可。
**想重跑**：`rm -rf RGBT_workspace/LasHeR` 后再跑。

#### 8.2 RGB-E：VisEvent

```bash
conda activate seatrack
cd /nas/liziyan103/seatrack/SEATrack
export CUDA_VISIBLE_DEVICES=0,1

python ./RGBE_workspace/test_rgbe_mgpus.py \
  --script_name seatrack --yaml_name rgbe --threads 8 \
  2>&1 | tee /nas/liziyan103/seatrack/eval_visevent.log

ls /nas/liziyan103/seatrack/SEATrack/RGBE_workspace/VisEvent | wc -l   # 应为 320
```

> 注意格式差异：RGB-T 结果是**空格分隔**，RGB-E 结果是**逗号分隔**（`test_rgbe_mgpus.py:88`），别用同一个脚本读。

#### 8.3 RGB-D：DepthTrack + VOT22-RGBD

前置：`Step 5.3 / 5.4` 的 `sequences/` 已放好，`Step 6-3` 的 `trackers.ini` 已改。

```bash
conda activate seatrack
cd /nas/liziyan103/seatrack/SEATrack
which vot || echo "vot 没装：pip install vot-toolkit==0.5.3 vot-trax==3.0.3"

bash eval_rgbd.sh
```

结果在：

```text
/nas/liziyan103/seatrack/SEATrack/Depthtrack_workspace/results/
/nas/liziyan103/seatrack/SEATrack/Depthtrack_workspace/analysis/rgbd/*.html
/nas/liziyan103/seatrack/SEATrack/VOT22RGBD_workspace/analysis/rgbd/*.html
```

HTML 报告可以直接 `scp` 回本子打开：

```bash
# 【Mac 端】
scp liziyan103@<服务器IP>:/nas/liziyan103/seatrack/SEATrack/Depthtrack_workspace/analysis/rgbd/*.html ~/Desktop/
```

❌ VOT 起来就退出 → 大概率是 `trackers.ini` 的 `paths` 或 `seatrack_baseline.py:7` 的 GPU 号，回 Step 6。
❌ 结果和预期差很远 → 加 `--nocache`（`eval_rgbd.sh` 里有），并 `rm -rf <workspace>/results` 重跑。

#### 8.4 评测耗时参考（RTX 4090 × 2）

| 数据集 | 序列数 | 单卡/两卡大致耗时 |
|--------|--------|------------------|
| LasHeR | 245 | 1 ~ 2 h |
| RGBT234 | 234 | 1 ~ 2 h |
| VisEvent | 320 | 2 ~ 4 h |
| DepthTrack | 50 | 2 ~ 3 h |
| VOT22-RGBD | 127 | 3 ~ 5 h |

---

## Step 9 · 自己训练（可选，L3）

前置：Step 4.2 的 OSTrack 权重 + 对应训练集。

```bash
source ~/anaconda3/etc/profile.d/conda.sh; conda activate seatrack
cd /nas/liziyan103/seatrack/SEATrack       # ★ 必须：yaml 里 pretrain 是相对路径 ./pretrained/...

tmux new -s train_rgbt

# RGB-T（论文：2 卡 × batch32 = 全局 64，60 epoch）
CUDA_VISIBLE_DEVICES=0,1 python tracking/train.py \
  --script seatrack --config rgbt --save_dir ./models --mode multiple \
  2>&1 | tee /nas/liziyan103/seatrack/train_rgbt.log
```

另外两个任务（**记得换 tmux 窗口**）：

```bash
# RGB-D（25 epoch）
CUDA_VISIBLE_DEVICES=0,1 python tracking/train.py \
  --script seatrack --config rgbd --save_dir ./models --mode multiple \
  2>&1 | tee /nas/liziyan103/seatrack/train_rgbd.log

# RGB-E（45 epoch，LR 6e-5）
CUDA_VISIBLE_DEVICES=0,1 python tracking/train.py \
  --script seatrack --config rgbe --save_dir ./models --mode multiple \
  2>&1 | tee /nas/liziyan103/seatrack/train_rgbe.log
```

看日志：

```bash
tail -f /nas/liziyan103/seatrack/SEATrack/models/logs/seatrack-rgbt.log
```

Checkpoint 落在（`lib/train/trainers/base_trainer.py:48-51,135`）：

```text
/nas/liziyan103/seatrack/SEATrack/models/checkpoints/rgbt/SEATrack_ep0060.pth.tar
/nas/liziyan103/seatrack/SEATrack/models/checkpoints/rgbd/SEATrack_ep0025.pth.tar
/nas/liziyan103/seatrack/SEATrack/models/checkpoints/rgbe/SEATrack_ep0045.pth.tar
```

**这正好就是 Step 4.1 测试脚本读的路径，训完直接评测，不用拷贝。**

⚠️ 仓库自带的 `train.sh` 里 RGB-D / RGB-E 两行**没限定 `CUDA_VISIBLE_DEVICES`**（`train.sh:5,7`），
会把服务器上所有卡都用上 —— 别直接 `bash train.sh`，用上面三条命令。

#### ⚠️ 训练前必做：LasHeR 帧名陷阱

`lib/train/dataset/lasher.py:91-92` 是这样取帧的：

```python
pattern = '*{}.jpg'.format(frame_id)                 # frame_id 从 0 开始
rgb_frame_path = glob.glob(.../visible/pattern)[0]   # glob 无序，且会匹配多个！
```

对第 25 帧，`*25.jpg` 会同时匹配 `25.jpg`、`125.jpg`、`225.jpg`…，而 `glob[0]` 顺序不确定
→ **训练会悄悄读错帧，指标明显偏低但不报错**。先改成确定性写法：

```bash
cd /nas/liziyan103/seatrack/SEATrack
python - <<'PY'
import re
p = 'lib/train/dataset/lasher.py'
s = open(p).read()
old = """        pattern = '*{}.jpg'.format(frame_id)
        rgb_frame_path = glob.glob(os.path.join(seq_path, 'visible', pattern))[0]
        ir_frame_path = glob.glob(os.path.join(seq_path, 'infrared', pattern))[0]"""
new = """        rgb_frame_path = os.path.join(seq_path, 'visible',  '{:06d}.jpg'.format(frame_id))
        ir_frame_path  = os.path.join(seq_path, 'infrared', '{:06d}.jpg'.format(frame_id))"""
assert old in s, "没找到目标片段，检查 lasher.py 版本"
open(p,'w').write(s.replace(old, new))
print("patched OK")
PY
grep -n "rgb_frame_path" lib/train/dataset/lasher.py
```

然后把训练集帧统一重命名为 `000000.jpg` 起：

```bash
python - <<'PY'
import os, glob
ROOT = '/nas/liziyan103/seatrack/datasets/lasher/trainingset'
n = 0
for seq in sorted(os.listdir(ROOT)):
    for sub in ('visible', 'infrared'):
        d = os.path.join(ROOT, seq, sub)
        if not os.path.isdir(d): continue
        fs = sorted(glob.glob(os.path.join(d, '*.jpg')))
        tmp = os.path.join(d, '__tmp__'); os.makedirs(tmp, exist_ok=True)
        for i, f in enumerate(fs):
            os.rename(f, os.path.join(tmp, f'{i:06d}.jpg'))
        for f in glob.glob(os.path.join(tmp, '*.jpg')):
            os.rename(f, os.path.join(d, os.path.basename(f)))
        os.rmdir(tmp)
    n += 1
print("renamed", n, "sequences")
PY
```

还要确认序列名对得上清单（训练读的是 `lib/train/data_specs/lasher_train.txt`，**881 行**，不是列目录）：

```bash
wc -l /nas/liziyan103/seatrack/SEATrack/lib/train/data_specs/lasher_train.txt
head -3 /nas/liziyan103/seatrack/SEATrack/lib/train/data_specs/lasher_train.txt
```

❌ 名字对不上（最省事的办法，反过来按你的目录生成清单）：

```bash
cd /nas/liziyan103/seatrack/datasets/lasher/trainingset && \
  ls -1 > /nas/liziyan103/seatrack/SEATrack/lib/train/data_specs/lasher_train.txt
```

#### 训练超参（论文配置，yaml 已配好，别改）

| 任务 | EPOCH | LR | BATCH/卡 | 模板/搜索 | 预训练 | 数据集（序列数） |
|------|-------|----|----------|-----------|--------|------------------|
| RGB-T | 60 | 4e-4 | 32 | 128/256 | ce 版 | LasHeR_train（881） |
| RGB-D | 25 | 4e-4 | 32 | 128/256 | 非 ce 版 | DepthTrack_train（146） |
| RGB-E | 45 | 6e-5 | 32 | 128/256 | ce 版 | VisEvent（500） |

共通：AdamW、wd 1e-4、step scheduler（drop@48, ×0.1）、grad clip 0.1、`PEFT: True`（**只训 AMG-LoRA + HMoE，骨干冻结**）、`AMP: False`（不用混合精度）。

每 epoch ≈ 937 iter（60000/64）。RTX 4090 ×2 约 8~11 min/epoch ⇒ RGB-T 约 **8~11 h**，RGB-D 约 3~5 h，RGB-E 约 6~9 h。

---

## Step 10 · 算指标

| 数据集 | 用哪个工具 | 喂哪个目录 | 指标 |
|--------|-----------|-----------|------|
| LasHeR | [LasHeR Toolkit](https://github.com/BUGPLEASEOUT/LasHeR) | `SEATrack/RGBT_workspace/LasHeR/*.txt` | PR / NPR / SR |
| RGBT234 | [RGB-T Toolkit](https://github.com/xuboyue1999/RGBT-Tracking) | `SEATrack/RGBT_workspace/RGBT234/*.txt` | MPR / MSR |
| VisEvent | [VisEvent Benchmark](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark) | `SEATrack/RGBE_workspace/VisEvent/*.txt` | PR / SR |
| DepthTrack | VOT toolkit（Step 8.3 已产出） | `SEATrack/Depthtrack_workspace/analysis/` | PR / RE / F |
| VOT22-RGBD | VOT toolkit（同上） | `SEATrack/VOT22RGBD_workspace/analysis/` | EAO / Acc / Rob |

#### 目标：复现成功 = 达到这几个数

| 基准 | 指标 | 论文值 |
|------|------|--------|
| **LasHeR** | PR / NPR / SR | **71.6 / 67.5 / 57.3** |
| RGBT234 | MPR / MSR | 87.8 / 63.9 |
| DepthTrack | PR / RE / F | 62.9 / 63.5 / 63.2 |
| VOT-RGBD2022 | EAO / Acc / Rob | 73.6 / 82.1 / 88.4 |
| VisEvent | PR / SR | 77.1 / 60.3 |
| 效率 | 可学习参数 / FPS | 0.6M / 63.5（4090，约 1GB 显存） |

允许 ±0.5 波动（GPU/驱动版本差异）。

---

## 附录 A · 本机（liziyan103）实机踩坑记录

> 这一节记录这台服务器上**实际发生过**的问题，按顺序执行时大概率会再遇到，直接照做。

### A-1 家目录不是 `/home/liziyan103`，是 `/nas/liziyan103`

```text
$ cd /home/liziyan103/seatrack
-bash: cd: /home/liziyan103/seatrack: No such file or directory
```

`pwd` 显示家目录是 `/nas/liziyan103`。全文所有路径已改好，
**不要凭直觉写 `/home/...`**；不确定就一律用 `~`（如 `~/seatrack/SEATrack`）。

### A-2 服务器 `curl` 能通，但 Python 访问 HuggingFace 报 Network is unreachable

```text
$ curl -sI https://huggingface.co --max-time 8 | head -1
HTTP/2 200                          ← 通

$ python -c "from huggingface_hub import HfApi; ..."
ConnectionError ... [Errno 101] Network is unreachable   ← 不通
```

结论：**服务器上不要用 `huggingface-cli` / `pip` 下 HF 的东西**。
所有 HuggingFace / Google Drive 的文件一律「Mac 下载 → `scp`/`rsync` 上传」。
`wget` 类（`curl` 能通）一般没问题，DepthTrack 的 Zenodo 可以照常 `wget`。

排查命令（若哪天又要联网）：

```bash
env | grep -i proxy
cat ~/.curlrc 2>/dev/null
python -c "import socket;print(socket.getaddrinfo('huggingface.co',443))"
curl -4 -sI https://huggingface.co --max-time 8 | head -1
```

### A-3 `torch.load` 报 `ModuleNotFoundError: No module named 'lib.train'`

```text
ModuleNotFoundError: No module named 'lib.train'
```

**不是权重损坏**。checkpoint 是用 pickle 存的，反序列化时要 import 训练时记录的 `lib.train...` 类，
所以必须在仓库根目录执行（让 `lib` 包能被 import）：

```bash
cd /nas/liziyan103/seatrack/SEATrack     # ← 加这一行就解决
python - <<'PY'
import torch
ck = torch.load('models/checkpoints/rgbt/SEATrack_ep0060.pth.tar', map_location='cpu')
print(list(ck.keys()))
PY
```

❌ **不要** `pip install lib.train` —— PyPI 上没有这个包，它只是本仓库的本地包。

### A-4 Mac 上 `huggingface-cli: command not found`

新版 `huggingface_hub`（1.x）只装了 `hf` 这个脚本，且装在
`/Users/lzy/Library/Python/3.9/bin`（不在 PATH 里）。两种解法：

```bash
export PATH="/Users/lzy/Library/Python/3.9/bin:$PATH"
hf download jbs99/SEATrack SEATrack_RGBT.pth.tar --local-dir ~/Desktop/seatrack_ckpt
```

或直接用 Python 接口（**推荐，最稳**）：

```bash
python3 - <<'PY'
from huggingface_hub import hf_hub_download
print(hf_hub_download("jbs99/SEATrack", "SEATrack_RGBT.pth.tar", local_dir="~/Desktop/seatrack_ckpt"))
PY
```

### A-5 conda 每次 activate 都刷 `conda-libmamba-solver` 告警

```text
Error while loading conda entry point: conda-libmamba-solver
(module 'libmambapy' has no attribute 'QueryFormat')
```

无害，只是 mamba 装进 base 后 conda 的 solver 插件版本打架。消掉：

```bash
conda config --set solver classic
```

### A-6 本服务器硬件实测

| 项 | 实测值 |
|----|--------|
| 家目录 | `/nas/liziyan103` |
| GPU | **7 张**，`torch.cuda.device_count() == 7` |
| 驱动 CUDA Version | **13.0**（≥12.1，所以 `environment.yaml` 的 cu121 可直接用，无需方案 B） |
| conda | `~/anaconda3`，需 `source ~/anaconda3/etc/profile.d/conda.sh` |
| 网络 | `curl` / `wget` 通；Python 访问 HF 不通 |

> 共享机器：跑之前先 `nvidia-smi` 挑两张**空闲**卡，别抢别人在用的。

---

## 排错 FAQ

| 现象 | 原因 / 解决 |
|------|-------------|
| `conda: command not found` | 先 `source ~/anaconda3/etc/profile.d/conda.sh` |
| `ModuleNotFoundError: No module named 'lib'` | 不在 `/nas/liziyan103/seatrack/SEATrack` 目录下执行 |
| `FileNotFoundError: models/checkpoints/rgbt/SEATrack_ep0060.pth.tar` | Step 3 `--save_dir` 填错 或 Step 4.1 没改名。检查 `lib/test/evaluation/local.py` 的 `save_dir` |
| `FileNotFoundError: experiments/seatrack/rgbt.yaml` | Step 3 `--workspace_dir` 填错，`prj_dir` 不对 |
| 评测结果数量永远是 0 | Step 6 的 `seq_home` 没改，还在指向 `/data/...` |
| 训练时 `IndexError` 在 `lasher.py:92` | 帧名 / 清单问题 → Step 9 的两个「必做」修复 |
| 训练 loss 不降、结果明显低 | 90% 是上面那个 LasHeR 帧序错位 |
| `CUDA out of memory`（评测） | `--threads` 降到 4~8 |
| `CUDA out of memory`（训练） | yaml 是 `AMP: False`；降 `--nproc_per_node 1`（batch 32），或调小 `NUM_WORKER`（默认 10） |
| `vot: command not found` | `conda activate seatrack` 后重试；或 `pip install vot-toolkit==0.5.3 vot-trax==3.0.3` |
| VOT 起来就卡住/退出 | `trackers.ini` 的 `paths` 或 `seatrack_baseline.py:7` 的 GPU 号，见 Step 6 |
| VOT 结果总是不变 | 加 `--nocache`，并 `rm -rf <workspace>/results` |
| 单个评测进程不报错但没输出 | 异常被并行进程吞掉，用 Step 7 的单序列命令复现堆栈 |
| SSH 断掉任务就停 | 用 `tmux`（Step 8 开头） |
| 磁盘爆了 | 删 `~/seatrack/downloads`（只是缓存，解压后可删） |

---

## 一键体检（任何时候怀疑哪步没做对就跑）

```bash
source ~/anaconda3/etc/profile.d/conda.sh; conda activate seatrack
CODE=/nas/liziyan103/seatrack/SEATrack
DATA=/nas/liziyan103/seatrack/datasets

echo "== checkpoint =="
for p in rgbt/SEATrack_ep0060 rgbd/SEATrack_ep0025 rgbe/SEATrack_ep0045; do
  f=$CODE/models/checkpoints/$p.pth.tar
  [ -f "$f" ] && echo "OK   $f ($(du -h $f|cut -f1))" || echo "MISS $f"
done

echo "== 硬编码残留 =="
grep -rn "/data/" $CODE/RGBT_workspace/test_rgbt_mgpus.py \
                  $CODE/RGBE_workspace/test_rgbe_mgpus.py \
                  $CODE/Depthtrack_workspace/trackers.ini \
                  $CODE/VOT22RGBD_workspace/trackers.ini || echo "全部已改 OK"

echo "== local.py =="
grep -E "prj_dir|save_dir"  $CODE/lib/test/evaluation/local.py
grep -E "lasher_dir|depthtrack_dir|visevent_dir" $CODE/lib/train/admin/local.py

echo "== 数据集 =="
printf "lasher  train/test : %s / %s\n" "$(ls $DATA/lasher/trainingset 2>/dev/null|wc -l)" "$(ls $DATA/lasher/testingset 2>/dev/null|wc -l)"
printf "rgbt234            : %s\n"       "$(ls $DATA/rgbt234 2>/dev/null|wc -l)"
printf "depthtrack train/te: %s / %s\n"  "$(ls $DATA/depthtrack/trainingset 2>/dev/null|wc -l)" "$(ls $DATA/depthtrack/testingset 2>/dev/null|wc -l)"
printf "visevent train/test: %s / %s\n"  "$(ls $DATA/visevent/trainingset 2>/dev/null|wc -l)" "$(ls $DATA/visevent/testingset 2>/dev/null|wc -l)"
printf "DepthTrack WS seq  : %s\n"       "$(ls $CODE/Depthtrack_workspace/sequences 2>/dev/null|wc -l)"
printf "VOT22 WS seq       : %s\n"       "$(ls $CODE/VOT22RGBD_workspace/sequences 2>/dev/null|wc -l)"

echo "== GPU/Python =="
python -c "import torch;print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'| nGPU',torch.cuda.device_count())"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

---

## 最短路线（L1：只跑官方权重 + LasHeR，约 3 小时）

```bash
# 1
source ~/anaconda3/etc/profile.d/conda.sh; conda activate seatrack
cd /nas/liziyan103/seatrack/SEATrack
# 2
python tracking/create_default_local_file.py \
  --workspace_dir /nas/liziyan103/seatrack/SEATrack \
  --data_dir      /nas/liziyan103/seatrack/datasets \
  --save_dir      /nas/liziyan103/seatrack/SEATrack/models
# 3
mkdir -p models/checkpoints/rgbt
cp ~/seatrack/downloads/hf/SEATrack_RGBT.pth.tar models/checkpoints/rgbt/SEATrack_ep0060.pth.tar
# 4
sed -i "s#seq_home = '/data/lasher/testingset'#seq_home = '/nas/liziyan103/seatrack/datasets/lasher/testingset'#" \
    RGBT_workspace/test_rgbt_mgpus.py
# 5 冒烟
export CUDA_VISIBLE_DEVICES=0
python ./RGBT_workspace/test_rgbt_mgpus.py --script_name seatrack --dataset_name LasHeR \
  --yaml_name rgbt --mode sequential --video $(ls ~/seatrack/datasets/lasher/testingset | head -1)
# 6 全量（进 tmux）
tmux new -s eval_t
CUDA_VISIBLE_DEVICES=0,1 python ./RGBT_workspace/test_rgbt_mgpus.py \
  --script_name seatrack --dataset_name LasHeR --yaml_name rgbt --threads 8
# 7
ls RGBT_workspace/LasHeR | wc -l        # 应为 245
```

---

## 引用

```bibtex
@misc{su2026seatracksimpleefficientadaptive,
      title={SEATrack: Simple, Efficient, and Adaptive Multimodal Tracker},
      author={Junbin Su and Ziteng Xue and Shihui Zhang and Kun Chen and Weiming Hu and Zhipeng Zhang},
      year={2026},
      eprint={2604.12502},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.12502},
}
```

## 参考链接

- 官方代码：<https://github.com/AutoLab-SAI-SJTU/SEATrack>
- 官方权重（HF）：<https://huggingface.co/jbs99/SEATrack>
- OSTrack 预训练：<https://drive.google.com/drive/folders/1ttafo0O5S9DXK2PX0YqPvPrQ-HWJjhSy>
- LasHeR / RGBT234 / DepthTrack / VOT22-RGBD / VisEvent：见 [Step 5](#step-5--数据集)
- 评测工具：LasHeR Toolkit / RGB-T Toolkit / VisEvent Benchmark / [VOT Toolkit](https://github.com/votchallenge/toolkit)

## 说明

- SEATrack 的实现与模型版权归原作者所有，本仓库仅整理复现流程。
- 若你也在其他服务器上复现，欢迎把新遇到的坑补进 [附录 A](#附录-a--本机liziyan103实机踩坑记录)。
