# PAFA + SPA：ICBHI 4-class / 2-class 完整训练与评测包

本代码包以 PAFA 官方仓库为基线，不修改其原始 `main.py`、ICBHI 数据预处理、BEATs 封装和 PAFA 损失实现。安装器会在 PAFA 克隆目录中增加独立的 SPA 模块、训练入口、基线评测入口、5-fold 脚本、official split 5-seed 脚本和结果汇总工具。

支持的实验如下：

| 方法 | 4-class | 2-class | patient-wise 5-fold | official 60/40 + 5 seeds |
|---|---:|---:|---:|---:|
| PAFA | 是 | 是 | 是 | 是 |
| PAFA + SPA | 是 | 是 | 是 | 是 |

4-class 标签为 `Normal / Crackle / Wheeze / Both`。2-class 标签为 `Normal / Abnormal`。

## 1. 目录说明

代码包本身是一个可安装 overlay：

```text
PAFA_SPA_ICBHI/
├── bootstrap.sh
├── install_overlay.py
├── environment.yml
├── requirements-spa.txt
├── README.md
├── UPSTREAM.md
└── overlay/
    ├── main_spa.py
    ├── eval_pafa_checkpoint.py
    ├── spa/
    │   ├── alignment.py
    │   ├── etf.py
    │   ├── head.py
    │   └── losses.py
    ├── training/
    │   ├── checkpoint.py
    │   ├── meters.py
    │   └── metrics.py
    ├── tools/
    │   ├── prepare_icbhi.py
    │   ├── verify_install.py
    │   ├── smoke_test_spa.py
    │   └── summarize_results.py
    ├── tests/
    │   └── test_spa.py
    └── scripts/
        ├── common.sh
        ├── train_pafa_4class_5fold.sh
        ├── train_pafa_2class_5fold.sh
        ├── train_pafa_spa_4class_5fold.sh
        ├── train_pafa_spa_2class_5fold.sh
        ├── eval_pafa_4class_5fold.sh
        ├── eval_pafa_2class_5fold.sh
        ├── eval_pafa_spa_4class_5fold.sh
        ├── eval_pafa_spa_2class_5fold.sh
        ├── train_pafa_4class_official_5seed.sh
        ├── train_pafa_2class_official_5seed.sh
        ├── train_pafa_spa_4class_official_5seed.sh
        ├── train_pafa_spa_2class_official_5seed.sh
        ├── run_all_5fold.sh
        └── summarize_5fold.sh
```

## 2. 安装 PAFA 和本代码包

### 2.1 新建完整工程

```bash
unzip PAFA_SPA_ICBHI.zip
cd PAFA_SPA_ICBHI
bash bootstrap.sh /home/USER/PAFA-SPA
cd /home/USER/PAFA-SPA
```

`bootstrap.sh` 会执行：

```bash
git clone https://github.com/wa976/pafa.git /home/USER/PAFA-SPA
python install_overlay.py --target /home/USER/PAFA-SPA
```

### 2.2 安装到已有 PAFA 克隆目录

```bash
cd PAFA_SPA_ICBHI
python install_overlay.py --target /home/USER/pafa
cd /home/USER/pafa
```

目标目录必须包含：

```text
main.py
method/pafa.py
util/icbhi_dataset.py
```

重新安装时，先检查已有同名文件，再使用：

```bash
python install_overlay.py --target /home/USER/pafa --force
```

## 3. Python 与 CUDA 环境

推荐使用论文实验环境对应的 Python 3.10、PyTorch 2.6 和 CUDA 12.4。机器驱动不支持 CUDA 12.4 时，应安装与本机驱动匹配的 PyTorch CUDA wheel。

### 3.1 Conda 环境

在代码包目录中执行：

```bash
conda env create -f environment.yml
conda activate pafa-spa
```

如果已经进入 PAFA-SPA 目录，可直接安装其余依赖：

```bash
pip install -r requirements-spa.txt
```

### 3.2 安装 PyTorch 2.6 CUDA 12.4

```bash
pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

检查环境：

```bash
python - <<'PY'
import torch
import torchaudio
print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

`cuda available` 必须为 `True`。PAFA 的 BEATs 配置按 GPU 训练设计，本包不将正式训练降级为 CPU。

PyTorch 2.6 将 `torch.load` 的 `weights_only` 默认值改为 `True`。PAFA 的 BEATs wrapper 未显式传入该参数，因此本包 shell 脚本设置：

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

这只应用于你已确认来源可信的 Microsoft BEATs 权重。手工运行命令时也应先设置该环境变量。

## 4. 下载和放置 BEATs 权重

PAFA 的 BEATs wrapper 固定读取：

```text
./pretrained_models/BEATs_iter3_plus_AS2M.pt
```

从 Microsoft UniLM BEATs 页面下载 `BEATs_iter3+ (AS2M)` 权重：

```text
https://github.com/microsoft/unilm/tree/master/beats
```

然后执行：

```bash
cd /home/USER/PAFA-SPA
mkdir -p pretrained_models
cp /path/to/BEATs_iter3_plus_AS2M.pt \
   pretrained_models/BEATs_iter3_plus_AS2M.pt
```

检查：

```bash
ls -lh pretrained_models/BEATs_iter3_plus_AS2M.pt
```

文件名必须与上面完全一致。

## 5. 下载和配置 ICBHI 2017

PAFA 官方 README 使用以下下载地址：

```bash
mkdir -p downloads
wget --no-check-certificate \
  https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip \
  -O downloads/ICBHI_final_database.zip
```

将数据整理为 PAFA 需要的目录：

```bash
python tools/prepare_icbhi.py \
  --archive downloads/ICBHI_final_database.zip \
  --repo_root .
```

工具会完成以下操作：

1. 解压 ICBHI zip；
2. 自动定位含 WAV 和标注 TXT 的目录；
3. 复制到 `data/icbhi_dataset/audio_test_data/`；
4. 从 PAFA 仓库复制 `metadata.txt`、`official_split.txt` 和 `patient_list_foldwise.txt`；
5. 删除可能导致不同任务或不同 fold 误读的旧 `data/training.pt` 和 `data/test.pt` 缓存；
6. 检查录音和标注文件数量。

整理后的关键结构：

```text
data/icbhi_dataset/
├── audio_test_data/
│   ├── 101_1b1_Al_sc_Meditron.wav
│   ├── 101_1b1_Al_sc_Meditron.txt
│   └── ...
├── metadata.txt
├── official_split.txt
└── patient_list_foldwise.txt
```

也可以使用已经解压的数据目录：

```bash
python tools/prepare_icbhi.py \
  --source_dir /path/to/ICBHI_final_database \
  --repo_root .
```

## 6. 安装检查和单元测试

先执行不依赖数据集和 GPU 的 SPA smoke test：

```bash
python tools/smoke_test_spa.py
python -m unittest tests/test_spa.py -v
```

再执行完整安装检查：

```bash
python tools/verify_install.py --repo_root .
```

检查项包括：

- PAFA 原始文件是否存在；
- SPA 新增文件是否存在；
- ICBHI 录音/标注是否完整；
- PAFA split 和 metadata 是否齐全；
- BEATs 权重是否位于固定路径；
- `torch` 和 `torchaudio` 是否可导入。

## 7. 重要说明：5-fold 与 official 5-seed 不是同一协议

PAFA 仓库支持两种划分：

1. `--test_fold 0/1/2/3/4`：依据 `patient_list_foldwise.txt` 进行 patient-wise 80/20 划分。依次运行 0 至 4 才是实际 5-fold cross-validation。
2. `--test_fold official`：ICBHI 官方 60/40 文件级划分。

SPA 论文在 ICBHI 主表中使用的是 official 60/40 split，并对 seeds `[1,2,3,4,5]` 报告均值和标准差。它不是 5-fold。由于本任务要求 5-fold，本代码包同时提供：

- 实际 patient-wise 5-fold；
- 与 SPA 论文主实验更接近的 official 60/40 + 5 seeds。

论文结果复现应使用 official 5-seed 脚本；交叉验证分析应使用 5-fold 脚本。两组结果不要混在同一个平均值中。

## 8. 运行 patient-wise 5-fold

所有脚本默认：

```text
GPU=0
SEED=1
EPOCHS=100
BATCH_SIZE=32
NUM_WORKERS=8
SAVE_ROOT=./experiments
```

可以通过环境变量覆盖，例如：

```bash
GPU=1 BATCH_SIZE=16 NUM_WORKERS=4 bash scripts/train_pafa_spa_4class_5fold.sh
```

### 8.1 4-class PAFA baseline

训练：

```bash
bash scripts/train_pafa_4class_5fold.sh
```

测试并补充 ECE、NLL 和 Brier：

```bash
bash scripts/eval_pafa_4class_5fold.sh
```

### 8.2 4-class PAFA + SPA

训练：

```bash
bash scripts/train_pafa_spa_4class_5fold.sh
```

测试：

```bash
bash scripts/eval_pafa_spa_4class_5fold.sh
```

### 8.3 2-class PAFA baseline

```bash
bash scripts/train_pafa_2class_5fold.sh
bash scripts/eval_pafa_2class_5fold.sh
```

### 8.4 2-class PAFA + SPA

```bash
bash scripts/train_pafa_spa_2class_5fold.sh
bash scripts/eval_pafa_spa_2class_5fold.sh
```

### 8.5 一次运行某个任务的全部 baseline 与 SPA 实验

4-class：

```bash
bash scripts/run_all_5fold.sh 4
```

2-class：

```bash
bash scripts/run_all_5fold.sh 2
```

该脚本按顺序执行 PAFA 训练、PAFA 测试、PAFA+SPA 训练、PAFA+SPA 测试和汇总。完整 5-fold 计算时间较长，建议在服务器上使用 `tmux` 或作业调度系统运行。

## 9. 运行 official 60/40 + 5 seeds

### 9.1 PAFA baseline

```bash
bash scripts/train_pafa_4class_official_5seed.sh
bash scripts/eval_pafa_4class_official_5seed.sh
bash scripts/train_pafa_2class_official_5seed.sh
bash scripts/eval_pafa_2class_official_5seed.sh
```

### 9.2 PAFA + SPA

```bash
bash scripts/train_pafa_spa_4class_official_5seed.sh
bash scripts/eval_pafa_spa_4class_official_5seed.sh
bash scripts/train_pafa_spa_2class_official_5seed.sh
bash scripts/eval_pafa_spa_2class_official_5seed.sh
```

## 10. 默认训练配置

PAFA 部分沿用官方 BEATs + PAFA 配置：

```text
epochs              100
batch size           32
optimizer            Adam
learning rate        5e-5
weight decay         1e-6
scheduler            cosine
warm-up              10 epochs
input duration       5 s
sample rate          16 kHz
padding              repeat
SpecAugment          disabled
PAFA PCSL weight     50
PAFA GPAL weight     0.0005
PAFA total weight    1.0
projection dim       768
projection norm      LayerNorm
moving average       enabled, beta=0.5
```

SPA 部分默认：

```text
prototype momentum m        0.99
geometric temperature tau   2.0
self-alignment weight       0.5
branch CE coefficient       1.0
fusion weight omega         0.5
spherical scale init        16.0
label smoothing             0.1
KD direction                geometric -> spherical
ECE bins                    15
```

对应代码分工：

- `spa/etf.py`：Simplex ETF；
- `spa/alignment.py`：prototype EMA 与 SVD Orthogonal Procrustes；
- `spa/head.py`：Spherical branch、Geometric branch 和概率融合；
- `spa/losses.py`：两分支分类损失和 self-alignment KL；
- `main_spa.py`：PAFA 与 SPA 联合训练；
- `eval_pafa_checkpoint.py`：不改变 baseline checkpoint 的独立概率评测。

SPA 的动态几何更新在每次 optimizer step 后执行：

```text
BEATs temporal features
    ├── PAFA ProjectionHead -> PCSL + GPAL
    └── temporal mean pooling
          ├── Spherical branch -> normalized cosine logits
          └── Geometric branch -> adapter -> rotated ETF logits

optimizer step
    -> update class prototypes by EMA
    -> solve rotation by SVD
    -> use the new rotation in the next mini-batch
```

为了避免训练初期某些类别尚未出现时对秩不足矩阵反复求解，本实现默认在所有类别至少出现一次后更新 rotation。可关闭该保护：

```bash
python main_spa.py ... --no-spa_update_after_all_classes
```

## 11. 单个 fold 的训练和测试命令

### 11.1 PAFA baseline，4-class，fold 0

```bash
python main.py \
  --dataset icbhi \
  --data_folder ./data \
  --class_split lungsound \
  --n_cls 4 \
  --test_fold 0 \
  --sample_rate 16000 \
  --desired_length 5 \
  --n_mels 128 \
  --pad_types repeat \
  --batch_size 32 \
  --num_workers 8 \
  --model beats \
  --method pafa \
  --epochs 100 \
  --optimizer adam \
  --learning_rate 5e-5 \
  --weight_decay 1e-6 \
  --cosine \
  --warm \
  --ma_update \
  --ma_beta 0.5 \
  --w_ce 1.0 \
  --w_pafa 1.0 \
  --lambda_pcsl 50 \
  --lambda_gpal 0.0005 \
  --norm_type ln \
  --output_dim 768 \
  --nospec \
  --seed 1 \
  --tag 4class_fold0_seed1 \
  --save_dir ./experiments/pafa
```

独立评测：

```bash
python eval_pafa_checkpoint.py \
  --checkpoint ./experiments/pafa/icbhi_beats_pafa_4class_fold0_seed1/best.pth \
  --output ./experiments/pafa/icbhi_beats_pafa_4class_fold0_seed1/metrics_eval.json \
  --data_folder ./data \
  --n_cls 4 \
  --test_fold 0 \
  --seed 1
```

### 11.2 PAFA + SPA，4-class，fold 0

```bash
python main_spa.py \
  --data_folder ./data \
  --n_cls 4 \
  --test_fold 0 \
  --seed 1 \
  --epochs 100 \
  --batch_size 32 \
  --num_workers 8 \
  --learning_rate 5e-5 \
  --weight_decay 1e-6 \
  --optimizer adam \
  --cosine \
  --warm \
  --ma_update \
  --ma_beta 0.5 \
  --desired_length 5 \
  --pad_types repeat \
  --nospec \
  --lambda_pcsl 50 \
  --lambda_gpal 0.0005 \
  --w_pafa 1.0 \
  --norm_type ln \
  --output_dim 768 \
  --spa_momentum 0.99 \
  --spa_temperature 2.0 \
  --spa_lambda 0.5 \
  --spa_fusion_weight 0.5 \
  --save_dir ./experiments/spa
```

测试：

```bash
python main_spa.py \
  --data_folder ./data \
  --n_cls 4 \
  --test_fold 0 \
  --seed 1 \
  --batch_size 32 \
  --num_workers 8 \
  --desired_length 5 \
  --pad_types repeat \
  --nospec \
  --save_dir ./experiments/spa \
  --eval \
  --checkpoint ./experiments/spa/icbhi_beats_pafa_spa_4class_fold0_seed1/best.pth
```

2-class 只需将 `--n_cls 4` 改为 `--n_cls 2`，并使用相应 checkpoint 目录。

## 12. 从 PAFA checkpoint 初始化 SPA

默认脚本从 BEATs 预训练权重联合训练 PAFA+SPA。也可以先训练 PAFA，再使用其 backbone 和 projector 初始化 SPA：

```bash
python main_spa.py \
  --data_folder ./data \
  --n_cls 4 \
  --test_fold 0 \
  --seed 1 \
  --init_pafa_checkpoint \
    ./experiments/pafa/icbhi_beats_pafa_4class_fold0_seed1/best.pth \
  --save_dir ./experiments/spa_initialized \
  --nospec \
  --cosine \
  --warm
```

该选项不会加载 PAFA 的线性 classifier，因为 SPA 使用双分支分类头；它只加载 BEATs backbone 和 PAFA projector。

## 13. 结果目录

PAFA fold 0：

```text
experiments/pafa/
└── icbhi_beats_pafa_4class_fold0_seed1/
    ├── best.pth
    ├── train_args.json
    └── metrics_eval.json
```

PAFA+SPA fold 0：

```text
experiments/spa/
└── icbhi_beats_pafa_spa_4class_fold0_seed1/
    ├── best.pth
    ├── epoch_100.pth
    ├── train_args.json
    ├── history.jsonl
    ├── metrics_best.json
    └── metrics_eval.json
```

`metrics_eval.json` 包含：

```text
specificity
sensitivity
score
accuracy
macro_f1
ece
nll
brier
confusion_matrix
class_counts
```

ICBHI Score 定义为：

```text
Score = (Specificity + Sensitivity) / 2
```

其中 Specificity 是 Normal 类召回率，Sensitivity 是所有异常类别合并后的召回率。该定义与 PAFA 仓库的 `util/icbhi_util.py:get_score` 保持一致。

## 14. 汇总 5-fold

```bash
bash scripts/summarize_5fold.sh
```

输出：

```text
experiments/summary/
├── runs.csv
├── summary.csv
└── summary.json
```

`summary.csv` 按以下字段分组：

```text
method
n_cls
protocol
```

并对各指标计算 mean 和 sample standard deviation。`5fold` 与 `official_5seed` 会被分开汇总。

## 15. 公平对比原则

比较 PAFA 与 PAFA+SPA 时，应保持以下条件一致：

1. 相同 ICBHI split；
2. 相同 seed；
3. 相同 BEATs checkpoint；
4. 相同输入时长、采样率和 padding；
5. 相同 epoch、batch size、optimizer、LR 和 WD；
6. 相同 PAFA PCSL/GPAL 参数；
7. 都以 validation/test ICBHI Score 选择 best checkpoint；
8. baseline 与 SPA 都由相同的独立评测代码计算 ECE、NLL 和 Brier。

不要将 official split 的 seed 均值与 patient-wise fold 均值直接比较。

## 16. 常见问题

### 16.1 `FileNotFoundError: BEATs_iter3_plus_AS2M.pt`

检查：

```bash
pwd
ls -lh pretrained_models/BEATs_iter3_plus_AS2M.pt
```

必须从 PAFA-SPA 仓库根目录启动脚本，因为 upstream wrapper 使用相对路径。

### 16.2 `ModuleNotFoundError: No module named 'BEATs'`

PAFA 克隆目录必须包含：

```text
BEATs/BEATs.py
BEATs/__init__.py
```

并且命令应从仓库根目录运行：

```bash
cd /home/USER/PAFA-SPA
python -c "from BEATs.BEATs import BEATs, BEATsConfig; print('BEATs import OK')"
```

### 16.3 训练时读取了错误 fold 或错误任务缓存

PAFA dataset 代码会优先读取：

```text
data/training.pt
data/test.pt
```

这些缓存没有编码任务和 fold 信息。本包在每次 SPA 构建 dataset 前，以及 baseline 独立评测前，自动删除它们。执行 upstream `main.py` 前也建议手动删除：

```bash
rm -f data/training.pt data/test.pt
```

5-fold shell 脚本可在每个 fold 前执行该命令。若上游代码未生成这些缓存，命令不会产生影响。

### 16.4 CUDA out of memory

先降低 batch size：

```bash
BATCH_SIZE=16 bash scripts/train_pafa_spa_4class_5fold.sh
```

仍不足时：

```bash
BATCH_SIZE=8 NUM_WORKERS=4 bash scripts/train_pafa_spa_4class_5fold.sh
```

如果 moving-average state copy 造成额外显存或主存压力，可以从 `scripts/common.sh` 的 `COMMON_TRAIN_ARGS` 中移除：

```text
--ma_update
--ma_beta 0.5
```

但这样不再与 PAFA 官方脚本的 moving-average 配置完全一致，baseline 和 SPA 必须同时修改后才能公平比较。

### 16.5 训练初期 rotation 没有更新

默认要等所有类别至少出现一次后才开始 SVD 更新。4-class 的 Both 类较少，通常在前几个 mini-batch 内出现。可以从 checkpoint 中检查：

```python
checkpoint = torch.load("best.pth", map_location="cpu")
print(checkpoint["spa_head"]["class_counts"])
```

所有值应大于 0。

### 16.6 结果不能与论文数字完全一致

先确认使用的是哪种协议：

- SPA 论文主结果：official 60/40 + 5 seeds；
- `*_5fold.sh`：patient-wise fold 0 至 4。

还需要核对 PyTorch/CUDA、BEATs 权重、随机种子、PAFA commit、batch size 和 moving-average 配置。不同 PAFA commit 或不同 CUDA kernel 可能产生轻微差异。

## 17. 代码验证范围

代码包包含可在 CPU 上运行的 ETF、Procrustes、SPA head、SPA loss 和 metric 单元测试。正式 BEATs 训练还需要：

- 完整 PAFA 仓库；
- ICBHI 数据；
- BEATs_iter3+ AS2M 权重；
- CUDA GPU。

安装后应先完成第 6 节检查，再提交正式 5-fold 作业。

## 18. 参考项目

- PAFA: https://github.com/wa976/pafa
- PAFA paper: https://arxiv.org/abs/2505.23834
- BEATs: https://github.com/microsoft/unilm/tree/master/beats
- ICBHI 2017: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
- SPA: https://github.com/wangying1586/SPA
