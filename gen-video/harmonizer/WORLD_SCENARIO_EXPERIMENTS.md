# World Scenario 实验与无 World Scenario 的 baseline 对照

## 2026-10-16：修复训练后自动测试的模型规格

此前 0.6B 模型规格仅在训练子脚本中 export，无法传回统一入口；后续自动测试因此回落到 `get_pipeline_config()` 的 2B 默认值。结果是 0.6B checkpoint（宽度 1280、20 层）被加载进 2B 模型（宽度 2048、28 层），严格加载时报 size mismatch。之前 GPU 评测检查手工 export 了 0.6B 环境变量，未覆盖这个 shell 作用域问题。

修复只在统一训练入口和独立评测入口明确 export 相同的 0.6B 规格及权重目录，不修改模型结构、checkpoint 加载规则、训练步数或损失。新增 shell 边界回归检查：不预设模型规格环境变量，覆盖五组实验和两个独立评测调用；修复前 7 项全部失败，修复后全部通过。

实际 GPU 回归也已通过：在 allocation 7852767 的四张 RTX A6000 上，启动前清除 `COSMOS_MODEL_SIZE` 和 `COSMOS_CHECKPOINT_DIR`，直接调用修复后的评测入口，严格加载用户正式 Stage II 的 `model_6000.pkl`。四个 worker 完成 96 帧四方向测试子集（含 V=1/V=5）、24 个 MP4，以及整体和各方向的有限 FID/KID/VideoGPT FVD，脚本正常退出。该检查不是完整 4,844 帧评测。日志与结果保存在 `../../world_scenario_gpu_checks/model_selection_fix/`。

现有 `baseline_mode1/stage2/checkpoints/model_6000.pkl` 无需重新训练。在 GPU 节点上，从 harmonizer 目录单独补跑完整测试：

```bash
ROOT=/fs/nexus-projects/AD_dashrecon/world-model-phucnda-wm
OUTPUT_ROOT="$ROOT/train_world_scenario/baseline_mode1" \
TEST_DATASET="$ROOT/data/dh_train/smoke_nuscenes_dr2m/dataset_map_box_test20_4dir_ordered.json" \
bash testing_script/eval_world_scenario.sh baseline 1 \
  "$ROOT/train_world_scenario/baseline_mode1/stage2/checkpoints/model_6000.pkl" 4
```

补跑成功后，再启动 baseline Mode 2 和 Mode 0；原来用 `&&` 连接的后两组命令在首组测试报错后不会执行。

状态：用户已完成 baseline Mode 1 的正式 Stage I/II 训练；自动测试的模型规格问题及恢复命令见上节。此前已完成模型、训练、推理和统一启动入口，77 项 CPU 检查通过，并在 allocation 7852764 的两节点、8 张 RTX A6000 上通过实际 GPU 检查、四组全阶段 DDP 短训，以及四组 96 帧视频与指标评测。

## 已明确的实验设置

- 只使用 `world_scenario_image`（地图＋实心 3D 框合成图），一个条件分支；不同时输入 `hdmap_image`。
- 两种方法都从 Cosmos Predict2 0.6B 初始化，四张参考图、相机/时间条件、冻结 VAE 和原来的损失定义保持不变。
- ours 使用原来的 independent mode-B 三阶段 MeanFlow；Stage III 使用当前 Stage III EMA 生成历史，仅 Stage III 开启 EMA。
- World Scenario 是外部条件，不作为 RGB/latent 监督目标，也不混入 opacity/noise source endpoint。
- 当前、参考、GT 历史和生成历史都使用各自 scene/frame 对应的条件图。生成历史的条件图取该历史帧的真实场景条件，不从预测 RGB 推断条件。
- Stage III 的单帧历史生成也要传入该历史帧及其参考帧的条件，不能只在随后五帧训练时添加条件。

## 数据核验

使用以下新清单，路径均相对于 `../../data/dh_train/smoke_nuscenes_dr2m`：

| 用途 | 清单 | 物理 scene | 当前目标记录 | 每方向记录 |
| --- | --- | ---: | ---: | ---: |
| 训练 | `dataset_map_box_train100_4dir_ordered.json` | 100 | 24,256 | 6,064 |
| 验证 | `dataset_map_box_val80_4scene_4dir_ordered.json` | 4 | 80 | 20 |
| 测试 | `dataset_map_box_test20_4dir_ordered.json` | 20 | 4,844 | 1,211 |

方向为 D2M/L2M/R2M/U2M。训练和测试的每个物理 scene 均覆盖四方向，且各方向保留帧相同。验证沿用四个不同 scene 各取一个方向、每个方向连续 20 个目标的选择方法，并非每个验证 scene 都有四方向。

本次重新检查了全部清单记录，以及 124 个物理 scene 的 12,400 张原始帧 World Scenario PNG：文件存在、PNG 校验通过、尺寸为 512×288。当前帧、四张参考图和显式历史路径与同 scene/frame 条件对应；共检查 136,217 个去重后的其他媒体/元数据路径存在。三个物理 scene 集合两两无交集。渲染元数据中的 source/target original 相机位姿一致，double reprojection 开启。该核验不包含重新投影原始地图几何。

机器可读报告：`../../data/dh_train/smoke_nuscenes_dr2m/WORLD_SCENARIO_DATA_AUDIT.json`。

保留原始选择：训练 scene-0021、scene-0072 没有满足清单筛选条件的五连续帧窗口；时序训练保持原有从其他合格窗口取样的行为。测试 scene-0179、scene-0192 也无五连续清单记录。测试集不是每 scene 完整的 100 帧视频，不能隐含假定全部保留帧连续。

## 样本预算与命令行

每 GPU batch=1、gradient accumulation=1。用户最新确认保留入口现有的两倍训练步数，以总共四卡为预算基准：

| 方法 | Stage I | Stage II | Stage III | 累计当前目标样本数 |
| --- | ---: | ---: | ---: | ---: |
| baseline（Mode 0/1/2） | 12,000 | 12,000 | — | 96,000 |
| ours | 12,000 | 4,800 | 7,200 | 96,000 |

总卡数 G = nodes × gpus_per_node，各阶段实际 optimizer steps = 四卡基准 steps × 4 / G。例如 2×4 卡时，baseline 为 6000/6000，ours 为 6000/2400/3600；4×4 卡时分别为 3000/3000 和 3000/1200/1800。不能整除的组合应明确报错，不静默取整。

此规则对齐累计当前目标样本预算。改变 global batch 后，随机时序替换、噪声、参数更新和数据分片可能不同，不能据此声称逐样本身份或训练轨迹完全相同。两种条件模式在相同方法、相同卡数下共用清单、seed 和 sampler 设置。参考/历史出现不计为额外当前目标监督。

统一入口：

```bash
bash training_script/train_world_scenario.sh \
  --method baseline --mode 1 --nodes 2 --gpus-per-node 4
```

`--method baseline|ours` 与 `--mode 1|2` 组合对应四组条件实验；新增 `--method baseline --mode 0` 为不输入 World Scenario、不创建条件 PatchEmbed 或 ControlNet 的原始 baseline 对照。Mode 0 仅支持 baseline。输出独立存放。所有训练阶段结束后，从该实验最终阶段 checkpoint 自动进行四方向测试推理、视频导出、FID/KID/FVD 计算并保存汇总。

## 条件 VAE 编码

使用现有冻结 Cosmos tokenizer，并沿用 `vae_encode` 中的 sigma_data 缩放；不引入另一个 VAE。

- 当前 RGB 路径为 V=1 时，条件图也按 V=1 编码。
- 当前＋四历史 RGB 路径为 V=5 时，同帧顺序的五张条件图也联合视频编码为两个 latent 时间位置。
- 四张参考图的条件各自独立编码，参考轴不能当作视频时间轴压缩。
- 按 visibility 重排参考 RGB 时，必须对参考条件使用相同 permutation。
- 所有条件 VAE 编码 no_grad；原来用于 temporal loss 的预测 latent 解码仍可微。

## Mode 1

直接使用本地 Cosmos 的 `PatchEmbed`，即 patch rearrange＋无 bias Linear，而非凭名称换成不同卷积结构。新增独立条件 PatchEmbed，权重全零；同一个条件 PatchEmbed 在当前/历史和参考流间共享。

```text
条件图 → 冻结 VAE → 条件 PatchEmbed → 加到对应 RGB patch embeddings
```

相加发生在当前/历史的原 patch embeddings 上，以及参考流按 top-1/其余参考分流之前。原 top-1 joint self-attention、其余 cross-attention K/V 机制继续使用。初始化时新增条件贡献严格为零。目标和损失不变；ours MeanFlow target 中的 JVP 自然经过增加条件后的预测函数。

参考：本地 `omni-dreams/post-training/omnidreams/_src/omnidreams/networks/minimal_v1_lvg_dit_hdmap.py` 第 39–67、103–108 行。官方实现将条件 PatchEmbed 输出加到原 tokens，并支持 zero_init。此处按用户要求扩展到参考流。

## Mode 2：用户确认的结构

本地 Cosmos-Drive-Dreams 的 cosmos-transfer1 子模块目录为空。已按该库 gitlink 锁定的 commit `b25a3ac8efe277931c62b4e23a5437afb468ef0c` 读取官方源码，未安装或替换任何依赖。

官方 AV experiment 为 28 层主干、3 个控制 blocks；它有独立 hint PatchEmbed、SiLU MLP 条件编码器（末层零初始化）、每个控制 block 的零初始化 residual projection。控制 blocks 由基础模型对应参数初始化，主干接收各层 residual。不是复制完整生成网络与输出 head。

已实现的适配：

1. 0.6B 主干有 20 层、宽度 1280；控制分支复制前 3 层及所需 patch/time 组件，使用原生 Cosmos blocks，不建第二个完整生成 DiT。
2. 条件编码器沿用官方 hint PatchEmbed＋SiLU MLP＋零初始化末层；每层输出使用零初始化 residual projection。控制权重固定为 1，不加入新损失或 CFG/dropout 策略。
3. 当前/历史序列作为整体走控制分支；输出 residual 分别加到主干第 0、1、2 层输出。
4. 参考图各自独立走同一个控制分支并共享权重；最终参考 residual 加到主干参考 patch embeddings，再执行原 top-1 joint self-attention／其余 cross-attention 分流。这是为本项目参考融合结构提出的适配，并非官方 AV 模型现成接口。
5. 主干继续全量训练，VAE 保持冻结。官方 AV 配置本身冻结主干；这里继续训练主干是为了保持用户指定的 baseline/ours 训练策略，已由用户确认。
6. ours 控制分支必须参与完整 z/t JVP，并接收 t-r 条件；不能 detach 分支、遗漏其 t 依赖或只在额外 RGB forward 中添加控制。

参考源码（锁定 commit）：

- https://github.com/nvidia-cosmos/cosmos-transfer1/blob/b25a3ac8efe277931c62b4e23a5437afb468ef0c/cosmos_transfer1/diffusion/networks/general_dit_ctrl_enc.py
- https://github.com/nvidia-cosmos/cosmos-transfer1/blob/b25a3ac8efe277931c62b4e23a5437afb468ef0c/cosmos_transfer1/diffusion/config/training/experiment/ctrl_7b_tp_sample_av.py
- https://github.com/nvidia-cosmos/cosmos-transfer1/blob/b25a3ac8efe277931c62b4e23a5437afb468ef0c/cosmos_transfer1/diffusion/training/models/model_ctrl.py

## 必要接入与验证

- 数据加载、batch preparation、训练 forward、生成历史、训练中验证、独立推理和最终测试均传递同一条件字段。
- baseline 当前要求训练 JSON 同时包含 train/test，但新训练清单只有 train；应让新实验显式传入独立验证清单，不能误用真正 test split 做训练中验证。
- checkpoint 保存条件模式/分支结构，按实验参数构造相同模式、核对元数据后严格加载，阶段间不重新零初始化已经训练的条件权重，EMA 包括新增可训练条件参数。
- 测试要覆盖 D2M/L2M/R2M/U2M。现有旧测试 wrapper 只跑 R2M/U2M，不能直接复用为四方向结果。
- 本次数据是 double reprojection，有对应 target RGB；生成指标使用新测试清单。现有 FVD 会拼接同 scene 所有保留帧而不检查间断，按用户选择继续使用这种保留帧顺序协议。
- 验证重点：零初始化时与无控制模型输出一致；训练后条件能影响对应 target/reference 流；五帧与四独立参考的条件顺序一致；非零分支的 MeanFlow JVP/梯度正确；阶段 checkpoint/EMA 连续；单/多节点 dry-run 的样本预算正确；最终测试未泄露到训练验证。

用户已明确：Mode 2 采用上述 3 层结构；ours 使用 WAFT；测试沿用各 scene 按保留帧顺序拼接（包含帧号间断）的旧协议。

## 已实现入口与低层细节

- `training_script/train_world_scenario.sh`：统一参数入口。在 Slurm allocation shell 运行时，单节点和多节点都通过 srun 启动每节点一个 launcher；未使用 Slurm 的单节点 GPU 机器可直接运行。`--dry-run` 打印所有阶段及测试命令。
- `testing_script/eval_world_scenario.sh`：最终 checkpoint 的四方向推理、视频和 FID/KID/VideoGPT FVD，评测使用节点 0 的 `gpus-per-node` 张卡。baseline 推理 4 步；ours 使用 Stage III EMA 推理 1 步。
- `testing_script/world_scenario_outputs.py`：记录清单 SHA256/实际样本预算、调用 FFmpeg 导出视频、配置离线 Clean-FID cache、汇总四组指标。FFmpeg 使用 concat demuxer、原 scene 元数据 fps、512×288、libx264/crf18/yuv420p；所有保留帧按帧号排序，不插值缺帧。参考 [FFmpeg concat 文档](https://ffmpeg.org/ffmpeg-formats.html#concat-1)。
- `src/model/world_scenario.py`：Mode 2 的唯一新模型模块。主干、分支和 VAE 使用现有环境，无新增依赖。

Mode 1 的条件 patch 是 16 latent channels → rearrange(temporal=1, spatial=2) → Linear(64,1280,bias=False)，权重全零。参考流和 target/history 共享同一个新增投影。

Mode 2 的条件 MLP 宽度为 `1280→16→16→32→32→96→96→256→1280`，中间 SiLU，末层全零；各控制 block 之后的 `1280→1280` residual Linear 权重和 bias 全零。hint 在控制分支第一个 block 后加入，与锁定 commit 的官方 AV 编码器一致。复制的分支不包含主干输出 head，也不复制新增的参考 K/V 或 camera 分支；原主干继续负责这些融合。

每张参考图的独立控制分支调用使用当前 DiT 调用的相同 t（native Cosmos scaling）和 t-r（未缩放）；参考 RGB/条件本身保持 clean。四张参考按 batch 展开、各自处理，随后恢复参考轴。主干原本的 camera/time/PRoPE 条件继续保留。目标/历史控制分支使用与主干一致的 fps，参考单帧按原生单帧位置编码处理。

新增模块初始化使用独立保存/恢复的 Torch RNG 上下文，避免 Mode 1/2 的模块初始化消耗改变原参考模块初始化和训练 sampler RNG。条件图不加噪，不改变原 RF bridge target。

无 World Scenario 的 baseline 输出为 `../../train_world_scenario/baseline_mode0/`；四组条件实验输出为 `../../train_world_scenario/{baseline,ours}_mode{1,2}/`，包含各阶段 checkpoint/验证记录、`run_config.json`、`test/evaluation_manifest.json`、`test/videos/*.mp4`、`test/generative_metrics.yaml`；父目录 `metrics_summary.csv` 汇总已经完成的实验。单独调用推理 CLI 时需传与 checkpoint 一致的 `--world_scenario_mode 1|2`；统一入口自动传入。

```bash
cd /fs/nexus-projects/AD_dashrecon/world-model-phucnda-wm/gen-video/harmonizer
# 每条命令在相应 allocation shell 中启动一次；示例为 2 节点，每节点 4 卡。
bash training_script/train_world_scenario.sh --method baseline --mode 0 --nodes 2 --gpus-per-node 4
bash training_script/train_world_scenario.sh --method baseline --mode 1 --nodes 2 --gpus-per-node 4
bash training_script/train_world_scenario.sh --method baseline --mode 2 --nodes 2 --gpus-per-node 4
bash training_script/train_world_scenario.sh --method ours --mode 1 --nodes 2 --gpus-per-node 4
bash training_script/train_world_scenario.sh --method ours --mode 2 --nodes 2 --gpus-per-node 4
```

CPU 检查包括 77 passed / 4 skipped；新增 9 项检查直接运行原生 Cosmos patch/attention/Transformer blocks，验证零初始化严格等价、当前/top-1/cross-reference 控制有效、checkpoint roundtrip、两个 JVP backend 的有限差分比较和生成历史条件顺序。完整 pipeline 的 text encoder 在无 CUDA 主机上导入时触发已有 FlashAttention 断言；未通过删改模型或替换 attention 绕过。GPU 检查结果另行补充。

## 实际 GPU 检查结果

2026-10-15，在用户指定 tmux 2 的 allocation 7852764（gammagpu13/17，各 4×RTX A6000 48GB）运行。未改变已有 tmux shell、未启动正式预算长程实验。

`tests/gpu_world_scenario_smoke.py` 的八个真实 0.6B 模型检查均通过：baseline/ours × Mode 1/2 × V=1/5。包含真实 VAE、四张参考、camera/time/PRoPE、严格零初始化输出等价、前向/反向、冻结 VAE 但 decoder 可微、RF/MeanFlow、WAFT、Stage III EMA 生成四张历史并联合重编码、checkpoint/EMA roundtrip 和推理保存图片。

实际 `resize_288x512` 的 tokenizer 输出为 `[B,16,1或2,36,64]`，空间压缩为 8；四参考输出 `[B,16,4,36,64]`。

| 检查 | Mode 1 模型进程峰值 GiB | Mode 2 模型进程峰值 GiB |
| --- | ---: | ---: |
| baseline V=1 | 8.86 | 9.79 |
| baseline V=5 | 16.08 | 17.65 |
| ours V=1（强制 r<t） | 6.89 | 7.80 |
| ours V=5＋WAFT＋Stage III | 23.55 | 26.89 |

以上是 `torch.cuda.max_memory_allocated` 的模型进程值，不包括 WAFT 独立 worker 和其他 CUDA runtime 分配。检查验证执行与梯度正确性，不表明模型已收敛或生成质量达到目标。

报告：`../../world_scenario_gpu_checks/smoke_summary.json`；完整日志 `../../world_scenario_gpu_checks/smoke.log`。

真正两节点八卡训练入口检查使用 `tests/gpu_world_scenario_ddp.sh`，每阶段 2 optimizer steps，四实验均测试原 trainer 的阶段交接；为覆盖分支，检查专用脚本强制 temporal probability 和 unequal MeanFlow interval probability 为 1。正式入口仍使用原来的 0.75 和 0.25。

发现并修复 baseline tracker 的接口问题：`--report_to none` 必须传给 Accelerate `log_with=None`，不能传字符串 `"none"`；不改变训练目标。依据 [Accelerate tracking guide](https://huggingface.co/docs/accelerate/usage_guides/tracking) 和本地 `filter_trackers` 实现。首次启动多进程同时下载 VGG 缓存触发 NFS stale-handle；缓存完整后单独验证了 32 个 tensor、138,357,544 参数，再运行相同 trainer，未修改 LPIPS 或模型数学。

条件图与相应 RGB 使用同一训练 resize（LANCZOS）和 [-1,1] normalization；推理沿用原 RGB 路径的 BILINEAR resize。没有新增条件 dropout、条件噪声、CFG 或损失项。所有条件图由同一冻结 tokenizer 编码。

Mode 2 在原 Option-A 模型基础上新增约 100.43M 可训练参数；实际日志中的 baseline Mode 1/2 trainable UNet 参数约为 704.56M/804.99M（不含随后 ours 的 interval embedder）。

## 四组两节点 DDP 与评测链路结果

四实验全部通过，两节点共八卡；baseline 两阶段、ours 三阶段，每阶段各 2 optimizer steps。10 个阶段都保存了 checkpoint，正常完成阶段交接；ours Stage III 包括 EMA 和 WAFT。完整日志为 `../../world_scenario_gpu_checks/ddp.log`，索引为 `ddp_summary.json`。

随后每组使用该次 DDP 的最终 checkpoint，从新测试清单截取的固定 96 帧执行原生自回归推理，覆盖 2 个物理 scene × 四方向 × 12 连续帧。每组均生成 96 张预测图、24 个 prediction/target/input MP4（8 条方向-scene 序列），再实际计算整体和 D/L/R/U 分组的 Clean-FID、KID、VideoGPT FVD。全部完成，无 NaN/Inf。正式入口仍测试原来的全部 4,844 帧，并沿用包含帧号间断的旧协议。

| 功能短训检查 | FID（96 帧） | KID mean | VideoGPT FVD（8 clips） |
| --- | ---: | ---: | ---: |
| baseline_mode1 | 465.8262 | 0.565904 | 5371.4438 |
| baseline_mode2 | 454.0547 | 0.588710 | 9690.6924 |
| ours_mode1 | 396.7063 | 0.395115 | 5119.6719 |
| ours_mode2 | 417.2654 | 0.505212 | 14346.5586 |

这些数字仅用于确认指标链路实际可执行：训练只有每阶段 2 步，baseline/ours 短训的总阶段数不同，且样本规模很小，不是正式公平质量比较，也不能说明方法优劣或收敛情况。本次没有为改善这些分数调整模型、损失或正式配置。

小清单结果：`../../world_scenario_gpu_checks/evaluation_summary.json` 和 `../../world_scenario_gpu_checks/evaluation/metrics_summary.csv`；各组 `test/` 中保留实际 PNG、MP4 和 YAML。首次将小清单移到检查目录后，按原 parser 要求为它补上了显式 source/target calibration 绝对路径；没有增加路径自动搜索，也没有修改正式清单或 evaluator 的校准逻辑。

完成 roundtrip 后清理了八份独立 smoke 的大型临时 checkpoint；其结果 JSON/预测 PNG 保留，四组 DDP 阶段 checkpoint 和最终评测输出保留。

## 训练验证入口的实际 GPU 检查

`tests/gpu_world_scenario_validation.py` 在八张 GPU 上分别运行两种条件模式 × Stage I / Stage II GT-history / Stage II autoregressive / Stage III EMA autoregressive，直接调用实际 `build_validation_dataset` 和 `run_validation`。八个检查全部通过，分别覆盖完整 80 个验证目标，总计 640 次当前帧验证。

每个模式的 Stage I 时序样本数为 0，Stage II GT-history 为 80，Stage II autoregressive 与 Stage III 为 64（四个场景各自前四帧 V=1，之后 16 帧 V=5）。Stage III 确认使用保存的 EMA；验证后模型恢复 training、VAE 保持 eval/frozen。结果为 `../../world_scenario_gpu_checks/validation_summary.json`，各目录保留真实预览图、history strip 和逐样本 metadata。

全部 GPU 检查已结束；正式 6000/6000 或 6000/2400/3600 的预算实验尚未启动。最终 CPU 回归仍为 77 passed / 4 skipped。

## 无 World Scenario 的 baseline 对照

新增 `--method baseline --mode 0`，复用同一训练、验证、测试清单与评测协议，但数据加载不读取条件图，模型不创建条件 PatchEmbed 或 ControlNet，也不编码 World Scenario。原有四张 RGB 参考、top-1 joint self-attention、其余参考 cross-attention、历史 RGB、camera/time 条件保留。运行记录中的 `controls` 为 null，`control_blocks` 为 0。最终仍自动测试全部 4,844 帧并输出视频、FID/KID/VideoGPT FVD。

训练入口现有步数未修改。运行记录修正为反映实际入口预算：baseline 48,000/48,000 个当前目标样本，ours 48,000/19,200/28,800，总计均为 96,000。
