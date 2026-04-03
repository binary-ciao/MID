# MID 项目工作日志

## 汇总
| 日期 | 会话时长 | 完成内容 | commit |
|------|----------|---------|--------|
| 2026-03-31 | ~1.5h | 环境搭建、数据处理、git初始化 | ✅ |
| 2026-04-02 | ~2h | 原始数据格式分析、ETH 训练 | ✅ |
| 2026-04-03 | ~2h | VIRAT 场景重构、ETH/UCY 3/5 训练完成 | ✅ |

---

## 2026-04-02

### 原始数据格式（ETH/UCY 数据集）

原始轨迹文件（`.txt`）每行 5 列，含义如下：

| 列索引 | 列名 | 含义 |
|--------|------|------|
| 第1列 | `frame_id` | 帧ID / 时间戳标识，对应视频帧序号 |
| 第2列 | `track_id` | 行人ID / 轨迹ID，标识同一行人的不同时间步 |
| 第3列 | 固定值 | 部分数据集全为 1.0 或 0（无意义） |
| 第4列 | `pos_x` | 行人世界坐标 X（水平方向位置） |
| 第5列 | `pos_y` | 行人世界坐标 Y（垂直方向位置） |

**注意**：ETH/UCY 数据集为 2D 坐标（无 pos_z），z 坐标仅在 Stanford Drone Dataset（SDD）中存在。

示例：
```
1  780  1.0  8.46  3.59
│   │    │     │     └── Y 坐标
│   │    │     └─────── X 坐标
│   │    └──────────── 固定值（1.0）
│   └──────────────── 行人ID = 780
└───────────────────── 帧ID = 1
```

`process_data.py` 预处理步骤：
- 帧间隔统一为 1（通过 `frame_id // 10` 采样）
- 计算每个行人的 `position` / `velocity` / `acceleration`（x, y 方向）
- 去中心化（减去均值）和归一化

### 模型预测基于的数据

MID 不仅依赖历史轨迹和行人距离，还包括以下多层次输入：

#### 1. 目标行人历史状态（`x`）
来自 `preprocessing.py:91`，包含 **6 维特征**：

| 特征维度 | 含义 |
|---------|------|
| `position.x/y` | 历史位置（x, y 坐标） |
| `velocity.x/y` | 历史速度（由位置差分计算） |
| `acceleration.x/y` | 历史加速度（由速度差分计算） |

#### 2. 邻居行人社会交互（`neighbors_data_st`）
来自 `preprocessing.py:110-150`：

- **邻居选择**：`attention_radius`（注意力半径），行人-行人默认 **5.0 单位**
- **邻居特征**：同样包含邻接行人的 position / velocity / acceleration
- **边权重**：`weight_cube = 1/dist`（距离倒数），距离越近权重越大
- **边编码（edge_encoding）**：动态边掩码，过滤短暂消失/出现的边

#### 3. 地图信息（可选，`use_map_encoding`）
来自 `preprocessing.py:166-188`：
- 地图 patch（裁剪后的地图块）
- 航向角（heading angle）：基于速度向量计算

#### 4. Robot/障碍物轨迹（可选，`incl_robot_node`）
如果场景中存在机器人或车辆，也会被纳入考虑。

### 总结

| 数据类型 | 核心信息 |
|---------|---------|
| **行人自身历史** | 位置 + 速度 + 加速度（8 帧） |
| **社会交互** | 邻居历史轨迹 + 基于距离的注意力权重 |
| **地图** | 场景拓扑结构（可选） |
| **障碍物/机器人** | 其他智能体的未来轨迹（可选） |

模型通过 `attention_radius` 构建**空间邻近的社会交互图**（Scene Graph），用 **Transformer** 编码**时空依赖关系**，并结合地图进行综合预测。距离本身是加权因子而非直接特征。

---

## 2026-03-31
- **21:53** - 开始会话
- **22:15** - 环境检查完成：GPU RTX 4060, PyTorch 2.5.0, NumPy 1.24.3
- **22:30** - 修复 Python 3.11 兼容性：collections.Sequence → collections.abc.Sequence
- **22:35** - 安装 ncls、orjson
- **22:45** - process_data.py 成功运行，5个数据集（eth/hotel/univ/zara1/zara2）处理完毕
- **22:50** - git init/commit 完成，创建 dev 分支
- **23:00** - 推送到 GitHub https://github.com/binary-ciao/MID.git (dev分支)

---

## 2026-04-02（续）

### VIRAT 数据集注释文件复制

原始注释数据已从 `D:\Codes\dataset\VIRAT-dataset\viratannotations-master` 复制到 `VIRAT-dataset/annotations/`。

**复制结果**：
- 共 18 个独立注释文件（train: 11, validate: 7）
- 每个文件包含 4 种注释类型：
  - `.geom.yml`：几何轨迹（每帧 bbox）
  - `.types.yml`：目标类型（Person/Vehicle/Other）
  - `.activities.yml`：活动标签（Talking/Walking 等）
  - `.regions.yml`：区域多边形（可能含地面投影坐标）

**各场景可用性**：

| 场景 | 原始视频 | 注释文件 | clip数 | 可用性 |
|------|---------|---------|--------|--------|
| VIRAT_S_000001 | ✓ | train/ | 133 | ✅ |
| VIRAT_S_000002 | ✓ | train/ | 84 | ✅ |
| VIRAT_S_000007 | ✓ | validate/ | 83 | ✅ |
| VIRAT_S_000008 | ✓ | validate/ | 81 | ✅ |
| VIRAT_S_000200 | ✓ | train+validate/ | 61 | ✅ |
| VIRAT_S_000201 | ✓ | train+validate/ | 92 | ✅ |
| VIRAT_S_000204 | ✓ | train+validate/ | 81 | ✅ |
| VIRAT_S_000205 | ✓ | train+validate/ | 27 | ✅ |
| VIRAT_S_040000 | ✓ | train+validate/ | 6 | ✅ |
| VIRAT_S_040001 | ✓ | train/ | 59 | ✅ |
| VIRAT_S_040003 | ✓ | train+validate/ | 54 | ✅ |

**总计：795 个 clip 全部有对应注释文件**

### VIRAT 数据能否用于 MID 模型训练/测试？

#### 原始注释数据 vs Qwen 处理数据

| 对比项 | Qwen clip 数据 | 原始 VIRAT 注释 |
|--------|---------------|----------------|
| 每帧行人数 | 1（单行人 clip） | 11 人 + 3 车（多行人） |
| 坐标系统 | 像素坐标 center_x/y | geom.yml: 像素坐标 bbox; regions.yml: 部分为地面投影坐标 |
| 帧范围 | 连续 ~239 帧/clip | 连续 ~9000-20000 帧/视频 |
| 社会交互 | ❌ 缺失 | ✅ 存在（多行人同帧） |

#### 关键发现

1. **regions.yml 坐标 = geom.yml 的浮点版本**：经对比同一 track 同一帧的 bbox 与 poly0，数值完全一致（仅多小数位），**不是地面投影坐标**，仅是 bbox 的另一种表示
2. **多行人轨迹丰富**：VIRAT_S_000001 有 11 个行人轨迹，前 1000 帧中 100% 帧有 2 人以上同时出现；VIRAT_S_000002 有 7 人，前 1000 帧 100% 多人在场

#### 坐标验证（关键发现）

| track | frame | geom bbox | regions poly0 |
|-------|-------|-----------|---------------|
| 4 | 3579 | left=0 top=772 right=76 bottom=917 | [[0.92, 772], [0.92, 917], [76.3, 917], [76.3, 772]] |
| 18 | 16927 | left=1883 top=98 right=1915 bottom=188 | [[1883, 98], [1883, 188], [1915, 188], [1915, 98]] |

**结论：regions.yml 是 bbox 的精确浮点版本，无坐标系转换。**

#### Q1：能否通过算法估计单应性矩阵（Homography）？

**答：不可行。**

| 原因 | 说明 |
|------|------|
| 无相机标定 | VIRAT 不提供内参+外参 |
| 无像素-世界对应点 | 无法建立 4+ 对应关系 |
| 行人身高未知 | 透视恢复所需约束不满足 |
| regions.yml 无额外信息 | 与 geom.yml 坐标一致 |

#### Q2：能否忽略坐标系差异，直接处理为 MID 格式？

**答：理论上可行，且实际有价值。**

| 对比项 | ETH/UCY | VIRAT（像素） |
|--------|---------|--------------|
| 坐标系统 | 世界坐标（米） | 像素坐标 |
| 帧率 | **10 fps** | **30 fps**（需降采样） |
| 社会交互 | 多行人 | **7-11 人/视频，100%覆盖** ✅ |
| 数据量 | ~500 轨迹 | 每视频 7-11 轨迹 × 数千帧 |

**VIRAT 的优势**：
- 原始注释覆盖完整视频（数千帧连续），非碎片化 clip
- 多行人场景比 ETH/UCY 更密集，社会交互数据更丰富
- 活动标签（Talking/Walking 等）可用于辅助预测

**需要处理的问题**：

| 问题 | 解决方案 |
|------|---------|
| standardization 基于米制 | 改用像素标准化（std=1, mean=0） |
| 帧率 30fps vs 10fps | 每 3 帧取 1 帧 |
| 评估指标单位 | 以像素为单位的 ADE/FDE |

**可行的处理方案**：
1. 新写 `process_virat.py`，从 geom.yml 直接提取多行人轨迹
2. 跳过年/米转换，用 `(center_x, center_y)` 作为位置
3. 降采样：30fps → 10fps
4. 用与 ETH 相同的 Scene Graph 方式构建社会交互
5. 以像素单位训练和评估（ADE 等指标仅改变数值尺度，预测误差模式仍然有效）

#### Q3：多视角问题

VIRAT 11 个场景来自不同摄像头，视角差异大（y 跨度从 350px 到 1053px 不等），无法混合训练。

**最终策略：ETH/UCY 和 VIRAT 完全独立处理**

| 数据集 | 处理方式 | 评估指标 | 适用场景 |
|--------|---------|---------|---------|
| ETH/UCY | 原有 `process_data.py` | ADE/FDE（米） | 与原论文 baselines 对比 |
| VIRAT | 新写 `process_virat.py` | ADE/FDE（像素） | 多模态/社会交互验证 |

两者互补，不混合、不对比、各自独立训练评估。

---

## 下一步要做
- [x] 尝试训练：`python main.py --config configs/baseline.yaml --dataset eth`
- [x] 阶段2：复现官方训练，生成 checkpoint ✅
- [x] 新写 `process_virat.py`（已完成）
- [ ] 补完 ETH/UCY 复现：zara1 / zara2
- [ ] VIRAT 独立训练和评估

### 阶段2完成：ETH/UCY 90 Epoch 训练（3/5 完成）

**训练配置**：lr=0.001, encoder_dim=256, tf_layer=3, batch_size=256, augment=True

#### ETH（已完成）
| 检查点 | ADE | FDE |
|--------|-----|-----|
| Epoch 30 | 1.590 | 2.065 |
| Epoch 60 | 1.547 | 2.005 |
| **Epoch 90** | **1.536** | **1.999** |

Checkpoint: `experiments/baseline/eth_epoch90.pt`
日志: `experiments/baseline/eth_2026-04-02-22-12.log`

#### HOTEL（已完成）
| 检查点 | ADE | FDE |
|--------|-----|-----|
| Epoch 30 | 0.976 | 1.391 |
| Epoch 60 | 0.959 | 1.362 |
| **Epoch 90** | **0.957** | **1.332** |

Checkpoint: `experiments/hotel/hotel_epoch90.pt`
日志: `experiments/hotel/hotel_2026-04-03-13-33.log`

#### UNIV（已完成）
| 检查点 | ADE | FDE |
|--------|-----|-----|
| Epoch 30 | 0.861 | 1.145 |
| Epoch 60 | 0.860 | 1.147 |
| **Epoch 90** | **0.862** | **1.149** |

Checkpoint: `experiments/univ/univ_epoch90.pt`
日志: `experiments/univ/univ_2026-04-03-18-02.log`

#### ZARA1（待训练）
- 配置文件: `configs/zara1.yaml`
- 训练命令: `python main.py --config configs/zara1.yaml --dataset zara1`

#### ZARA2（待训练）
- 配置文件: `configs/zara2.yaml`
- 训练命令: `python main.py --config configs/zara2.yaml --dataset zara2`

---

## 2026-04-03

### VIRAT 场景重构

**发现场景分组问题**：VIRAT 的 11 个 clip 实际对应 **3 个独立摄像头**，不是 8 个。

| 场景 | 包含 clips | 来源 |
|------|-----------|------|
| **S1** | 000001 + 02 + 07 + 08 | 同一摄像头不同时间段 |
| **S2** | 040003 + 01 + 00 | 同一摄像头不同时间段 |
| **S3** | 0200_03 + 05 + 00 + 01_03~07 + 204_00 + 04 + 205_01 | 同一摄像头 |

**train/test 策略**：统一 80/20 时间切分

| 场景 | 训练 nodes | 测试 nodes | 训练 rows | 测试 rows |
|------|-----------|-----------|-----------|-----------|
| **S1** | 38 | 17 | 58,548 | 10,519 |
| **S2** | 46 | **4** ⚠️ | 29,694 | 2,165 |
| **S3** | 95 | 31 | 38,373 | 14,001 |

**输出文件**（逐场景，对齐 ETH/UCY 模式）：
- `processed_data_virat/virat_s1_train.pkl` + `virat_s1_test.pkl`
- `processed_data_virat/virat_s2_train.pkl` + `virat_s2_test.pkl`
- `processed_data_virat/virat_s3_train.pkl` + `virat_s3_test.pkl`

### 代码修复

| 修改 | 说明 |
|------|------|
| `process_virat.py` | 重构为 3 场景、每场景独立 train/test、`clip_split` 跟踪、`NodeType` enum 修复 |
| `dataset/preprocessing.py` | `collate` 增加 numpy scalar 处理（PyTorch 2.x 兼容） |
| `models/autoencoder.py` | 修复 `.numpy()` 调用 |
| 环境 | NumPy 从 2.2.6 降至 1.26.4（PyTorch 2.5.0 编译要求） |
| 安装 tensorboardX | 依赖缺失 |

### VIRAT S3 Pipeline 验证通过
```
Epoch 1: MSE 1.03 -> 0.06
Epoch 2: MSE 0.06 -> 0.04
Epoch 3: MSE 0.04 -> 0.03
----- Evaluating -----
Epoch 3 Best Of 20: ADE: 0.1350  FDE: 0.0981
Checkpoint: experiments/virat_s3_test/virat_s3_epoch3.pt
```

### 下一步
- [x] 补完 ETH/UCY 复现：hotel / univ ✅
- [ ] 补完 ETH/UCY 复现：zara1 / zara2
- [ ] VIRAT S1/S2/S3 逐场景训练

脚本 `process_virat.py` 已完成，输出到 `processed_data_virat/`：

| 输出文件 | 大小 | 场景 |
|---------|------|------|
| `virat_train.pkl` | 6.6 MB | 5 scenes (s1, s2, s3, s4, s7) |
| `virat_test.pkl` | 821 KB | 3 scenes (s5, s6, s8) |

**训练数据详情**：

| 场景 | 来源 | 人数 | 帧数 @10fps |
|------|------|------|-------------|
| virat_s1 | 000001+002+007+008 合并 | 55 | 17538 |
| virat_s2 | 000200_03+05+00 | 17 | 3615 |
| virat_s3 | 000201_03+05+06+07+04 | 43 | 4430 |
| virat_s4 | 000204_00+04 | 47 | 3478 |
| virat_s7 | 040001_01 | 22 | 6523 |

**脚本核心逻辑**：
1. 从 geom.yml + types.yml 提取 Person 轨迹（跳过 Vehicle/Other）
2. 同一场景多 clip 合并：帧偏移累加 + 全局 Person ID 重新编号
3. 30fps → 10fps 降采样（每 3 帧取 1）
4. 逐场景 standardization（mean=0, std=1）
5. 计算 velocity、acceleration
6. 输出与 ETH 相同的 .pkl 格式（dill）

**训练命令**：
```bash
python main.py --config configs/baseline.yaml --dataset virat
```
