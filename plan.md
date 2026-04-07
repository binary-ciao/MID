# MID 项目工作计划

## 背景
复现 CVPR 2022 论文 "Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion"，后期替换数据集为 VIRAT，最终集成 Qwen2-VL 意图识别（见 essay-mem.md）。

## 目标
- [x] 复现 MID 论文官方代码，训练出基线模型 ✅
- [x] 在 ETH/UCY 数据集上验证效果 ✅ (5/5 完成)
- [x] 替换为 VIRAT 数据集 ✅ (S1/S2/S3 完成)
- [ ] 集成意图识别模块

## 里程碑

### 阶段1：环境搭建与数据准备 ✅
- 预期产出：数据集处理完毕，可训练
- 完成标准：运行 `python process_data.py` 无报错
- 2026-03-31: 完成！processed_data_noise/ 包含 5 个数据集的 train/val/test pkl 文件

### 阶段2：复现官方训练 ✅
- 预期产出：checkpoint 生成，loss 下降
- 完成标准：完成至少一个 dataset 的完整训练
- 2026-04-06: 完成！ETH/UCY 5/5 数据集全部完成 90 epoch 训练

### 阶段3：复现官方评估 ✅
- 预期产出：ADE/FDE 指标输出
- 完成标准：指标与论文相近
- 2026-04-07: 完成！DDIM 采样：ETH ADE=0.460, HOTEL ADE=0.163, UNIV ADE=0.222, ZARA1 ADE=0.216, ZARA2 ADE=0.175, VIRAT S1 ADE=0.012, S2 ADE=0.009, S3 ADE=0.168

### 阶段4：替换数据集为 VIRAT ✅
- 预期产出：VIRAT 数据训练正常
- 完成标准：用你的 VIRAT 数据集替代 ETH/UCY
- 2026-04-07: 完成！VIRAT S1/S2/S3 全部完成 90 epoch 训练

### 阶段5：集成意图识别（后期）
- 预期产出：MLLM 意图 → MID 生成
- 完成标准：完整链路打通

## checkpoint
最后活动：2026-04-07
状态：全部完成！ETH/UCY 5/5 + VIRAT S1/S2/S3 训练完毕，所有模型 DDIM 评估完成，待上传 git
