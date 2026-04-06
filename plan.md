# MID 项目工作计划

## 背景
复现 CVPR 2022 论文 "Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion"，后期替换数据集为 VIRAT，最终集成 Qwen2-VL 意图识别（见 essay-mem.md）。

## 目标
- [x] 复现 MID 论文官方代码，训练出基线模型 ✅
- [x] 在 ETH/UCY 数据集上验证效果 ✅ (5/5 完成)
- [ ] 替换为 VIRAT 数据集
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
- 2026-04-06: 完成！ETH ADE=1.536, HOTEL ADE=0.957, UNIV ADE=0.862, ZARA1 ADE~0.99, ZARA2 ADE=1.038
- 预期产出：ADE/FDE 指标输出
- 完成标准：指标与论文相近

### 阶段4：替换数据集为 VIRAT
- 预期产出：VIRAT 数据训练正常
- 完成标准：用你的 VIRAT 数据集替代 ETH/UCY

### 阶段5：集成意图识别（后期）
- 预期产出：MLLM 意图 → MID 生成
- 完成标准：完整链路打通

## checkpoint
最后活动：2026-04-06 23:50
状态：阶段2/3完成！ETH/UCY 5/5 训练全部完成，准备进入阶段4（VIRAT）或收尾
