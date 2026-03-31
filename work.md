# MID 项目工作日志

## 汇总
| 日期 | 会话时长 | 完成内容 | commit |
|------|----------|---------|--------|
| 2026-03-31 | ~1.5h | 环境搭建、数据处理、git初始化 | ✅ |

## 2026-03-31
- **21:53** - 开始会话
- **22:15** - 环境检查完成：GPU RTX 4060, PyTorch 2.5.0, NumPy 1.24.3
- **22:30** - 修复 Python 3.11 兼容性：collections.Sequence → collections.abc.Sequence
- **22:35** - 安装 ncls、orjson
- **22:45** - process_data.py 成功运行，5个数据集（eth/hotel/univ/zara1/zara2）处理完毕
- **22:50** - git init/commit 完成，创建 dev 分支
- **23:00** - 推送到 GitHub https://github.com/binary-ciao/MID.git (dev分支)

## 下一步要做
- [ ] 尝试训练：`python main.py --config configs/baseline.yaml --dataset eth`
- [ ] 阶段2：复现官方训练，生成 checkpoint
