# MID 项目工作日志

## 汇总
| 日期 | 会话时长 | 完成内容 | commit |
|------|----------|---------|--------|
| 2026-03-31 | - | 开始工作，创建 plan.md/work.md | - |

## 2026-03-31
- **-** - 开始会话
- **-** - 阅读 essay-mem.md，确认目标：先复现 MID，再集成意图识别

## 2026-03-31
- **21:53** - 开始会话
- **22:15** - 环境检查完成：GPU RTX 4060, PyTorch 2.5.0, NumPy 1.24.3
- **22:30** - 修复 Python 3.11 兼容性：collections.Sequence → collections.abc.Sequence
- **22:35** - 安装 ncls、orjson
- **22:45** - process_data.py 成功运行，6个数据集处理完毕

## 未解决的问题
- stanford 数据集未处理（raw_data/stanford 存在但未在输出中出现）
- 需要确认生成的 pkl 文件位置

## 下一步要做
- [ ] 检查生成的 pkl 文件
- [ ] 尝试训练：python main.py --config configs/... --dataset eth
