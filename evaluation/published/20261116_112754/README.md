# 已发表实验快照说明

本目录来自2026-11-16既有实验，未在公开版上重新推理或重新调参。真实原图、预览和裁剪不分发。
source统一改为withheld/page_XX.jpg；这只是数据连接标识，不是本地可读图片路径。
保留真值几何、预测框、参数、逐页指标和统计图，支持不读图重算指标。
JSON中的历史代码SHA256和原图SHA256仍对应私有原始实验，不代表公开版修改后代码的摘要。
failure_cases.json中的file和run.json中的裁剪文件名为历史记录，相关图像已省略。
本目录只读归档；新图片的标注请通过网页保存至evaluation/dev_truth.json与test_truth.json。
重算命令：python scripts/check_published_metrics.py。新数据全流程见docs/REPRODUCE.md。
