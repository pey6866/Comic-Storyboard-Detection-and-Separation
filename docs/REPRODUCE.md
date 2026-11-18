# 复现与调试

## 环境与入口

Python 3.9.13，直接依赖：numpy1.26.4、Pillow11.3.0、opencv-python-headless4.11.0.86、onnxruntime1.19.2。
完整安装命令见根目录README。使用CPUExecutionProvider，输入RGB/float32/NCHW/640等比留白。
模型来源、固定提交和SHA256见models/manifest.json；没有自动升级模型。
网页为127.0.0.1:8877，标注为/annotate；输出按任务隔离。可选PANEL_MODEL_PATH环境变量指定已有本地权重，
只用于网页和网页集成测试；CLI使用--model；整套历史实验脚本使用models/best.onnx。

## 不读原图复算历史指标

执行python scripts/check_published_metrics.py，会用已保存的真值/预测框重新匹配，核对3种方法×3个IoU阈值，
并重算开发参数选择与宏平均。该操作不加载模型、不读取漫画，不重新测量速度。
快照source改成withheld/page_XX.jpg，保持连接一致；这不是实际文件路径。
原实验源图及代码摘要用于溯源，不能当作公开版修改后代码摘要；公开包另有发布清单。

## 新数据完整实验

1. 将自己有权使用的图片放入漫画目录，启动网页，打开/annotate。
2. 在任何阈值试验前选定dev/test；建议按作品隔离。原开发后缀_03/_06/_08/_09/_11/_12锁定dev。
3. 手动绘制全部分镜，从空白开始，确认整页后勾选核验保存。不要将预测复制为真值。
4. 两个划分均有核验记录后执行python scripts/run_evaluation.py。
5. 脚本先审计数据、在dev比较15组配置并冻结，然后评估test；结果写evaluation/runs新目录。

新数据结果不应预期等于历史0.9939。不要将published里的脱敏快照复制为本机标注，那里没有可读原图。

## 测试与调试样例

python -m unittest discover -s tests -v：坐标、一对一匹配、异常导出、网页及标注8项。
未下载模型时实际推理网页测试跳过，其他7项运行；有权重时8项全部运行。
python scripts/create_debug_samples.py：生成三格、空白、损坏图片。
python panel_assistant.py evaluation/synthetic --detector baseline：不需模型，损坏图片导致预期返回1。
python panel_assistant.py evaluation/synthetic：需权重，同样保留故意损坏失败；只验证流程。
样例目录、网页输出及新人工标注均由.gitignore排除。公开静态样例见examples。

## 关键代码

panel_assistant.py：letterbox、restore、iou、Inkwell.detect、Baseline.detect、process、draw_panels。
web_server.py：WorkflowAgent.run、Handler.do_GET/do_POST、main。
annotations.py：AnnotationStore.pages/load/save；evaluate.py：match、metrics。
scripts/run_evaluation.py：score、main；scripts/check_published_metrics.py：历史数值一致性核对。

## 限制

未做训练、OCR、翻译、异形分割或阅读顺序推断。粗框和单册测试不能证明普遍精度。
模型卡还说明整页splash曾按负样本训练；本项目将第18页标为1个分镜，与预训练标注政策可能不同，
这也是整页误差需后续验证的因素，不把它当作已证实的因果解释。
