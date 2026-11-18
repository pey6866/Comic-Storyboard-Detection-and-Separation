# 许可与贡献归属

本公开版的项目自编源码、合成测试图片及说明材料按GNU AGPL v3发布，完整条款见LICENSE。
此选择用于本次公开发布，不改变第三方组件原有许可，也不授予漫画素材的分发权。

Inkwell检测权重由cedarrapidsboy提供，固定提交fa7afba52ba7dc7b0840c7702b767972b5e17a9b。
模型卡正文将检测权重标为AGPL-3.0；其阅读顺序模型使用不同许可，本项目没有使用。
权重不随本仓库分发，通过scripts/download_model.py下载并验证SHA256。
来源：https://huggingface.co/cedarrapidsboy/inkwell-panel-models 。模型卡原文见docs/model_card.md。

yolo-inference skill来自Ultralytics/skills，固定提交11e45578024f8bcfdcf1ff3faf2d9e8a80e3c919。
完整原文和上游AGPL v3许可证保留在docs/skill_reuse/upstream；文件未修改，摘要见source_manifest.json。
来源：https://github.com/ultralytics/skills 。作为开发指导适配，没有运行时Markdown插件加载。

运行依赖沿用各自许可：ONNX Runtime 1.19.2（MIT）、OpenCV 4.11.0.86（Apache-2.0）、
NumPy 1.26.4（BSD-3-Clause）、Pillow 11.3.0（HPND）。pip安装的完整第三方包及许可以各发行包为准。
仓库不捆绑依赖二进制；完整版本见requirements-lock.txt。

项目代码与文档由Codex辅助编写、调试与整理；使用者确定需求、提供资料并完成真实人工标注。
模型预训练属于原作者，不计作本项目自行训练。公开版新增了合成网页测试、发布审计和快照复算入口。
不包含漫画原图、裁剪、漫画对照图、私人封面信息、API凭据或个人环境。
