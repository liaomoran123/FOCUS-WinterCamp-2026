# 开发日志
## 1.：环境配置
- 解决`pip install -e .`路径错误：切换到`D:\_desktop\ultralytics`目录执行
- 安装LabelImg标注工具，验证Coco128自动下载功能
## 2.：数据集转换
- 编写`voc2yolo.py`脚本，将VOC XML标注转为YOLO TXT格式
- 修正yaml配置文件的路径转义问题（用`/`替代`\`）
## 3.：模型训练
- 用Coco128验证训练流程，解决`runs`目录不显示问题（原来是我一个放在E盘一个放在D盘了（汗颜））
- 调整训练参数：batch=4（CPU适配），epochs=20
