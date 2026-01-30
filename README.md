# FOCUS-WinterCamp-2026
# FOCUS-WinterCamp-2026: YOLOv11目标检测项目
## 项目简介
基于YOLOv11实现目标检测，支持Coco128/VOC数据集训练，包含数据集转换、模型训练、推理全流程。
## 小心得
- yaml配置文件的'\'要改成正斜杠`/`，冒号后加空格
- 训练命令必须在项目根目录执
- VOC数据集需转换为YOLO格式后再训练
## 运行指南
```bash
# 训练VOC数据集
yolo task=detect mode=train model=yolo11n.pt data=config/voc.yaml epochs=20 batch=4
# 推理测试
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt source=ultralytics/assets/bus.jpg
