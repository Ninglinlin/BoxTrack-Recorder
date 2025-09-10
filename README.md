# 🎥 BoxTrack-Recorder

基于 [Boxmot](https://github.com/BoxCars/BoxMot) 项目的目标跟踪增强版，专为监控场景设计，支持目标出现/消失记录、视频片段保存、区域触发记录以及实时视频流接入。

## 🚀 项目亮点

- ✅ 记录目标出现与消失时间，保存为 `.txt` 文件
- 🎞️ 自动保存目标在监控画面中的视频片段
- 📍 支持区域触发记录：当目标进入指定坐标区域时，记录时间并保存
- 🔴 支持接入实时视频流，实现边追踪边记录
- 🧩 仅修改 Boxmot 中 3 个核心文件，轻量易集成
- 🐍 兼容 Python 3.6.9，适配端侧硬件资源受限设备
- 🛡️ 添加中断保护机制：若程序异常退出或被中止，目标的“消失时间”将自动记录为程序终止时间，确保数据完整性

## 🛠️ 修改文件列表

请将以下文件替换至原始 Boxmot 项目中对应位置：
- boxmot/trackers/bytetrack/bytetrack.py
- boxmot/trackers/basetracker.py 
- tracking/track.py

## 📦 使用说明

1. 克隆原始 [Boxmot](https://github.com/BoxCars/BoxMot) 项目
2. 替换上述三个文件为本项目提供的版本
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
