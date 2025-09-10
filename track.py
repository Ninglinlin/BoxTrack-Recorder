# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
import cv2
import numpy as np
from functools import partial
from pathlib import Path

import torch

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS, SOURCE_HIGH, CLASSES,DEVICE
from boxmot.utils.checks import RequirementsChecker
from tracking.detectors import (get_yolo_inferer, default_imgsz,
                                is_ultralytics_model, is_yolox_model)

# checker = RequirementsChecker()
# checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box

from boxmot.trackers.basetracker import convert_to_worldtime

import signal
import configparser
import sys
import os
import json
import atexit


def get_basic_info():
    device_position_path='/home/nvidia/APP/config/task_info.json'
    # config_file='/home/nvidia/APP/config/config.ini'
    config_file = r'D:\Codes\Next_station\yolo_tracking\config.ini'

    station = ''
    st = ''
    di = ''
    devid = ''
    saverootpath = ''
    cam_ip = ''
    cam_user = ''
    cam_passwd = ''

    if os.path.exists(device_position_path):
        with open(device_position_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    else:
        json_data = {}

    animal_cam_id=''
    for cam_id in json_data.keys():
        one_cam_info= json_data.get(cam_id)
        if 'animal' in list(one_cam_info.keys()):
            animal_cam_id=cam_id
            break

    config = configparser.ConfigParser()
    if os.path.exists(config_file):
        config.read(config_file, encoding='utf-8')
        station = config.get('devinfo', 'qz')
        st = config.get('devinfo', 'st')
        di = config.get('devinfo', 'di')
        devid = config.get('devinfo', 'devid')
        saverootpath = config.get('devinfo', 'saverootpath')

        if animal_cam_id!='':
            for cfg in config:
                if 'camera' in cfg:
                    if animal_cam_id == config.get(cfg, 'cam_id'):
                        cam_ip=config.get(cfg, 'cam_ip')
                        cam_user=config.get(cfg, 'cam_user')
                        cam_passwd=config.get(cfg, 'cam_passwd')

    return station,st,di,devid,saverootpath,animal_cam_id,cam_ip,cam_user,cam_passwd

# 定义全局状态
class GlobalState:
    def __init__(self):
        # 获取基本信息
        self.station, self.st, self.di, self.devid, self.saverootpath, \
        self.animal_cam_id, self.cam_ip, self.cam_user, self.cam_passwd = get_basic_info()
        # 视频流地址
        self.source = f'rtsp://{self.cam_user}:{self.cam_passwd}@{self.cam_ip}/Streaming/Channels/103'
        self.root_folder_path = '.' + os.path.join(self.saverootpath, self.station)
        # 确保路径存在
        os.makedirs(self.root_folder_path, exist_ok=True)
        # 动态路径属性
        self._world_time = None
        self._date_str = None
        
        # 其他状态
        self.yolo_instance = None
        self.video_writer = None
        self.current_video_path = None
        self.last_frame = None

    @property
    def world_time(self):
        return convert_to_worldtime()
    
    @property
    def date_str(self):
        return self.world_time.strftime('%Y%m%d')
    
    @property
    def base_text_folder_path(self):
        """动态生成文本文件夹路径"""
        return os.path.join(self.root_folder_path, 'data')
    
    @property
    def base_video_folder_path(self):
        """动态生成视频文件夹路径"""
        path1 = os.path.join(self.root_folder_path, 'video')
        path2 = os.path.join(path1, self.animal_cam_id)
        return path2
    
    @property
    def daily_video_folder_path(self):
        """动态生成每日视频文件夹路径"""
        path1 = self.base_video_folder_path
        path2 = os.path.join(path1, self.date_str)
        return path2
    
    @property
    def base_txt_file_path(self):
        """动态生成文本文件路径"""
        os.makedirs(self.base_text_folder_path, exist_ok=True)
        filename = f"{self.station}_securitymonitor_{self.date_str}.txt"
        return os.path.join(self.base_text_folder_path, filename)
    
    def update_time_cache(self):
        """更新时间缓存，用于处理跨日期情况"""
        self._world_time = None
        self._date_str = None
    
# 创建全局状态实例
global_state = GlobalState()

def handle_exit(signum=None):
    """退出处理函数"""
    print(f"\nSignal {signum} received. Saving data...")
    
    if global_state.yolo_instance is not None:
        try:
            # 确保 trackers 存在且非空
            if hasattr(global_state.yolo_instance.predictor, 'trackers') and global_state.yolo_instance.predictor.trackers:
                global_state.yolo_instance.predictor.trackers[0].finalize_records()
                print("Data saved successfully.")
            else:
                print("No trackers available to save data.")
        except Exception as e:
            print(f"Error during save: {str(e)}")
    else:
        print("YOLO instance not available for saving.")
    
    # 关闭视频写入器
    if global_state.video_writer is not None:
        try:
            global_state.video_writer.release()
            print(f"Video writer closed for {global_state.current_video_path}")
        except Exception as e:
            print(f"Error closing video writer: {str(e)}")

    sys.exit(0)
    

def handle_exception(exc_type, exc_value, exc_traceback):
    """未处理异常捕获"""
    print(f"Unhandled exception: {exc_value}")
    handle_exit()  # 重用退出处理逻辑
    sys.__excepthook__(exc_type, exc_value, exc_traceback)



def start_video_recording():
    """
    开始视频录制
    
    参数:
        video_name (str): 视频名称（不含扩展名）
        duration_minutes (int): 录制时长（分钟），默认为1分钟
    """
    # 确保有帧可用于获取尺寸
    if not hasattr(global_state, 'last_frame') or global_state.last_frame is None:
        print("No frame available for video recording")
        return False
    # 创建视频文件夹
    os.makedirs(global_state.daily_video_folder_path, exist_ok=True)
    
    video_name = 'Z_SURF_I' + '_' + global_state.station + '_' + global_state.world_time.strftime("%Y%m%d%H%M%S") + '_' + 'O' + '_' + 'AWS_WLRD-' + global_state.animal_cam_id + '-' + '06'

    video_path = os.path.join(global_state.daily_video_folder_path, f"{video_name}.mp4")
    global_state.current_video_path = video_path
    
    # 获取帧尺寸
    h, w = global_state.last_frame.shape[:2]
    
    # 设置帧率
    fps = 15
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    
    if not video_writer.isOpened():
        print(f"Failed to open video writer for {video_path}")
        return False
    
    # 设置全局状态
    global_state.video_writer = video_writer

    print(f"Started recording video: {video_path}")
    return True

def write_video_frame(frame):
    """写入视频帧"""
    if global_state.video_writer is None:
        return False
        
    try:
        global_state.video_writer.write(frame)

        return True
    except Exception as e:
        print(f"Video write error: {e}")
        # 确保异常时释放资源
        try:
            global_state.video_writer.release()
        except:
            pass
        global_state.video_writer = None
        global_state.current_video_path = None
        return False
    

def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = TRACKER_CONFIGS / (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


@torch.no_grad()
def run(args):
    
    global global_state
    # args.source = global_state.source
    first_run = True
    consecutive_false_count = 0  # Counter for consecutive False saves
    max_false_frames = 30  # Maximum number of consecutive frames to wait for no detection


    if args.imgsz is None:
        args.imgsz = default_imgsz(args.yolo_model)
    # 创建 YOLO 实例并存储在全局状态中
    global_state.yolo_instance = YOLO(
        args.yolo_model if is_ultralytics_model(args.yolo_model)
        else 'yolov8n.pt',
    )
    # 立即注册信号处理（此时 yolo 实例已存在）
    signal.signal(signal.SIGTERM, handle_exit)  # kill默认信号
    signal.signal(signal.SIGINT, handle_exit)   # Ctrl+C
    atexit.register(handle_exit)                # 正常退出
    sys.excepthook = handle_exception           # 未处理异常

    results = global_state.yolo_instance.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=False,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    global_state.yolo_instance.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if not is_ultralytics_model(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        yolo_model = m(model=args.yolo_model, device=global_state.yolo_instance.predictor.device,
                       args=global_state.yolo_instance.predictor.args)
        global_state.yolo_instance.predictor.model = yolo_model

        # If current model is YOLOX, change the preprocess and postprocess
        if not is_ultralytics_model(args.yolo_model):
            # add callback to save image paths for further processing
            global_state.yolo_instance.add_callback(
                "on_predict_batch_start",
                lambda p: yolo_model.update_im_paths(p)
            )
            global_state.yolo_instance.predictor.preprocess = (
                lambda imgs: yolo_model.preprocess(im=imgs))
            global_state.yolo_instance.predictor.postprocess = (
                lambda preds, im, im0s:
                yolo_model.postprocess(preds=preds, im=im, im0s=im0s))

    # store custom args in predictor
    global_state.yolo_instance.predictor.custom_args = args
    try:
         # 记录上一帧的日期，用于检测日期变化
        prev_date = global_state.date_str

        for r in results:
            # 检查日期是否变化（跨天运行）
            current_date = global_state.date_str
            tracker = global_state.yolo_instance.predictor.trackers[0]
            if current_date != prev_date:
                print(f"Date changed from {prev_date} to {current_date}")
                prev_date = current_date
                # 更新tracker中的文件路径
                tracker.file_path = global_state.base_txt_file_path
            if first_run:
                tracker.station = global_state.station
                tracker.st = global_state.st
                tracker.di = global_state.di
                tracker.devid = global_state.devid
                tracker.saverootpath = global_state.saverootpath
                tracker.file_path = global_state.base_txt_file_path
                first_run = False
            
            # 处理跟踪结果
            img = tracker.plot_results(r.orig_img, args.show_trajectories)
            # 保存当前帧用于可能的视频录制
            global_state.last_frame = img.copy()
            # 事件检测
            tracker.action_detection(DEVICE)
            save_frame = tracker.has_objects()
            
            # 事件开始：启动录制
            if args.save_video and save_frame and global_state.video_writer is None:
                start_video_recording()
            
            # 事件结束：停止录制
            if not save_frame and global_state.video_writer is not None:
                consecutive_false_count += 1
                if consecutive_false_count >= max_false_frames:
                    try:
                        global_state.video_writer.release()
                        print(f"Video writer closed for {global_state.current_video_path}")
                    except Exception as e:
                        print(f"Error closing video writer: {str(e)}")
                    finally:
                        global_state.video_writer = None
                        global_state.current_video_path = None

                        print("Recording stopped after 30 frames of no detection.")
                else:
                    pass
                    # print(f"Waiting for {max_false_frames - consecutive_false_count} more frames to confirm no detection...")
            else:
                consecutive_false_count = 0  # Reset the counter if an object is detected again

            if global_state.video_writer is not None:
                write_video_frame(img)
            
            # 如果正在录制，写入当前帧
            if global_state.video_writer is not None:
                write_video_frame(img)
                
            if args.show is True:
                cv2.imshow('BoxMOT', img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') or key == ord('q'):
                    break
    finally:
        # 确保在正常结束时也保存数据
        tracker.finalize_records()
        
        # 关闭视频写入器
        if global_state.video_writer is not None:
            try:
                global_state.video_writer.release()
                print(f"Closed video writer for {global_state.current_video_path}")
            except Exception as e:
                print(f"Error closing video writer: {str(e)}")

def parse_opt():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov10n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='bytetrack',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc, boosttrack')
    parser.add_argument('--source', type=str, default= SOURCE_HIGH,
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1920, 1080],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',default=False,
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',default=False,
                        help='save video tracking results')
    parser.add_argument('--classes', nargs='+', type=int, default= CLASSES,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--show-trajectories', action='store_true',
                        help='show confidences')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--save-video', type=bool, default=True,
                        help='save device maintenance information')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":

    opt = parse_opt()
    run(opt)

