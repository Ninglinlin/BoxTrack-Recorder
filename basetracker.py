import numpy as np
import cv2 as cv
import hashlib
import colorsys
from abc import ABC, abstractmethod
from boxmot.utils import logger as LOGGER
from boxmot.utils.iou import AssociationFunction

import time
import datetime
import os


def convert_to_worldtime(timestamp=None):
    """
    将时间戳转换为本地时间字符串
    参数: timestamp - 时间戳（整数/浮点数）。若没有参数，则返回当前时间。
    返回: 本地时间datetime对象
    """
    if timestamp is None:
        # 获取当前时间
        timestamp = time.time()
    else:
        # 如果传入的是毫秒级时间戳，转换为秒
        timestamp = timestamp / 1000.0 if timestamp > 1e12 else timestamp
    
    # 创建UTC时区对象
    utc_tz = datetime.timezone.utc
    # 转换为UTC时间
    utc_time = datetime.datetime.fromtimestamp(timestamp, tz=utc_tz)

    return utc_time
    

def calculate_time_difference(start_time, end_time):
    """
    计算两个北京时间之间的时间差（单位：分钟），end_time > start_time
    Returns:
    - float: 时间差（分钟）
    """
    # 计算时间差并转换为分钟
    delta = end_time- start_time
    minutes = round(delta.total_seconds() / 60)
    return minutes

def checksum(data):
    sum = 0
    for i in range(len(data)):
        sum += ord(data[i])
    checksum = str(sum % 10000).rjust(4, '0')
    return checksum

def _is_point_in_area(observation, area):
        """
        判断观测点是否在指定区域内

        Parameters:
        - observation (tuple): 观测点坐标， (x1, y1, x2, y2)，取中心点
        - area (tuple): 区域坐标，格式为 (x1, y1, x2, y2)

        Returns:
        - bool: 若观测点在区域内返回 True，否则返回 False
        """
        x = (observation[0] + observation[2]) / 2
        y = (observation[1] + observation[3]) / 2
        x1, y1, x2, y2 = area
        return x1 <= x <= x2 and y1 <= y <= y2



class BaseTracker(ABC):
    def __init__(
        self, 
        det_thresh: float = 0.3,
        max_age: int = 50,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_obs: int = 50,
        nr_classes: int = 80,
        per_class: bool = False,
        asso_func: str = 'iou',
        is_obb: bool = False
    ):
        """
        Initialize the BaseTracker object with detection threshold, maximum age, minimum hits, 
        and Intersection Over Union (IOU) threshold for tracking objects in video frames.

        Parameters:
        - det_thresh (float): Detection threshold for considering detections.
        - max_age (int): Maximum age of a track before it is considered lost.
        - min_hits (int): Minimum number of detection hits before a track is considered confirmed.
        - iou_threshold (float): IOU threshold for determining match between detection and tracks.

        Attributes:
        - frame_count (int): Counter for the frames processed.
        - active_tracks (list): List to hold active tracks, may be used differently in subclasses.
        """
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.max_obs = max_obs
        self.min_hits = min_hits
        self.per_class = per_class  # Track per class or not
        self.nr_classes = nr_classes
        self.iou_threshold = iou_threshold
        self.last_emb_size = None
        self.asso_func_name = asso_func+"_obb" if is_obb else asso_func
        self.is_obb = is_obb
        
        self.frame_count = 0
        self.active_tracks = []  # This might be handled differently in derived classes
        self.per_class_active_tracks = None
        self._first_frame_processed = False  # Flag to track if the first frame has been processed
        self._first_dets_processed = False

        self.last_active_track_ids = set()  # 上一帧的 active_tracks id 集合
        self.last_removed_track_ids = set()  # 上一帧的 removed_stracks id 集合
        self.is_maintenance = {}  # 记录目标id检测设备id是否在维护
        self.action_records = {}  # 用于存储动作记录的list
        self.wroten_ids = set()  # 已经写入文件的id集合

        self.station = ''
        self.st = ''
        self.di = ''
        self.devid = ''
        self.file_path = ''
        

        # Initialize per-class active tracks
        if self.per_class:
            self.per_class_active_tracks = {}
            for i in range(self.nr_classes):
                self.per_class_active_tracks[i] = []
        
        if self.max_age >= self.max_obs:
            LOGGER.warning("Max age > max observations, increasing size of max observations...")
            self.max_obs = self.max_age + 5
            print("self.max_obs", self.max_obs)


    @abstractmethod
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        """
        Abstract method to update the tracker with new detections for a new frame. This method 
        should be implemented by subclasses.

        Parameters:
        - dets (np.ndarray): Array of detections for the current frame.
        - img (np.ndarray): The current frame as an image array.
        - embs (np.ndarray, optional): Embeddings associated with the detections, if any.

        Raises:
        - NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("The update method needs to be implemented by the subclass.")
    
    def get_class_dets_n_embs(self, dets, embs, cls_id):
        # Initialize empty arrays for detections and embeddings
        class_dets = np.empty((0, 6))
        class_embs = np.empty((0, self.last_emb_size)) if self.last_emb_size is not None else None

        # Check if there are detections
        if dets.size > 0:
            class_indices = np.where(dets[:, 5] == cls_id)[0]
            class_dets = dets[class_indices]
            
            if embs is not None:
                # Assert that if embeddings are provided, they have the same number of elements as detections
                assert dets.shape[0] == embs.shape[0], "Detections and embeddings must have the same number of elements when both are provided"
                
                if embs.size > 0:
                    class_embs = embs[class_indices]
                    self.last_emb_size = class_embs.shape[1]  # Update the last known embedding size
                else:
                    class_embs = None
        return class_dets, class_embs
    
    @staticmethod
    def setup_decorator(method):
        """
        Decorator to perform setup on the first frame only.
        This ensures that initialization tasks (like setting the association function) only
        happen once, on the first frame, and are skipped on subsequent frames.
        """
        def wrapper(self, *args, **kwargs):
            # If setup hasn't been done yet, perform it
            # Even if dets is empty (e.g., shape (0, 7)), this check will still pass if it's Nx7
            if not self._first_dets_processed:
                dets = args[0]
                if dets is not None:
                    if dets.ndim == 2 and dets.shape[1] == 6:
                        self.is_obb = False
                        self._first_dets_processed = True
                    elif dets.ndim == 2 and dets.shape[1] == 7:
                        self.is_obb = True
                        self._first_dets_processed = True

            if not self._first_frame_processed:
                img = args[1]
                self.h, self.w = img.shape[0:2]
                self.asso_func = AssociationFunction(w=self.w, h=self.h, asso_mode=self.asso_func_name).asso_func

                # Mark that the first frame setup has been done
                self._first_frame_processed = True

            # Call the original method (e.g., update)
            return method(self, *args, **kwargs)
        
        return wrapper
    
    
    @staticmethod
    def per_class_decorator(update_method):
        """
        Decorator for the update method to handle per-class processing.
        """
        def wrapper(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None):
            
            #handle different types of inputs
            if dets is None or len(dets) == 0:
                dets = np.empty((0, 6))
            
            if self.per_class:
                # Initialize an array to store the tracks for each class
                per_class_tracks = []
                
                # same frame count for all classes
                frame_count = self.frame_count

                for cls_id in range(self.nr_classes):
                    # Get detections and embeddings for the current class
                    class_dets, class_embs = self.get_class_dets_n_embs(dets, embs, cls_id)
                    
                    LOGGER.debug(f"Processing class {int(cls_id)}: {class_dets.shape} with embeddings {class_embs.shape if class_embs is not None else None}")

                    # Activate the specific active tracks for this class id
                    self.active_tracks = self.per_class_active_tracks[cls_id]
                    
                    # Reset frame count for every class
                    self.frame_count = frame_count
                    
                    # Update detections using the decorated method
                    tracks = update_method(self, dets=class_dets, img=img, embs=class_embs)

                    # Save the updated active tracks
                    self.per_class_active_tracks[cls_id] = self.active_tracks

                    if tracks.size > 0:
                        per_class_tracks.append(tracks)
                
                # Increase frame count by 1
                self.frame_count = frame_count + 1

                return np.vstack(per_class_tracks) if per_class_tracks else np.empty((0, 8))
            else:
                # Process all detections at once if per_class is False
                return update_method(self, dets=dets, img=img, embs=embs)
        return wrapper


    def check_inputs(self, dets, img):
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img_numpy' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        if self.is_obb:
            assert (
                dets.shape[1] == 7
            ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6 (cx,cy,w,h,angle,conf,cls)"
        else :
            assert (
                dets.shape[1] == 6
            ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6 (x1,y1,x2,y2,conf,cls)"


    def id_to_color(self, id: int, saturation: float = 0.75, value: float = 0.95) -> tuple:
        """
        Generates a consistent unique BGR color for a given ID using hashing.

        Parameters:
        - id (int): Unique identifier for which to generate a color.
        - saturation (float): Saturation value for the color in HSV space.
        - value (float): Value (brightness) for the color in HSV space.

        Returns:
        - tuple: A tuple representing the BGR color.
        """

        # Hash the ID to get a consistent unique value
        hash_object = hashlib.sha256(str(id).encode())
        hash_digest = hash_object.hexdigest()
        
        # Convert the first few characters of the hash to an integer
        # and map it to a value between 0 and 1 for the hue
        hue = int(hash_digest[:8], 16) / 0xffffffff
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Convert RGB from 0-1 range to 0-255 range and format as hexadecimal
        rgb_255 = tuple(int(component * 255) for component in rgb)
        hex_color = '#%02x%02x%02x' % rgb_255
        # Strip the '#' character and convert the string to RGB integers
        rgb = tuple(int(hex_color.strip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Convert RGB to BGR for OpenCV
        bgr = rgb[::-1]
        
        return bgr

    def plot_box_on_img(self, img: np.ndarray, box: tuple, conf: float, cls: int, id: int, thickness: int = 2, fontscale: float = 0.5) -> np.ndarray:
        """
        Draws a bounding box with ID, confidence, and class information on an image.

        Parameters:
        - img (np.ndarray): The image array to draw on.
        - box (tuple): The bounding box coordinates as (x1, y1, x2, y2).
        - conf (float): Confidence score of the detection.
        - cls (int): Class ID of the detection.
        - id (int): Unique identifier for the detection.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with the bounding box drawn on it.
        """
        if self.is_obb:
            
            angle = box[4] * 180.0 / np.pi  # Convert radians to degrees
            box_poly = ((box[0], box[1]), (box[2], box[3]), angle)
            # print((width, height))
            rotrec = cv.boxPoints(box_poly)
            box_poly = np.int_(rotrec)  # Convert to integer

            # Draw the rectangle on the image
            img = cv.polylines(img, [box_poly], isClosed=True, color=self.id_to_color(id), thickness=thickness)

            img = cv.putText(
                img,
                f'id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}, a: {box[4]:.2f}',
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                self.id_to_color(id),
                thickness
            )
        else :

            img = cv.rectangle(
                img,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                self.id_to_color(id),
                thickness
            )
            img = cv.putText(
                img,
                f'id: {int(id)}, conf: {conf:.2f}, c: {int(cls)}',
                (int(box[0]), int(box[1]) - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                fontscale,
                self.id_to_color(id),
                thickness
            )
        return img


    def plot_trackers_trajectories(self, img: np.ndarray, observations: list, id: int) -> np.ndarray:
        """
        Draws the trajectories of tracked objects based on historical observations. Each point
        in the trajectory is represented by a circle, with the thickness increasing for more
        recent observations to visualize the path of movement.

        Parameters:
        - img (np.ndarray): The image array on which to draw the trajectories.
        - observations (list): A list of bounding box coordinates representing the historical
        observations of a tracked object. Each observation is in the format (x1, y1, x2, y2).
        - id (int): The unique identifier of the tracked object for color consistency in visualization.

        Returns:
        - np.ndarray: The image array with the trajectories drawn on it.
        """
        for i, box in enumerate(observations):
            trajectory_thickness = int(np.sqrt(float (i + 1)) * 1.2)
            if self.is_obb:
                img = cv.circle(
                    img,
                    (int(box[0]), int(box[1])),
                    2,
                    color=self.id_to_color(int(id)),
                    thickness=trajectory_thickness 
                )
            else:

                img = cv.circle(
                    img,
                    (int((box[0] + box[2]) / 2),
                    int((box[1] + box[3]) / 2)), 
                    2,
                    color=self.id_to_color(int(id)),
                    thickness=trajectory_thickness
                )
        return img


    def plot_results(self, img: np.ndarray, show_trajectories: bool, thickness: int = 2, fontscale: float = 0.5) -> np.ndarray:
        """
        Visualizes the trajectories of all active tracks on the image. For each track,
        it draws the latest bounding box and the path of movement if the history of
        observations is longer than two. This helps in understanding the movement patterns
        of each tracked object.

        Parameters:
        - img (np.ndarray): The image array on which to draw the trajectories and bounding boxes.
        - show_trajectories (bool): Whether to show the trajectories.
        - thickness (int): The thickness of the bounding box.
        - fontscale (float): The font scale for the text.

        Returns:
        - np.ndarray: The image array with trajectories and bounding boxes of all active tracks.
        """

        # if values in dict
        if self.per_class_active_tracks is not None:
            for k in self.per_class_active_tracks.keys():
                active_tracks = self.per_class_active_tracks[k]
                for a in active_tracks:
                    if a.history_observations:
                        if len(a.history_observations) > 2:
                            box = a.history_observations[-1]
                            img = self.plot_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
                            if show_trajectories:
                                img = self.plot_trackers_trajectories(img, a.history_observations, a.id)
        else:
            for a in self.active_tracks:
                if a.history_observations:
                    if len(a.history_observations) > 2:
                        box = a.history_observations[-1]
                        img = self.plot_box_on_img(img, box, a.conf, a.cls, a.id, thickness, fontscale)
                        if show_trajectories:
                            img = self.plot_trackers_trajectories(img, a.history_observations, a.id)    
                
        return img


    def action_detection(self, DEVICE: dict,time_th: int = 0):
        
        """
        检测目标动作，包括进入/离开或设备检修。

        Parameters:
        - DEVICE (dict): 设备信息字典，格式为 {device_id: {'area': (x1, y1, x2, y2), 'device_name': 'device_name'}}。
        - time_th (int): 设备检修时间阈值，单位为秒。
        - frame_final (bool): 是否是最后一帧。

        Returns:
        - save_vedio (bool): 是否保存到本地文件。
        - save_vedio_tm (int) : 以当前帧开始，保存save_tm段时间内的视频,默认为10s。
        """
        current_active_track_ids = set()
        current_removed_track_ids = set()
        
        # 处理活跃轨迹
        for track in self.active_tracks:            
            if len(track.history_observations) > 2:
            # 为稳定观测，放弃前两帧，会导致产生一些时间差
                track_id = track.id
                current_active_track_ids.add(track_id)
                # 一旦有稳定观测的对象，则保存当前帧
                save_frame = True
                
                # 初始化记录
                if track_id not in self.action_records:
                    self.action_records[track_id] = []
                
                # 新目标进入 
                if track_id not in self.last_active_track_ids:
                    entry_time = track.history_observations_tmsp[-1]
                    self.action_records[track_id].append({
                        'id': track.id,
                        'class': track.cls,
                        'action': 'entry',
                        'start_time': convert_to_worldtime(entry_time),
                        'end_time': None,
                        'duration':0
                    }) 
                    
                # 仅对人类进行设备检修检测       
                if track.cls == 0 :
                    # 初始化维护状态
                    if track_id not in self.is_maintenance:
                        self.is_maintenance[track_id] = {}

                    for device_id, device_info in DEVICE.items():
                        area = device_info['area']
                        device_name = device_info['device_name']

                        # 初始化设备状态
                        if device_id not in self.is_maintenance[track_id]:
                            self.is_maintenance[track_id][device_id] = {
                                "state": False,
                                "start_maintenance_time": None
                            }
                                
                        current_state = self.is_maintenance[track_id][device_id]
                        in_area = _is_point_in_area(track.history_observations[-1], area)

                        # 进入设备区域
                        if in_area and not current_state["state"]:
                            current_state["state"] = True
                            start_time = convert_to_worldtime(track.history_observations_tmsp[-1])
                            current_state["start_maintenance_time"] = start_time
                        # 离开设备区域
                        elif not in_area and current_state["state"]:
                            # 结束维护
                            current_state["state"] = False 
                            start_time = current_state["start_maintenance_time"]
                            maintenance_end_time = convert_to_worldtime(track.history_observations_tmsp[-1])
                            if start_time:  # 确保开始时间存在
                                maintenance_duration = calculate_time_difference(start_time,maintenance_end_time) 
                                # 超过阈值则记录
                                if maintenance_duration > time_th:
                                    self.action_records[track_id].append({
                                        'id': track_id,
                                        'class': 0,
                                        'action': f'detection_{device_id}_{device_name}',
                                        'start_time': start_time,
                                        'end_time': maintenance_end_time,
                                        'duration': maintenance_duration
                                    })
                                
                            # 重置设备状态
                            current_state["start_maintenance_time"] = None

        # 处理被移除的轨迹
        for track in self.removed_stracks:
            if len(track.history_observations) > 2:
                track_id = track.id
                current_removed_track_ids.add(track_id)

                # 新移除的目标
                if track_id not in self.last_removed_track_ids:
                    # 更新进入记录的结束时间
                    if track_id in self.action_records:
                        for record in self.action_records[track_id]:
                            if record['action'] == 'entry' and record['end_time'] is None:
                                end_time = track.history_observations_tmsp[-1]
                                record['end_time'] = convert_to_worldtime(end_time)
                                record['duration'] = calculate_time_difference(record['start_time'],record['end_time'])
                                
                    
        # 写入新移除目标的记录
        new_removed_ids = current_removed_track_ids - self.last_removed_track_ids
        for track_id in new_removed_ids:
            if track_id in self.action_records:
                current_count = len(current_active_track_ids)
                records = self.action_records[track_id]
                try:
                    self.append_dict_to_txt(records, current_count)
                    self.wroten_ids.add(track_id)
                except Exception as e:
                    LOGGER.error(f"写入记录失败: {e}")

        # 更新状态，取集合并集，防止目标闪烁导致的entry重复记录
        self.last_active_track_ids |= current_active_track_ids
        self.last_removed_track_ids |= current_removed_track_ids

        return

    def finalize_records(self):
        unwritten_ids = set(self.action_records.keys()) - self.wroten_ids
        for track_id in unwritten_ids:
            if track_id in self.action_records:
                for record in self.action_records[track_id]:
                    if record['action'] == 'entry' and record['end_time'] is None:
                        record['end_time'] = convert_to_worldtime(int(time.time()))
                        record['duration'] = calculate_time_difference(record['start_time'],record['end_time'])
                    elif record['action'].split('_')[0] == 'detection' and record['end_time'] is None:
                        record['end_time'] = convert_to_worldtime(int(time.time()))
                        record['duration'] = calculate_time_difference(record['start_time'],record['end_time'])
                try:
                    current_count = len(unwritten_ids)
                    self.append_dict_to_txt(self.action_records[track_id], current_count)
                    self.wroten_ids.add(track_id)
                except Exception as e:
                    LOGGER.error(f"最终写入记录失败: {e}")



    def append_dict_to_txt(self, action_dict, current_count):
        """
        将字典中的所有内容以键值对的形式追加到指定的文本文件末尾。

        Args:
                data_dict (dict): 要追加到文件中的字典。
                current_count (int): 当前画面内的目标计数。
                file_name (str): 目标文本文件的名称。
        """
        # 检查当前文件路径是否仍是当天
        current_date = convert_to_worldtime()
        current_date_str = current_date.strftime("%Y%m%d")
        file_name = os.path.basename(self.file_path)
        file_folder = os.path.dirname(self.file_path)
        file_date = file_name.split('_')[2]  # 提取日期部分
        
        # 如果日期变更，则更新文件路径
        if file_date != current_date_str:
            new_base_name = f"{self.station}_securitymonitor_{current_date_str}.txt"
            self.file_path = os.path.join(file_folder, new_base_name)
        
        file_path = self.file_path
        station = self.station
        st = self.st
        di = self.di
        devid = self.devid

        file_temp_path = './home/nvidia/APP/web_NextStation/data/animal/06.txt'
        os.makedirs(os.path.dirname(file_temp_path), exist_ok=True)
        

        try:
            hdev = hnum = hid = stime = etime = hdur = ""  # 初始化所有变量
            # 以追加模式打开文件
            with open(file_path, 'a', encoding='utf-8') as file:
                for record in action_dict:
                    if record['action'] == 'entry':
                        hnum = f",HNUM,{current_count:02d}"
                        hid = f",HID,{record['id']:04d}"
                        stime = record['start_time'].strftime("%Y%m%d%H%M%S")
                        etime = record['end_time'].strftime("%Y%m%d%H%M%S")
                        hdur = f",HDUR,{int(record['duration']):03d},"

                    elif record['action'].split('_')[0] == 'detection':
                        device_id = record['action'].split('_')[1]
                        device_id_2 = str(int(device_id)).zfill(2)
                        hdev += f"{device_id_2}"
                if hdev:
                    line_parts = [hnum, hid, f",STIME,{stime}", f",ETIME,{etime}", hdur, f",HDEV,{hdev}"]  # 添加换行符
                else:
                    line_parts = [hnum, hid, f",STIME,{stime}", f",ETIME,{etime}", hdur] # 添加换行符

                str_staion = f"BG,8{station}"
                str_st = f",{st}"
                str_di = f",{di}"
                str_devid = f",{devid}"
                str_time = f",{stime}"
                s_num = f",{str(len(line_parts)).zfill(3)}"
                device_status = ',00'
                
                weather_reg_data = str_staion + str_st + str_di + str_devid + str_time + s_num + device_status + ''.join(line_parts)

                check_sum = checksum(weather_reg_data)
                weather_reg_data = weather_reg_data+ check_sum + ',ED'

                file.write(weather_reg_data+ '\n')

            with open(file_temp_path, 'w', encoding='utf-8') as tmpfile:
                tmpfile.write(weather_reg_data + '\n')

        except Exception as e:
            print(f"写入文件时出错: {e}")   

    def has_objects(self):
        for a in self.active_tracks:
            if a.history_observations:
                if len(a.history_observations) > 2:
                    return True
        return False