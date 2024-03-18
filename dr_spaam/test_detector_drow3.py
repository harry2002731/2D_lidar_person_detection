import glob
from dr_spaam.detector import Detector
from dr_spaam.utils.utils import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

path = r"C:\Projects\Python\2D_lidar_person_detection\dr_spaam\datasets\DROWv2-data\\"
ckpt = r'C:\Projects\Python\2D_lidar_person_detection\dr_spaam\logs\20240314_141409_drow\ckpt\ckpt_e40.pth'
# ckpt = r'C:\Projects\Python\2D_lidar_person_detection\dr_spaam\weights\ckpt_jrdb_ann_dr_spaam_e20.pth'
# path = r"..\dr_spaam\datasets\DROWv2-data\train\\"
# ckpt = r'..\dr_spaam\weights\ckpt_jrdb_pl_drow3_e40.pth'
detector = Detector(
    ckpt,
    model="DROW3",          # Or DR-SPAAM
    gpu=False,              # Use GPU
    stride=1,               # Optionally downsample scan for faster inference
    panoramic_scan=False     # Set to True if the scan covers 360 degree
)

# tell the detector field of view of the LiDAR
laser_fov_deg = 180
detector.set_laser_fov(laser_fov_deg)

global line_num ,scans, persons,persons_seqs,scan_seqs
line_num = 0

csv = path + "n-1r_2024-03-18-18-08-31.bag.csv" #修改方括号内数字测试其余数据
scans = load_scan(csv)[2] # load_scan返回值 seqs, times, scans
scan_seqs = load_scan(csv)[0]

# persons = load_dets(path + "n-1r_2015-11-24-18-08-31.bag")[-1]
# persons_seqs = load_dets(path + "n-1r_2015-11-24-18-08-31.bag")[0]
# half_fov_rad = 0.5 * np.deg2rad(laser_fov_deg)
# per_rad = laser_fov_deg/half_fov_rad


def convetTOTorchScript():
    scan = scans[0]
    detector(scan)  # xy坐标位置、检测类别、暂时未知
    if line_num == 0:
        detector.convert()


def update(frame):
    global line_num, persons,persons_seqs
    global scans,scan_seqs

    # 生成随机数据
    for i ,scan in enumerate(scans):
        if line_num == i:
            x_,y_ = scan_to_xy(scan,laser_fov_deg)
            start_time = time.time()
            dets_xy, dets_cls, instance_mask = detector(scan)  # xy坐标位置、检测类别、暂时未知
            # print(1/(time.time()-start_time))
            cls_thresh = 0.3
            cls_mask = dets_cls > cls_thresh
            dets_xy = dets_xy[cls_mask]
            dets_cls = dets_cls[cls_mask]

            line.set_data(x_,y_)
            point.set_data(dets_xy[:,0], dets_xy[:,1])
 
            ax.set_xlim(-10, 10)  # 指定 y 轴的上下限范围
            ax.set_ylim(-10, 10)  # 指定 x 轴的上下限范围
            line_num += 1
            break
        

fig, ax = plt.subplots()
line, = ax.plot([], [], 'b*')
point,  = ax.plot([], [], marker='o', linestyle='None', markersize=20, markerfacecolor='none', markeredgecolor='r')  # 空心圆，蓝色边框
point_people,  = ax.plot([], [], marker='o', linestyle='None', markersize=20, markerfacecolor='none', markeredgecolor='b')  # 空心圆，蓝色边框
ani = FuncAnimation(fig, update, frames=100, interval=350)
plt.show()
# convetTOTorchScript()