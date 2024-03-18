import glob
from dr_spaam.detector import Detector
from dr_spaam.utils.utils import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


path = r"C:\Projects\Python\2D_lidar_person_detection\dr_spaam\datasets\DROWv2-data\train\\"
ckpt = r'C:\Projects\Python\2D_lidar_person_detection\dr_spaam\weights\ckpt_jrdb_pl_drow3_e40.pth'
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
laser_fov_deg = 225
detector.set_laser_fov(laser_fov_deg)

global line_num ,scans
line_num = 0

csv = glob.glob(path + "*.csv")[10] #修改方括号内数字测试其余数据
scans = load_scan(csv)[2] # load_scan返回值 seqs, times, scans

def update(frame):
    global line_num
    global scans

    # 生成随机数据
    for i ,scan in enumerate(scans):
        if line_num == i:
            x_,y_ = scan_to_xy(scan,laser_fov_deg)
            dets_xy, dets_cls, instance_mask = detector(scan)  # xy坐标位置、检测类别、暂时未知
            cls_thresh = 0.9
            cls_mask = dets_cls > cls_thresh
            dets_xy = dets_xy[cls_mask]
            dets_cls = dets_cls[cls_mask]

            line.set_data(x_,y_)
            point.set_data(dets_xy[:,0], dets_xy[:,1])

            ax.set_xlim(-6, 6)  # 指定 y 轴的上下限范围
            ax.set_ylim(-6, 6)  # 指定 x 轴的上下限范围
            line_num += 1
            break
        

fig, ax = plt.subplots()
line, = ax.plot([], [], 'b*')
point,  = ax.plot([], [], marker='o', linestyle='None', markersize=20, markerfacecolor='none', markeredgecolor='r')  # 空心圆，蓝色边框
ani = FuncAnimation(fig, update, frames=100, interval=100)
plt.show()