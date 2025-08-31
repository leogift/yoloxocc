import math
import numpy as np

def fov_2_camera_intrinsic(fov_x:float,fov_y:float, \
        pixel_width=1280,pixel_height=720):
    camera_intrinsics = np.eye(3)
    camera_intrinsics[0,0] = (pixel_width/2.0)/math.tan(math.radians(fov_x/2.0))
    camera_intrinsics[0,2] = pixel_width/2.0
    camera_intrinsics[1,1] = (pixel_height/2.0)/math.tan(math.radians(fov_y/2.0))
    camera_intrinsics[1,2] = pixel_height/2.0
    return camera_intrinsics

def camera_intrinsic_2_fov(camera_intrinsics:np.ndarray):
    pixel_width = camera_intrinsics[0,2]*2
    pixel_height = camera_intrinsics[1,2]*2
    fov_x = 2.0*math.atan((pixel_width/2.0)/camera_intrinsics[0,0])*180.0/math.pi
    fov_y = 2.0*math.atan((pixel_height/2.0)/camera_intrinsics[1,1])*180.0/math.pi
    return fov_x, fov_y


if __name__ == '__main__':
    print("fov_2_camera_intrinsic", fov_2_camera_intrinsic(fov_x=102,fov_y=60, \
        pixel_width=1280, pixel_height=720))
    
    camera_intrinsics = np.eye(3)
    camera_intrinsics[0,0] = 200
    camera_intrinsics[0,2] = 256
    camera_intrinsics[1,1] = 200
    camera_intrinsics[1,2] = 119
    print("camera_intrinsic_2_fov", camera_intrinsic_2_fov(camera_intrinsics))
