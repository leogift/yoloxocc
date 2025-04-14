import torch
import numpy as np

def eye_4x4(B):
    rt = torch.eye(4).view(1,4,4).repeat([B, 1, 1])
    return rt

def safe_inverse(a): #parallel version
    B, _, _ = list(a.shape)
    inv = a.clone()
    r_transpose = a[:, :3, :3].transpose(1,2).contiguous()

    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])

    return inv

def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:,:,0:1], device=xyz.device)
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2).contiguous()
    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2).contiguous()
    # xyz2 = xyz2 / xyz2[:,:,3:4]
    xyz2 = xyz2[:,:,:3]
    return xyz2

def split_intrinsics(K):
    # K is B x 3 x 3 or B x 4 x 4
    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0

def merge_intrinsics(fx, fy, x0, y0):
    B = list(fx.shape)[0]
    K = eye_4x4(B)
    K[:,0,0] = fx
    K[:,1,1] = fy
    K[:,0,2] = x0
    K[:,1,2] = y0
    K[:,2,2] = 1.0
    K[:,3,3] = 1.0
    return K

def scale_intrinsics(K, rx, ry):
    fx, fy, x0, y0 = split_intrinsics(K)
    fx = fx*rx
    fy = fy*ry
    x0 = x0*rx
    y0 = y0*ry
    K = merge_intrinsics(fx, fy, x0, y0).to(K.device)
    return K

def translate_intrinsics(K, sx, sy):
    fx, fy, x0, y0 = split_intrinsics(K)
    x0 = x0 - sx
    y0 = y0 - sy
    K = merge_intrinsics(fx, fy, x0, y0).to(K.device)
    return K

# ------------------------------------------------------------------------------------------------------------
def merge_extrinsics_single(r11, r12, r13, t1, r21, r22, r23, t2, r31, r32, r33, t3):
    RT = np.eye(4)
    RT[0,0] = r11
    RT[0,1] = r12
    RT[0,2] = r13
    RT[0,3] = t1
    RT[1,0] = r21
    RT[1,1] = r22
    RT[1,2] = r23
    RT[1,3] = t2
    RT[2,0] = r31
    RT[2,1] = r32
    RT[2,2] = r33
    RT[2,3] = t3
    return RT

def split_intrinsics_single(K):
    # K is 3 x 3
    fx = K[0,0]
    fy = K[1,1]
    x0 = K[0,2]
    y0 = K[1,2]
    return fx, fy, x0, y0

def merge_intrinsics_single(fx, fy, cx, cy):
    K = np.eye(3)
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = cx
    K[1,2] = cy
    K[2,2] = 1.0
    return K

def scale_intrinsics_single(K, rx, ry):
    fx, fy, x0, y0 = split_intrinsics_single(K)
    fx = fx*rx
    fy = fy*ry
    x0 = x0*rx
    y0 = y0*ry
    K = merge_intrinsics_single(fx, fy, x0, y0)
    return K

def translate_intrinsics_single(K, sx, sy):
    fx, fy, x0, y0 = split_intrinsics_single(K)
    x0 = x0 - sx
    y0 = y0 - sy
    K = merge_intrinsics_single(fx, fy, x0, y0)
    return K
