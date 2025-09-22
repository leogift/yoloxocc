import torch
import torch.nn.functional as F
from yoloxocc.utils import geom, basic
import random

class VoxUtil():
    def __init__(self, 
                 vox_xyz_size,
                 world_xyz_bounds,
                ):
        
        self.vox_x_size, self.vox_y_size, self.vox_z_size = vox_xyz_size
        self.world_xmin, self.world_xmax, self.world_ymin, self.world_ymax, self.world_zmin, self.world_zmax = world_xyz_bounds

        # voxel坐标系和参考坐标系互转
        self.vox_T_ref = None
        self.ref_T_vox = None

        self.vox_T_ref = self.get_vox_T_ref()
        self.ref_T_vox = self.get_ref_T_vox()
        
        self.gridsample = None
        self.valid_vox = None
        # gridsample
        self.normgridsample = None
        # remapping
        self.matrix = None

    def Ref2Vox(self, xyz_ref):
        # xyz is B x N x 3, in ref coordinates
        B, N, C = list(xyz_ref.shape)
        assert(C==3)
        vox_T_ref = self.vox_T_ref.repeat(B, 1, 1).to(xyz_ref.device)
        xyz_vox = geom.apply_4x4(vox_T_ref, xyz_ref)
        return xyz_vox

    def Vox2Ref(self, xyz_vox):
        # xyz is B x N x 3, in vox coordinates
        B, N, C = list(xyz_vox.shape)
        assert(C==3)
        ref_T_vox = self.ref_T_vox.repeat(B, 1, 1).to(xyz_vox.device)
        xyz_ref = geom.apply_4x4(ref_T_vox, xyz_vox)
        return xyz_ref

    # ------------------------------------------------
    # voxel坐标系和参考坐标系互转
    def get_vox_T_ref(self):
        if self.vox_T_ref is not None:
            return self.vox_T_ref
        
        vox_grid_x_size = (self.world_xmax-self.world_xmin)/float(self.vox_x_size)
        vox_grid_y_size = (self.world_ymax-self.world_ymin)/float(self.vox_y_size)
        vox_grid_z_size = (self.world_zmax-self.world_zmin)/float(self.vox_z_size)

        # translation
        # (this makes the left edge of the leftmost voxel correspond to world_xmin)
        center_T_ref = geom.eye_4x4(1)
        center_T_ref[:,0,3] = -self.world_xmin
        center_T_ref[:,1,3] = -self.world_ymin
        center_T_ref[:,2,3] = -self.world_zmin

        # scaling
        # (this makes the right edge of the rightmost voxel correspond to world_xmax)
        vox_T_center = geom.eye_4x4(1)
        vox_T_center[:,0,0] = 1./vox_grid_x_size
        vox_T_center[:,1,1] = 1./vox_grid_y_size
        vox_T_center[:,2,2] = 1./vox_grid_z_size

        self.vox_T_ref = basic.torch.matmul(vox_T_center, center_T_ref) # 等价于vox_T_center @ center_T_ref

        return self.vox_T_ref

    def get_ref_T_vox(self):
        if self.ref_T_vox is not None:
            return self.ref_T_vox
        
        vox_T_ref = self.get_vox_T_ref()
        # note safe_inverse is inapplicable here,
        # since the transform is nonrigid
        self.ref_T_vox = vox_T_ref.inverse()
        return self.ref_T_vox

    # ------------------------------------------------
    # Pix2Vox 将前视图像素坐标转换到voxel坐标系
    def prepare_Pix2Vox(self, 
                        cam_T_pix, 
                        ref_T_cam,
                        grid_vox,
                        W, H):
        # cam_T_pix is B x 4 x 4
        # ref_T_cam is B x 4 x 4
        # xyz_cam is B x N x 3
        B = grid_vox.shape[0]

        # 将xyz_vox转换到Ref坐标系
        grid_ref = self.Vox2Ref(grid_vox)

        # 将xyz_ref转换到cam坐标系
        grid_cam = geom.apply_4x4(ref_T_cam, grid_ref)

        # 将xyz_cam转换到像素坐标系
        grid_pix = geom.apply_4x4(cam_T_pix, grid_cam)
        norm = torch.unsqueeze(grid_pix[:,:,2], 2)
        grid_pix = grid_pix/torch.clamp(norm, min=1.5e-5)# 16bit到1.5e-5最小值，转换到2D要归一化
        grid_pix_w, grid_pix_h, grid_pix_d = grid_pix[:,:,0], grid_pix[:,:,1], grid_pix[:,:,2]

        # 限制像素坐标在图像内
        w_valid = (grid_pix_w>=0).bool() & (grid_pix_w<W).bool()
        h_valid = (grid_pix_h>=0).bool() & (grid_pix_h<H).bool()
        d_valid = (grid_pix_d>=0).bool()
        valid_vox = (w_valid & h_valid & d_valid).reshape(B, self.vox_x_size, self.vox_y_size, self.vox_z_size)
        
        self.valid_vox = valid_vox.permute(0, 2, 3, 1).contiguous() # B x Y x Z x X
        
        # 保留wh
        gridsample = torch.stack([grid_pix_w, grid_pix_h], axis=2).reshape(B, self.vox_x_size, self.vox_y_size, self.vox_z_size, 2) # B x XxYxZ x 2 原输入是XYZ，这里也只能是XYZ
        self.gridsample = gridsample.permute(0, 2, 3, 1, 4).contiguous() # B x Y x Z x X x 2

        # 以下生成remapping矩阵
        # 获取所有有效索引
        valid_indices = torch.argwhere(self.valid_vox==True)
        # 提取有效位置的bs, y, z, x
        bs_valid = valid_indices[:, 0].int()
        y_valid = valid_indices[:, 1].int()
        z_valid = valid_indices[:, 2].int()
        x_valid = valid_indices[:, 3].int()
        w_values = self.gridsample[bs_valid, y_valid, z_valid, x_valid, 0].int()
        h_values = self.gridsample[bs_valid, y_valid, z_valid, x_valid, 1].int()
        # 使用这些索引来填充trans数组
        self.matrix = torch.zeros(B, H, W, self.vox_y_size, self.vox_z_size, self.vox_x_size).to(self.gridsample.device)
        self.matrix[bs_valid, h_values, w_values, y_valid, z_valid, x_valid] = 1


    def gridsample_Pix2Vox(self,
                        rgb_pix):
        # rgb_pix is B x C x H x W
        B, _, H, W = rgb_pix.shape

        values = None

        if self.normgridsample is None:
            self.normgridsample = torch.zeros(B, self.vox_y_size, self.vox_z_size, self.vox_x_size, 2).to(rgb_pix.device)
        else:
            self.normgridsample.zero_()
        self.normgridsample[..., 0] = 2.0*(self.gridsample[..., 0]/W) - 1.0
        self.normgridsample[..., 1] = 2.0*(self.gridsample[..., 1]/H) - 1.0

        # 对每个高度的平面进行插值
        for y in range(self.vox_y_size):
            normgridsample_layer = self.normgridsample[:,y] # B x Z x X x 2
            
            value = F.grid_sample(rgb_pix, normgridsample_layer, \
                               mode=random.choice(['bilinear', 'nearest']), \
                               align_corners=False) # B x C x Z x X

            value = torch.unsqueeze(value, 2) # B x C x 1 x Z x X
            if values is None:
                values = value
            else:
                values = torch.cat([values, value], dim=2)
        values = values * self.valid_vox.reshape(B, 1, self.vox_y_size, self.vox_z_size, self.vox_x_size).float()
        return self.valid_vox, values.reshape(B, -1, self.vox_z_size*self.vox_x_size) # B x CxY x ZxX


    def remapping_Pix2Vox(self,
                        rgb_pix):
        # rgb_pix is B x C x H x W
        B, C, H, W = list(rgb_pix.shape)

        return self.valid_vox, torch.matmul(
                rgb_pix.reshape(B, C, H*W), \
                self.matrix.reshape(B, H*W, self.vox_y_size*self.vox_z_size*self.vox_x_size)
            ).reshape(B, -1, self.vox_z_size*self.vox_x_size) # B x CxY x ZxX


    # ------------------------------------------------
    # 将Ref坐标系的点投影到voxel坐标系，生成occupancy mask
    def occ_centermask(self, xyz_ref, radius=1.5):
        # xyz is B x N x 3
        # output is B x N x Z x Y x X
        B, N, _ = list(xyz_ref.shape)
        xyz_vox = self.Ref2Vox(xyz_ref)

        # 超出Y的点聚集，vox用格子中心表示
        xyz_vox = torch.round(xyz_vox)

        # BEV方向，X为W方向，Z为H方向
        grid_z, grid_x = basic.meshgrid2d(1, self.vox_z_size, self.vox_x_size)
        grid_z = grid_z.to(xyz_vox.device)
        grid_x = grid_x.to(xyz_vox.device)
        # note the default stack is on -1
        grid_zx = torch.stack([grid_z, grid_x], dim=1).reshape(1, 1, 2, self.vox_z_size, self.vox_x_size)
        # this is 1 x 1 x 2 x Z x X

        batch_centermask = []
        for b in range(B):
            _centermask = []
            for y in range(self.vox_y_size):
                # 取高度为y的平面
                xyz_vox_y = xyz_vox[b, xyz_vox[b,:,1].long()==y]
                if xyz_vox_y.shape[0]>0:
                    vox_zx = torch.stack([xyz_vox_y[:,2], xyz_vox_y[:,0]], dim=1)
                    vox_zx = vox_zx.reshape(1, -1, 2, 1, 1) # 1,N,2,1,1

                    dist = grid_zx - vox_zx # 1,N,2,Z,X
                    # z**2 + x**2
                    dist = torch.sum(dist**2, dim=2, keepdim=False)
                    # this is B x N x Z x X
                    mask = torch.exp(-dist/(2*radius*radius))
                    # 太远的值为0
                    mask[mask < 0.001] = 0.0 # 1,N/Y,Z,X
                    mask = torch.max(mask, dim=1, keepdim=True)[0]

                else:
                    mask = torch.zeros(1, 1, self.vox_z_size, self.vox_x_size)

                mask = mask.to(xyz_ref.device)
                _centermask.append(mask)

            _centermask = torch.cat(_centermask, dim=1)
            batch_centermask.append(_centermask)

        batch_centermask = torch.cat(batch_centermask, dim=0)

        return batch_centermask # B,Y,Z,X
