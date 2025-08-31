import torch
import math

def pack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    S = shapelist[1]
    otherdims = shapelist[2:]
    tensor = torch.reshape(tensor, [B*S]+otherdims)
    return tensor

def unpack_seqdim(tensor, B):
    shapelist = list(tensor.shape)
    BS = shapelist[0]
    otherdims = shapelist[1:]
    S = BS//B
    tensor = torch.reshape(tensor, [B,S]+otherdims)
    return tensor

def meshgrid2d(B, X, Y):
    # returns a meshgrid sized B x X x Y

    grid_x = torch.linspace(0.0, X-1, X)
    grid_x = torch.reshape(grid_x, [1, X, 1])
    grid_x = grid_x.repeat(B, 1, Y)

    grid_y = torch.linspace(0.0, Y-1, Y)
    grid_y = torch.reshape(grid_y, [1, 1, Y])
    grid_y = grid_y.repeat(B, X, 1)

    return grid_x, grid_y
    
def meshgrid3d(B, X, Y, Z):
    # returns a meshgrid sized B x X x Y x Z

    grid_x = torch.linspace(0, X-1, X)
    grid_x = torch.reshape(grid_x, [1, X, 1, 1])
    grid_x = grid_x.repeat(B, 1, Y, Z)

    grid_y = torch.linspace(0, Y-1, Y)
    grid_y = torch.reshape(grid_y, [1, 1, Y, 1])
    grid_y = grid_y.repeat(B, X, 1, Z)

    grid_z = torch.linspace(0, Z-1, Z)
    grid_z = torch.reshape(grid_z, [1, 1, 1, Z])
    grid_z = grid_z.repeat(B, X, Y, 1)

    return grid_x, grid_y, grid_z

def cloudgrid3d(B, X, Y, Z):
    # we want to sample for each location in the grid
    grid_x, grid_y, grid_z = meshgrid3d(B, X, Y, Z)
    x = torch.reshape(grid_x, [B, -1])
    y = torch.reshape(grid_y, [B, -1])
    z = torch.reshape(grid_z, [B, -1])
    # these are B x N
    xyz = torch.stack([x, y, z], dim=2)
    # this is B x N x 3
    return xyz

def gaussian_radius(feature_size, min_overlap=0.985, stride=32):
    height, width = feature_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3) * stride / 32 + 0.5

def special_multiples(input_num, base_num=8):
    multiples = math.ceil(input_num / base_num)
    return int(multiples * base_num)
