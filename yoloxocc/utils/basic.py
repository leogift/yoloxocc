import torch

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
