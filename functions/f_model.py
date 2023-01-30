import torch
import torch.nn as nn
import open3d as o3d
import numpy as np
from dgl.geometry import farthest_point_sampler


def index_pc(pc, indx):
        device = pc.device
        B = pc.shape[0]
        view_shape = list(indx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(indx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_pc = pc[batch_indices, indx, :]
        return new_pc

def compute_dist_mat(pc1, pc2):
    B, N, _ = pc1.shape
    _, M, _ = pc2.shape
    dist = -2 * torch.matmul(pc1, pc2.permute(0, 2, 1))
    dist += torch.sum(pc1 ** 2, -1).view(B, N, 1)
    dist += torch.sum(pc2 ** 2, -1).view(B, 1, M)
    return dist

def query_ball_point(nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    sqrdists = compute_dist_mat(new_xyz, xyz)
    group_idx = torch.topk(sqrdists,nsample,-1,False)[1]
    return group_idx

class PCup(nn.Module):
    def __init__(self, num_coarse=64, latent_dim=128,grid_size=2):
        super(PCup, self).__init__()
        self.latent_dim = latent_dim
        self.num_coarse = num_coarse
        self.grid_size = grid_size
        # self.num_coarse = self.num_dense // (self.grid_size ** 2)
        self.num_dense = self.num_coarse * (self.grid_size ** 2)

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128,self.latent_dim,1)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(self.latent_dim + 3 + 2, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128,self.latent_dim,1)
        )
        # self.final_linear = nn.Sequential(
        #     nn.Linear(3,64),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(64,256)
        # )

        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2)    #.cuda()  # (1, 2, S)

    def forward(self,coarse):

        feature_global = torch.max(coarse,1)[0]
        B = coarse.shape[0]

        coarse = self.first_conv(coarse.transpose(2,1)).transpose(2,1).contiguous()
        point_idx = farthest_point_sampler(coarse, self.num_coarse)
        coarse_f = index_pc(coarse,point_idx)
        
        point_feat = coarse_f.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, self.latent_dim).transpose(2, 1)               # (B, 3, num_fine)
        
        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)
        
        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, latent_dim, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, latent_dim+2+3, num_fine)

        final = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud

        return final.transpose(2,1).contiguous(), coarse


class PCdown(nn.Module):
    def __init__(self, input_dim=3, output_dim=64, k=5, type='enc1', reduction=True):
        super(PCdown, self).__init__()
        self.input_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k = k
        self.mid_dim = input_dim//2 if type == 'enc2' else output_dim//2
        self.reduction = reduction

        self.first_mlp = nn.Sequential(
            nn.Conv1d(input_dim, self.mid_dim,1),
            nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.mid_dim, output_dim, 1)
        )
        self.last_mlp = nn.Sequential(
            nn.Conv1d(output_dim+input_dim, output_dim, 1),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(output_dim, output_dim, 1),
        )

    @staticmethod
    def sample_and_group(npoint, nsample, xyz, points, returnfps=False):
        """
        Input:
            npoint:
            radius:
            nsample:
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, npoint, nsample, 3]
            new_points: sampled points data, [B, npoint, nsample, 3+D]
        """
        B, N, C = xyz.shape
        S = npoint
        fps_idx = farthest_point_sampler(xyz, npoint).long() # [B, npoint, C]
        new_xyz = index_pc(xyz, fps_idx)
        idx = query_ball_point(nsample, xyz, new_xyz)
        grouped_xyz = index_pc(xyz, idx) # [B, npoint, nsample, C]
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

        if points is not None:
            grouped_points = index_pc(points, idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
        else:
            new_points = grouped_xyz_norm
        if returnfps:
            return new_xyz, new_points, grouped_xyz, fps_idx
        else:
            return new_xyz, new_points
        
    def forward(self,x_in):
        if self.reduction:
            x = self.first_mlp(x_in.transpose(1,2)).transpose(1,2).contiguous()
            _,x_cat = self.sample_and_group(x_in.shape[1]//2,self.k,x_in,x)
            x = torch.max(x_cat,dim=-2)[0].transpose(1,2).contiguous()
        else:
            x = self.first_mlp(x_in.transpose(1,2))
            x = torch.cat([x,x_in.transpose(1,2)],dim=1)
        
        x = self.last_mlp(x)
        return x.transpose(1,2).contiguous()


class PCRegister(nn.Module):
    def __init__(self):
        super(PCRegister, self).__init__()
        self.expansion = PCup()
        self.contraction1 = PCdown(128,512)
        self.contraction2 = PCdown(512,128,reduction=False)
        self.contraction3 = PCdown(128,3,reduction=False)

        # self.dec_mlp = nn.Sequential(
        #     nn.Conv1d(512, 256,1),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(256, 64,1),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(64, 3, 1)
        # )

    def forward(self,x):
        x, coarse = self.expansion(x)
        print('expanded size',x.size())
        x = self.contraction1(x)
        print('contracted size',x.size())
        x_f = torch.matmul(coarse,x)

        x = self.contraction2(x)
        x = self.contraction3(x)
        # x = self.dec_mlp(x.transpose(1,2))


        return x,x_f

