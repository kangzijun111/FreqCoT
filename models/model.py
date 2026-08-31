import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp

class WindowAttBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num, size, mlp_ratio=4.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num, self.size = num, size
        self.nnorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.nnorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.nmlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)
        self.snorm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.snorm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.smlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

    def forward(self, x):
        B, T, _, D = x.shape
        P, N = self.num, self.size
        x = x.reshape(B, T, P, N, D)
        qkv = self.snorm1(x.reshape(B*T*P,N,D))
        x = x + self.sattn(qkv).reshape(B,T,P,N,D)
        x = x + self.smlp(self.snorm2(x))
        qkv = self.nnorm1(x.transpose(2,3).reshape(B*T*N,P,D))
        x = x + self.nattn(qkv).reshape(B,T,N,P,D).transpose(2,3)
        x = x + self.nmlp(self.nnorm2(x))
        return x.reshape(B,T,-1,D)


class HD_CoTAR_Module(nn.Module):
    def __init__(self, d_model, num_communities, max_nodes):
        super().__init__()
        self.d_model = d_model
        self.num_comm = num_communities
        self.internal_dim = 128 
        self.proj_in = nn.Linear(d_model, self.internal_dim)
        
        self.local_tokens = nn.Parameter(torch.randn(1, 1, num_communities, self.internal_dim))
        self.global_tokens = nn.Parameter(torch.randn(1, 1, 2, self.internal_dim))
        self.spatial_proj = nn.Linear(2, self.internal_dim)
        
        self.local_aggr = Attention(self.internal_dim, num_heads=4, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.global_comm = Attention(self.internal_dim, num_heads=4, qkv_bias=True, attn_drop=0.1, proj_drop=0.1) 
        
        self.norm = nn.LayerNorm(self.internal_dim)
        self.proj_out = nn.Linear(self.internal_dim, d_model)
        self.dropout = nn.Dropout(0.2)

    def init_priors(self, node_coords, partition_map):
        device = self.local_tokens.device
        with torch.no_grad():
            coords = torch.tensor(node_coords).to(device).float()
            for i in range(self.num_comm):
                mask = (torch.tensor(partition_map).to(device) == i)
                if mask.sum() > 0:
                    centroid = coords[mask].mean(0)
                    self.local_tokens.data[0, 0, i] = self.spatial_proj(centroid)
            geo_center = coords.mean(0)
            self.global_tokens.data[0, 0, 0] = self.spatial_proj(geo_center)
            self.global_tokens.data[0, 0, 1] = self.spatial_proj(geo_center)

    def forward(self, x):
        B, T, N, D_in = x.shape
        x_inner = self.proj_in(x) 
        D = self.internal_dim
        C = self.num_comm
        M = N // C 
        
        x_reshaped = x_inner.reshape(B, T, C, M, D)
        loc_t = self.local_tokens.expand(B, T, C, D).reshape(B*T*C, 1, D)
        nodes = x_reshaped.reshape(B*T*C, M, D)
        
        combined = torch.cat([loc_t, nodes], dim=1)
        updated_local = self.local_aggr(combined)[:, 0, :].reshape(B, T, C, D)
        
        gbl_t = self.global_tokens.expand(B, T, 2, D)
        global_in = torch.cat([gbl_t, updated_local], dim=2)
        global_in_flat = global_in.reshape(B*T, -1, D)
        global_out = self.global_comm(global_in_flat).reshape(B, T, -1, D)
        
        refined_local = global_out[:, :, 2:, :].unsqueeze(3)
        out_inner = x_reshaped + refined_local
        out_inner = self.norm(out_inner.reshape(B, T, N, D))
        
        out = self.proj_out(out_inner)
        return self.dropout(out)


class FreqCoT(nn.Module):
    def __init__(self, configs, partition_map=None, node_coords=None):
        super().__init__()
        self.output_len = configs['output_len']
        self.tem_patchsize = configs['tem_patchsize']
        self.tem_patchnum = configs['tem_patchnum']
        self.node_num = configs['node_num']
        self.spa_patchsize = configs['spa_patchsize']
        self.spa_patchnum = configs['spa_patchnum']
        self.tod = configs['tod']
        self.dow = configs['dow']
        self.layers = configs['layers']
        self.factors = configs['factors']
        self.input_len = configs['input_len']
        
        input_dims = configs['input_dims']
        node_dims = configs['node_dims']
        tod_dims = configs['tod_dims']
        dow_dims = configs['dow_dims']
        
        self.register_buffer('ori_parts_idx', torch.tensor(configs['ori_parts_idx'], dtype=torch.long))
        self.register_buffer('idx_reo', torch.tensor(configs['reo_parts_idx'], dtype=torch.long)) 
        self.register_buffer('idx_all', torch.tensor(configs['reo_all_idx'], dtype=torch.long)) 

        dims = input_dims + tod_dims + dow_dims + node_dims
        self.dims = dims

        freq_len = self.input_len // 2 + 1
        self.freq_gate = nn.Parameter(torch.ones(1, freq_len, 1, 1)) 
        nn.init.constant_(self.freq_gate, 5.0) 


        self.input_st_fc = nn.Conv2d(in_channels=3, out_channels=input_dims, 
                                     kernel_size=(1, self.tem_patchsize), stride=(1, self.tem_patchsize), bias=True)
        

        self.high_input_fc = nn.Conv2d(in_channels=4, out_channels=input_dims, 
                                     kernel_size=(1, self.tem_patchsize), stride=(1, self.tem_patchsize), bias=True)

        self.node_emb = nn.Parameter(torch.empty(self.node_num, node_dims))
        nn.init.xavier_uniform_(self.node_emb)
        self.time_in_day_emb = nn.Parameter(torch.empty(self.tod, tod_dims))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(torch.empty(self.dow, dow_dims))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        self.spa_encoder = nn.ModuleList([
            WindowAttBlock(dims, 1, self.spa_patchnum//self.factors, self.spa_patchsize*self.factors, mlp_ratio=1) 
            for _ in range(self.layers)
        ])


        self.hd_cotar = HD_CoTAR_Module(dims, self.spa_patchnum, self.spa_patchsize)
        if node_coords is not None:
            self.hd_cotar.init_priors(node_coords, partition_map)


        self.fusion_gate = nn.Parameter(torch.tensor(5.0))

  
        self.regression_conv = nn.Conv2d(in_channels=self.tem_patchnum*dims, out_channels=self.output_len, kernel_size=(1, 1), bias=True)


    def embedding(self, x, te):
        b, t, n, _ = x.shape
        # x + TE + TE -> 3 channels
        x1 = torch.cat([x, (te[...,0:1]/self.tod), (te[...,1:2]/self.dow)], -1).float()
        input_data = self.input_st_fc(x1.transpose(1,3)).transpose(1,3)
        
        t_patch = input_data.shape[1]
        t_i_d_data = te[:, -t_patch:, :, 0]
        input_data = torch.cat([input_data, self.time_in_day_emb[(t_i_d_data).type(torch.LongTensor)]], -1)
        d_i_w_data = te[:, -t_patch:, :, 1]
        input_data = torch.cat([input_data, self.day_in_week_emb[(d_i_w_data).type(torch.LongTensor)]], -1)
        node_emb = self.node_emb.unsqueeze(0).unsqueeze(1).expand(b, t_patch, -1, -1)
        input_data = torch.cat([input_data, node_emb], -1)
        return input_data


    def embedding_high(self, x_high, x_trend, te):
        b, t, n, _ = x_high.shape

        x1 = torch.cat([x_high, x_trend, (te[...,0:1]/self.tod), (te[...,1:2]/self.dow)], -1).float()

        input_data = self.high_input_fc(x1.transpose(1,3)).transpose(1,3)
        

        t_patch = input_data.shape[1]
        t_i_d_data = te[:, -t_patch:, :, 0]
        input_data = torch.cat([input_data, self.time_in_day_emb[(t_i_d_data).type(torch.LongTensor)]], -1)
        d_i_w_data = te[:, -t_patch:, :, 1]
        input_data = torch.cat([input_data, self.day_in_week_emb[(d_i_w_data).type(torch.LongTensor)]], -1)
        node_emb = self.node_emb.unsqueeze(0).unsqueeze(1).expand(b, t_patch, -1, -1)
        input_data = torch.cat([input_data, node_emb], -1)
        return input_data

    def forward(self, x, te):
        B, T, N, C = x.shape


        with torch.cuda.amp.autocast(enabled=False):
            x_32 = x.float()
            x_fft = torch.fft.rfft(x_32, dim=1, norm='ortho')
            weight = torch.sigmoid(self.freq_gate)
            x_low_fft = x_fft * weight
            x_low_smooth = torch.fft.irfft(x_low_fft, n=T, dim=1, norm='ortho').type_as(x)
            
            x_high_raw = x - x_low_smooth


           
            batch_std = x_high_raw.std() 

            threshold_tensor = 0.5 * batch_std 
            threshold_val = threshold_tensor.item()
            x_high = F.hardshrink(x_high_raw, lambd=threshold_val)
            

            
            x_raw = x 


        emb_low = self.embedding(x_raw, te) 
        rex_low = emb_low[:, :, self.idx_all, :]
        f_low = rex_low
        for block in self.spa_encoder:
            f_low = block(f_low)


        emb_high = self.embedding_high(x_high, x_low_smooth, te) 
        rex_high = emb_high[:, :, self.idx_all, :]
        f_high = self.hd_cotar(rex_high)


        alpha = torch.sigmoid(self.fusion_gate)
        f_fused = alpha * f_low + (1 - alpha) * f_high


        original = torch.zeros(B, f_fused.shape[1], self.node_num, f_fused.shape[-1], device=x.device, dtype=f_fused.dtype)
        original[:, :, self.ori_parts_idx, :] = f_fused[:, :, self.idx_reo, :]
        pred_y = self.regression_conv(original.transpose(2,3).reshape(B, -1, N, 1))

        return pred_y