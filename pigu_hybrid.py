import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# ==========================================
# 1. 模型定义
# ==========================================

class HybridEmbedding(nn.Module):
    def __init__(self, in_dim=2, mapping_size=64, high_scale=12.0):
        super().__init__()
        half_dim = mapping_size // 2
        self.raw_proj = nn.Linear(in_dim, half_dim)
        self.register_buffer('B_high', torch.randn(in_dim, half_dim) * high_scale)

    def forward(self, x):
        feat_smooth = F.silu(self.raw_proj(x)) 
        proj_high = 2 * np.pi * x @ self.B_high
        feat_detail = torch.cat([torch.sin(proj_high), torch.cos(proj_high)], dim=-1)
        return feat_smooth, feat_detail

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class GraphFeatureEncoder(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(in_features, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        h = self.mlp1(x) 
        h_attn, _ = self.attn(h, h, h)
        h = self.norm(h + h_attn)
        return h

class SpatiallyModulatedHybridAttention(nn.Module):
    def __init__(self, coord_dim=2, feature_dim=64):
        super().__init__()
        self.sensor_embedding = HybridEmbedding(in_dim=coord_dim, mapping_size=64, high_scale=12.0)
        self.key_proj_smooth = nn.Linear(32, feature_dim)
        self.key_proj_detail = nn.Linear(64, feature_dim)
        
        self.query_mlp = nn.Sequential(
            nn.Linear(coord_dim, 64), nn.SiLU(), nn.Linear(64, feature_dim * 2)
        )
        self.scale = feature_dim ** -0.5
        self.transition_x = 0.25 

    def forward(self, grid_coords, sensor_coords, sensor_feats):
        B, H, W, _ = grid_coords.shape
        flat_grid = grid_coords.view(B, -1, 2)
        
        Q_total = self.query_mlp(flat_grid)
        Q_smooth = Q_total[..., :64]
        Q_detail = Q_total[..., 64:]
        
        grid_x = flat_grid[..., 0:1]
        beta = torch.sigmoid(50.0 * (grid_x - self.transition_x))
        Q_detail = Q_detail * beta
        
        k_raw_smooth, k_raw_detail = self.sensor_embedding(sensor_coords)
        K_smooth = self.key_proj_smooth(k_raw_smooth)
        K_detail = self.key_proj_detail(k_raw_detail)
        
        score_smooth = torch.matmul(Q_smooth, K_smooth.transpose(-2, -1))
        score_detail = torch.matmul(Q_detail, K_detail.transpose(-2, -1))
        
        attn_logits = (score_smooth + score_detail) * self.scale
        attn = F.softmax(attn_logits, dim=-1)
        
        out = torch.matmul(attn, sensor_feats)
        return out.view(B, H, W, -1).permute(0, 3, 1, 2)

class PIGU_Hybrid(nn.Module):
    def __init__(self, sensor_in_dim=3, sensor_count=65, hidden_dim=64, out_dim=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sensor_encoder = GraphFeatureEncoder(sensor_in_dim + 2, hidden_dim)
        self.projector = SpatiallyModulatedHybridAttention(coord_dim=2, feature_dim=hidden_dim)
        
        self.in_conv = DoubleConv(hidden_dim + 2 + hidden_dim + 3, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up1 = DoubleConv(256 + 128, 128)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up2 = DoubleConv(128 + 64, 64)
        self.out_conv = nn.Conv2d(64, out_dim, kernel_size=1)

    def forward(self, sensor_vals, sensor_coords, grid_coords, base_flow):
        B = sensor_vals.shape[0]
        H, W = grid_coords.shape[1], grid_coords.shape[2]
        node_in = torch.cat([sensor_vals, sensor_coords], dim=-1)
        node_feats = self.sensor_encoder(node_in) 
        global_context = torch.mean(node_feats, dim=1) 
        global_map = global_context.view(B, self.hidden_dim, 1, 1).expand(-1, -1, H, W)
        
        grid_feats = self.projector(grid_coords, sensor_coords, node_feats) 
        coords_map = grid_coords.permute(0, 3, 1, 2) 
        x = torch.cat([grid_feats, coords_map, global_map, base_flow], dim=1) 
        
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up1(x)
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up2(x)
        
        fluc_out = self.out_conv(x)
        return fluc_out + base_flow

# ==========================================
# 2. 提取脚本
# ==========================================

try:
    from graph_net.torch.extractor import extract
except ImportError:
    print("Warning: graph_net not found. Using dummy extractor.")
    def extract(name):
        def decorator(model):
            return model
        return decorator

def run_model(name: str = "pigu_hybrid_pinn", device_str: str = "cpu") -> None:
    device = torch.device(device_str)
    print(f"\n[1/3] Instantiating model: {name} on {device_str}")

    model = PIGU_Hybrid(sensor_in_dim=3, sensor_count=65, hidden_dim=64, out_dim=3)
    model = model.to(device).eval()

    print("[2/3] Constructing inputs...")
    B = 1
    H, W = 128, 256  
    sensor_count = 65
    sensor_in_dim = 3
    
    sensor_vals = torch.randn(B, sensor_count, sensor_in_dim, device=device)
    sensor_coords = torch.randn(B, sensor_count, 2, device=device)
    grid_coords = torch.randn(B, H, W, 2, device=device)
    base_flow = torch.randn(B, 3, H, W, device=device)

    print(f"      Input Shapes: {sensor_vals.shape}, {sensor_coords.shape}, {grid_coords.shape}, {base_flow.shape}")

    print("[3/3] Extracting graph...")
    
    extractor_wrapper = extract(name=name)(model)
    
    try:
        if not os.path.exists("./output"):
            os.makedirs("./output", exist_ok=True)
            
        with torch.no_grad():
            extractor_wrapper(sensor_vals, sensor_coords, grid_coords, base_flow)
            
        print(f"\n[SUCCESS] {name} graph extracted successfully!")
        
    except Exception as e:
        print(f"\n[FAIL] Extraction Error: {e}")
        raise e

if __name__ == "__main__":
    run_model()
