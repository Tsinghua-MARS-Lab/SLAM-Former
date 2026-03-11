# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Modified: Added RoPE3D_ChunkAware and PositionGetter3D_ChunkAware for chunk-aware position encoding
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------



import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, n_cls_token=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [n_cls_token+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if n_cls_token>0:
        pos_embed = np.concatenate([np.zeros([n_cls_token, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


#----------------------------------------------------------
# RoPE2D: RoPE implementation in 2D
#----------------------------------------------------------

try:
    from models.curope import cuRoPE2D
    RoPE2D = cuRoPE2D
except ImportError:
    print('Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead')

    class RoPE2D(torch.nn.Module):
        
        def __init__(self, freq=100.0, F0=1.0):
            super().__init__()
            self.base = freq 
            self.F0 = F0
            self.cache = {}

        def get_cos_sin(self, D, seq_len, device, dtype):
            if (D,seq_len,device,dtype) not in self.cache:
                inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
                t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
                freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
                freqs = torch.cat((freqs, freqs), dim=-1)
                cos = freqs.cos() # (Seq, Dim)
                sin = freqs.sin()
                self.cache[D,seq_len,device,dtype] = (cos,sin)
            return self.cache[D,seq_len,device,dtype]
            
        @staticmethod
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
            
        def apply_rope1d(self, tokens, pos1d, cos, sin):
            assert pos1d.ndim==2
            cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
            sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
            return (tokens * cos) + (self.rotate_half(tokens) * sin)
            
        def forward(self, tokens, positions):
            """
            input:
                * tokens: batch_size x nheads x ntokens x dim
                * positions: batch_size x ntokens x 2 (y and x position of each token)
            output:
                * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
            """
            assert tokens.size(3)%2==0, "number of dimensions should be a multiple of two"
            D = tokens.size(3) // 2
            assert positions.ndim==3 and positions.shape[-1] == 2 # Batch, Seq, 2
            cos, sin = self.get_cos_sin(D, int(positions.max())+1, tokens.device, tokens.dtype)
            # split features into two along the feature dimension, and apply rope1d on each half
            y, x = tokens.chunk(2, dim=-1)
            y = self.apply_rope1d(y, positions[:,:,0], cos, sin)
            x = self.apply_rope1d(x, positions[:,:,1], cos, sin)
            tokens = torch.cat((y, x), dim=-1)
            return tokens
     
# patch embedding
class PositionGetter(object):
    """ return positions of patches """

    def __init__(self):
        self.cache_positions = {}
        
    def __call__(self, b, h, w, device):
        if not (h,w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h,w] = torch.cartesian_prod(y, x) # (h, w, 2)
        pos = self.cache_positions[h,w].view(1, h*w, 2).expand(b, -1, 2).clone()
        return pos


#----------------------------------------------------------
# RoPE3D: RoPE implementation in 3D (ref_id + space)
# Fixed dimension split: ref_id=22, y=21, x=21 (total 64)
#----------------------------------------------------------

class RoPE3D(torch.nn.Module):
    """
    RoPE implementation in 3D (ref_id, height, width)
    
    Fixed dimension split for head_dim=64:
    - ref_id: 22 dimensions
    - y: 21 dimensions  
    - x: 21 dimensions
    
    This ensures no dim divisibility issues (64 = 22 + 21 + 21)
    """
    
    def __init__(self, freq=100.0, F0=1.0, ref_freq=10.0):
        super().__init__()
        self.base = freq  # for spatial dimensions (y, x)
        self.ref_base = ref_freq  # for ref_id dimension
        self.F0 = F0
        self.cache = {}
        
        # Fixed dimension split
        self.D_ref = 22  # ref_id gets 22 dims
        self.D_y = 21    # y gets 21 dims
        self.D_x = 21    # x gets 21 dims

    def get_cos_sin(self, D, seq_len, device, dtype, base):
        key = (D, seq_len, device, dtype, base)
        if key not in self.cache:
            # For D dimensions, we need D/2 frequencies
            # When D is odd, we compute ceil(D/2) frequencies and truncate
            half_D = (D + 1) // 2  # ceil(D/2)
            inv_freq = 1.0 / (base ** (torch.arange(0, half_D).float().to(device) * 2 / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            # Duplicate and truncate to exactly D dimensions
            freqs = torch.cat((freqs, freqs), dim=-1)[:, :D]
            cos = freqs.cos()  # (Seq, D)
            sin = freqs.sin()
            self.cache[key] = (cos, sin)
        return self.cache[key]
        
    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)
        
    def forward(self, tokens, positions):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 3 (ref_id, y, x)
        output:
            * tokens after applying RoPE3D (batch_size x nheads x ntokens x dim)
        """
        dim = tokens.size(3)
        assert dim == self.D_ref + self.D_y + self.D_x, \
            f"dim {dim} != {self.D_ref} + {self.D_y} + {self.D_x}"
        assert positions.ndim == 3 and positions.shape[-1] == 3  # Batch, Seq, 3
        
        # Get cos/sin for ref_id dimension
        cos_r, sin_r = self.get_cos_sin(self.D_ref, int(positions[:,:,0].max())+1, 
                                         tokens.device, tokens.dtype, self.ref_base)
        # Get cos/sin for spatial dimensions (y and x have different D)
        cos_y, sin_y = self.get_cos_sin(self.D_y, int(positions[:,:,1].max())+1, 
                                         tokens.device, tokens.dtype, self.base)
        cos_x, sin_x = self.get_cos_sin(self.D_x, int(positions[:,:,2].max())+1, 
                                         tokens.device, tokens.dtype, self.base)
        
        # Split features using fixed dimensions
        r = tokens[..., :self.D_ref]                           # [0:22]
        y = tokens[..., self.D_ref:self.D_ref+self.D_y]        # [22:43]
        x = tokens[..., self.D_ref+self.D_y:]                  # [43:64]
        
        # Apply rope1d on each part with corresponding position
        r = self.apply_rope1d(r, positions[:,:,0], cos_r, sin_r)  # ref_id
        y = self.apply_rope1d(y, positions[:,:,1], cos_y, sin_y)  # height
        x = self.apply_rope1d(x, positions[:,:,2], cos_x, sin_x)  # width
        
        tokens = torch.cat((r, y, x), dim=-1)
        return tokens


class PositionGetter3D(object):
    """ return 3D positions of patches (ref_id, y, x) """

    def __init__(self):
        self.cache_positions = {}
        
    def __call__(self, b, n, h, w, device, chunk_size=None):
        """
        Args:
            b: batch size
            n: number of frames
            h: height in patches
            w: width in patches
            device: torch device
            chunk_size: size of each chunk for computing ref_id
                        If None, uses frame_idx directly (backward compatible)
        Returns:
            pos: (b, n*h*w, 3) with (ref_id, y, x)
            
        ref_id formula:
        - frame 0: ref_id = 0
        - other frames: ref_id = ((frame_idx - 1) // chunk_size) * chunk_size
        """
        # Default chunk_size for backward compatibility
        if chunk_size is None:
            chunk_size = n // 2
            
        cache_key = (n, h, w, chunk_size)
        if cache_key not in self.cache_positions:
            positions = []
            for frame_idx in range(n):
                # Compute ref_id based on chunk_size
                if frame_idx == 0:
                    ref_id = 0
                else:
                    ref_id = ((frame_idx - 1) // chunk_size) * chunk_size
                
                for py in range(h):
                    for px in range(w):
                        positions.append([ref_id, py, px])
            pos = torch.tensor(positions, dtype=torch.long, device=device)
            self.cache_positions[cache_key] = pos
        
        pos = self.cache_positions[cache_key].to(device).view(1, n*h*w, 3).expand(b, -1, 3).clone()
        return pos


#----------------------------------------------------------
# RoPE4D_RefAware: RoPE with reference awareness (ref_id, t, y, x)
# 4D encoding solves dim divisibility issues (64 / 4 = 16)
#----------------------------------------------------------

class RoPE4D_ChunkAware(torch.nn.Module):
    """
    Chunk-Aware 4D RoPE implementation: (chunk_idx, t, y, x)
    
    Key design improvements over ref_id-based encoding:
    - chunk_idx: which chunk this frame belongs to (0, 1, 2, ...) - CONTINUOUS values
    - t: temporal position within the chunk (0 for anchor, 1-chunk_size for others)
    - y, x: spatial position in the patch grid
    
    Dimension split for head_dim=64 (optimized for value ranges):
    - chunk_idx: 8 dims (small value range: 0, 1, 2, 3, ...)
    - t: 8 dims (small value range: 0 to chunk_size)
    - y: 24 dims (larger range: 0 to ~37 for 518x518 images)
    - x: 24 dims (larger range: 0 to ~37)
    
    This gives more capacity to spatial dimensions while keeping temporal compact.
    """
    
    def __init__(self, freq=100.0, chunk_freq=10000, temporal_freq=1000.0):
        super().__init__()
        self.base = freq              # for spatial dimensions (y, x)
        self.chunk_base = chunk_freq  # for chunk_idx dimension
        self.temporal_base = temporal_freq  # for temporal dimension t
        self.cache = {}
        
        # Fixed dimension split (8 + 8 + 24 + 24 = 64)
        self.D_chunk = 8   # chunk_idx gets 8 dims
        self.D_t = 8       # t gets 8 dims  
        self.D_y = 24      # y gets 24 dims
        self.D_x = 24      # x gets 24 dims

    def get_cos_sin(self, D, seq_len, device, dtype, base):
        key = (D, seq_len, device, dtype, base)
        if key not in self.cache:
            # For D dimensions, we need D/2 frequencies
            half_D = (D + 1) // 2
            inv_freq = 1.0 / (base ** (torch.arange(0, half_D).float().to(device) * 2 / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)[:, :D]  # Truncate to D
            cos = freqs.cos()  # (Seq, D)
            sin = freqs.sin()
            self.cache[key] = (cos, sin)
        return self.cache[key]
        
    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)
        
    def forward(self, tokens, positions):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 4 (chunk_idx, t, y, x)
        output:
            * tokens after applying RoPE (batch_size x nheads x ntokens x dim)
        
        Dimension layout: [chunk(8) | t(8) | y(24) | x(24)]
        """
        dim = tokens.size(3)
        expected_dim = self.D_chunk + self.D_t + self.D_y + self.D_x
        assert dim == expected_dim, f"dim {dim} != expected {expected_dim}"
        assert positions.ndim == 3 and positions.shape[-1] == 4
        
        # Get cos/sin for each dimension with appropriate base frequency
        cos_c, sin_c = self.get_cos_sin(self.D_chunk, int(positions[:,:,0].max())+1, 
                                         tokens.device, tokens.dtype, self.chunk_base)
        cos_t, sin_t = self.get_cos_sin(self.D_t, int(positions[:,:,1].max())+1, 
                                         tokens.device, tokens.dtype, self.temporal_base)
        cos_y, sin_y = self.get_cos_sin(self.D_y, int(positions[:,:,2].max())+1, 
                                         tokens.device, tokens.dtype, self.base)
        cos_x, sin_x = self.get_cos_sin(self.D_x, int(positions[:,:,3].max())+1, 
                                         tokens.device, tokens.dtype, self.base)
        
        # Split features using fixed dimensions
        c = tokens[..., :self.D_chunk]                                    # [0:8]
        t = tokens[..., self.D_chunk:self.D_chunk+self.D_t]              # [8:16]
        y = tokens[..., self.D_chunk+self.D_t:self.D_chunk+self.D_t+self.D_y]  # [16:40]
        x = tokens[..., self.D_chunk+self.D_t+self.D_y:]                 # [40:64]
        
        # Apply rope1d on each part
        c = self.apply_rope1d(c, positions[:,:,0], cos_c, sin_c)  # chunk_idx
        t = self.apply_rope1d(t, positions[:,:,1], cos_t, sin_t)  # temporal
        y = self.apply_rope1d(y, positions[:,:,2], cos_y, sin_y)  # height
        x = self.apply_rope1d(x, positions[:,:,3], cos_x, sin_x)  # width
        
        return torch.cat((c, t, y, x), dim=-1)


# Legacy alias - redirect to new class
class RoPE4D_RefAware(RoPE4D_ChunkAware):
    """Backward compatibility alias for RoPE4D_ChunkAware"""
    def __init__(self, freq=100.0, ref_freq=10.0, temporal_freq=10.0):
        super().__init__(freq=freq, chunk_freq=ref_freq, temporal_freq=temporal_freq)

# Aliases for backward compatibility
RoPE3D_RefAware = RoPE4D_RefAware
RoPE3D_ChunkAware = RoPE4D_RefAware


class PositionGetter4D_ChunkAware(object):
    """
    Return 4D positions: (chunk_idx, t, y, x)
    
    Key improvement: uses chunk_idx (continuous: 0, 1, 2, ...) instead of ref_id (sparse: 0, 4, 8, ...)
    
    chunk_idx: which chunk this frame belongs to
    t: temporal position within the chunk
    y, x: spatial position in patch grid
    
    Position encoding distribution (N=12, chunk_size=4):
    Frame:     0  1  2  3  4  5  6  7  8   9  10 11
    chunk_idx: 0  0  0  0  1  1  1  1  2   2   2  2  ← CONTINUOUS!
    t:         0  1  2  3  0  1  2  3  0   1   2  3  ← Repeating pattern
    
    This gives RoPE a much better distribution to work with compared to:
    ref_id:    0  0  0  0  0  4  4  4  4   8   8  8  ← SPARSE (problematic!)
    """

    def __init__(self):
        self.cache = {}
        
    def __call__(self, b, n, h, w, device, chunk_size=None, ref_ids=None):
        """
        Args:
            b: batch size
            n: total number of frames
            h, w: patch grid size (height, width in patches)
            chunk_size: size of each chunk (default: n//2 for backward compatibility)
            ref_ids: Deprecated, ignored. Use chunk_size instead.
        Returns:
            pos: (b, n*h*w, 4) with (chunk_idx, t, y, x)
        """
        if chunk_size is None:
            chunk_size = n // 2  # backward compatible
            
        key = (n, h, w, chunk_size)
        if key not in self.cache:
            positions = []
            for frame_idx in range(n):
                # Compute chunk_idx (continuous: 0, 1, 2, ...)
                if frame_idx == 0:
                    chunk_idx = 0
                    t = 0  # First frame (anchor of first chunk)
                else:
                    # chunk_idx = floor((frame_idx - 1) / chunk_size)
                    chunk_idx = (frame_idx - 1) // chunk_size
                    # t = position within chunk (1 to chunk_size for non-first-frame of chunk)
                    t = (frame_idx - 1) % chunk_size + 1
                
                for py in range(h):
                    for px in range(w):
                        positions.append([chunk_idx, t, py, px])
            pos = torch.tensor(positions, dtype=torch.long, device=device)
            self.cache[key] = pos
        
        # Clone and expand for batch
        pos = self.cache[key].to(device).view(1, n*h*w, 4).expand(b, -1, 4).clone()
        return pos
    
    def get_position_for_single_frame(self, frame_idx, h, w, device, chunk_size):
        """
        Get position encoding for a single frame (useful for inference).
        
        Args:
            frame_idx: index of the frame
            h, w: patch grid size
            chunk_size: size of each chunk
        Returns:
            pos: (1, h*w, 4) with (chunk_idx, t, y, x)
        """
        if frame_idx == 0:
            chunk_idx = 0
            t = 0
        else:
            chunk_idx = (frame_idx - 1) // chunk_size
            t = (frame_idx - 1) % chunk_size + 1
        
        positions = []
        for py in range(h):
            for px in range(w):
                positions.append([chunk_idx, t, py, px])
        
        pos = torch.tensor(positions, dtype=torch.long, device=device)
        return pos.view(1, h*w, 4)


# Legacy aliases for backward compatibility
class PositionGetter3D_RefAware(PositionGetter4D_ChunkAware):
    """Backward compatibility alias"""
    pass

PositionGetter3D_ChunkAware = PositionGetter4D_ChunkAware


#----------------------------------------------------------
# ALiBi2D_Temporal: ALiBi for temporal dimensions only
# RoPE2D handles spatial (y, x), ALiBi handles (chunk_id, t)
#----------------------------------------------------------

class ALiBi2D_Temporal(torch.nn.Module):
    """
    ALiBi (Attention with Linear Biases) for temporal dimensions (chunk_id, t).
    
    This is used in HYBRID mode with RoPE2D:
    - RoPE2D encodes spatial positions (y, x) via rotary embeddings
    - ALiBi2D_Temporal encodes temporal positions (chunk_id, t) via attention bias
    
    Key advantage: Perfect extrapolation for inference
    - Training: chunk_id ∈ [0, 3]
    - Inference: chunk_id ∈ [0, 200+]
    - ALiBi uses linear distance, so no distribution shift!
    
    Formula:
        bias(i, j) = -m * (chunk_weight * |chunk_i - chunk_j| + temporal_weight * |t_i - t_j|)
    
    where m is the slope for each attention head (different per head).
    """
    
    def __init__(self, num_heads, chunk_weight=1.0, temporal_weight=0.5):
        """
        Args:
            num_heads: number of attention heads
            chunk_weight: weight for chunk_id distance (default 1.0, most important)
            temporal_weight: weight for temporal t distance (default 0.5)
        """
        super().__init__()
        self.num_heads = num_heads
        # Store as Python floats to avoid tensor conversion issues
        self.chunk_weight = float(chunk_weight)
        self.temporal_weight = float(temporal_weight)
        
        # Generate slopes for each head using geometric sequence
        # Paper recommendation: slopes = 2^(-8/n * i) for i in [1, 2, ..., n]
        slopes = torch.pow(2.0, -torch.linspace(0, 8, num_heads))
        self.register_buffer('slopes', slopes)
    
    def forward(self, positions_4d):
        """
        Compute ALiBi temporal bias from 4D positions.
        
        Args:
            positions_4d: (B, L, 4) with (chunk_id, t, y, x)
                         We only use chunk_id and t here.
        
        Returns:
            bias: (B, num_heads, L, L) attention bias matrix
                  Negative values penalize distant frames.
        """
        B, L, _ = positions_4d.shape
        device = positions_4d.device
        
        # Extract temporal dimensions only (chunk_id, t)
        chunk_ids = positions_4d[:, :, 0:1].float()  # (B, L, 1)
        t_ids = positions_4d[:, :, 1:2].float()      # (B, L, 1)
        
        # Compute pairwise distance matrices
        # chunk_dist[b, i, j] = |chunk_i - chunk_j|
        chunk_dist = torch.abs(
            chunk_ids - chunk_ids.transpose(1, 2)
        )  # (B, L, L)
        
        t_dist = torch.abs(
            t_ids - t_ids.transpose(1, 2)
        )  # (B, L, L)
        
        # Weighted combination of distances
        total_dist = (
            self.chunk_weight * chunk_dist +
            self.temporal_weight * t_dist
        )  # (B, L, L)
        
        # Apply per-head slopes: bias = -slope * distance
        # slopes: (num_heads,) -> (1, num_heads, 1, 1)
        # total_dist: (B, L, L) -> (B, 1, L, L)
        bias = -self.slopes.view(1, -1, 1, 1) * total_dist.unsqueeze(1)
        
        return bias  # (B, num_heads, L, L)