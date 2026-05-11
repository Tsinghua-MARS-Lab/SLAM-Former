import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from copy import deepcopy

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d
from .layers.camera_head import CameraHead
from .layers.conv_head import ConvHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from huggingface_hub import PyTorchModelHubMixin
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass
from transformers.file_utils import ModelOutput



import pdb
import time



class SLAMFormer(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            retention_ratio=0.5,
            bn_every=10,
            use_conv_head=False
        ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        self.retention_ratio = retention_ratio
        self.bn_every = bn_every

        self.use_conv_head = use_conv_head  

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features        # 1024
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
            use_checkpoint=True
        )

        if self.use_conv_head:
            self.point_head = ConvHead(
                num_features=4,
                dim_in=dec_embed_dim,
                projects=nn.Identity(),
                dim_out=[2, 1],
                dim_proj=1024,
                dim_upsample=[256, 128, 64],
                dim_times_res_block_hidden=2,
                num_res_blocks=2,
                res_block_norm='group_norm',
                last_res_blocks=0,
                last_conv_channels=32,
                last_conv_size=1,
                using_uv=True,
            )
        else:
            self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)


        # ----------------------
        #     Conf Decoder
        # ----------------------
        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)


        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=True
        )
        self.camera_head = CameraHead(dim=512)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        # token-level prune cache (for streaming hidden_F)
        self._prune_idx_cache = None
        self._prune_idx_cache_N = 0
        self._prune_idx_cache_hw = None

    # pruner
    def pairwise_cosine_similarity(self, matrix):
        norm = matrix.norm(dim=1, keepdim=True).clamp_min(1e-6)
        norm_matrix = matrix / norm
        return torch.mm(norm_matrix, norm_matrix.t())

    def Pruner(self, visual_feature_vectors, image_feature_length, cosine_matrix=None, threshold_ratio=0.1, threshold_terms=None):            
        if threshold_terms is None:
            threshold_terms = int(round(threshold_ratio * image_feature_length))
        # clamp threshold to valid range
        threshold_terms = max(1, min(threshold_terms, image_feature_length))

        if cosine_matrix is None:
            cosine_matrix = 1.0 - self.pairwise_cosine_similarity(visual_feature_vectors)

        dev = visual_feature_vectors.device

        s = torch.empty(threshold_terms, dtype=torch.long, device=dev)

        initial_scores = torch.topk(cosine_matrix, 2, dim=0, largest=False).values[1]  # [M]
        first_idx = torch.argmax(initial_scores)
        s[0] = first_idx
        
        scores = cosine_matrix[first_idx].clone()
        scores[first_idx]=-float('inf')
        filled = 1
        while filled < threshold_terms:
            idx = torch.argmax(scores)
            s[filled] = idx
            filled += 1
            scores = torch.minimum(scores, cosine_matrix[idx])
            scores[idx]=-float('inf')
        
        return s.sort()[0], cosine_matrix

 
    # divprune (diversity-based pruning)
    def DivPrune(
        self,
        visual_feature_vectors: torch.Tensor,
        image_feature_length: int,
        cosine_matrix: Optional[torch.Tensor] = None,
        threshold_ratio: float = 0.25,
        threshold_terms: Optional[int] = None,
    ):
        """Diversity-based token pruning.

        This is a thin wrapper to reuse the existing incremental diversity selector.
        Returns selected indices (sorted) and the distance matrix (1 - cosine similarity).
        """
        return self.Pruner(
            visual_feature_vectors,
            image_feature_length=image_feature_length,
            cosine_matrix=cosine_matrix,
            threshold_ratio=threshold_ratio,
            threshold_terms=threshold_terms,
        )
    
    
    def decode(self, N, H, W, hidden_I=None, hidden_F=None, use_cache=False):
        '''
            during inference there is only branch1 and branch2 because branch3==branch1 now with just different cache
        '''

        branch = 1
        if hidden_I is not None:
            branch = 1
            hidden = hidden_I

            BN, hw, C = hidden.shape
            B = BN // N

            
            hidden = hidden.reshape(B*N, hw, -1)

            register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

            # Concatenate special tokens with patch tokens
            hidden = torch.cat([register_token, hidden], dim=1)
            hw = hidden.shape[1]

            if hidden_F is None:
                branch = 1
            else:
                branch = 3
                hidden = hidden.view(B,N,-1,C)
                hidden = torch.cat([hidden_F.view(B,N,-1,C)[:,:int(N//2)], hidden[:,int(N//2):]],axis=1).reshape(B*N,-1,C) 


        elif hidden_I is None and hidden_F is not None:
            branch = 2
            hidden = hidden_F
            BN, P, C = hidden.shape
            B = 1
            #hw = int(NP//N)
            hw = P




        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        final_output = []
        if branch==1 and not hasattr(self,"fkv"):
            self.fkv = [None] * (len(self.decoder)//2)
        elif branch==2:
            self.fkv = [None] * (len(self.decoder)//2)

        kv = self.fkv

        idx = None

        if use_cache and hidden_I is None and self.retention_ratio < 1.0:
            try:
                dev = hidden.device

                # branch==2: hidden shape is [BN, hw, C], and BN == N (B=1)
                cur_len = int(N * hw)
                special_cnt = int(min(self.patch_start_idx, hw))
                patch_cnt = int(hw - special_cnt)

                # cache validation
                cache_ok = (
                    (self._prune_idx_cache is not None)
                    and (self._prune_idx_cache_hw == int(hw))
                    and (int(self._prune_idx_cache_N) <= int(N))
                )

                # If stream resets or token layout changes, drop cache and recompute from scratch.
                if (not cache_ok) or (int(self._prune_idx_cache_N) > int(N)):
                    self._prune_idx_cache = None
                    self._prune_idx_cache_N = 0
                    self._prune_idx_cache_hw = int(hw)

                prev_N = int(self._prune_idx_cache_N) if (self._prune_idx_cache is not None) else 0
                # hidden_F increases by 10 frames each step; only compute for the last 10 new frames.
                new_start = max(prev_N, int(N) - self.bn_every)

                if patch_cnt <= 0:
                    idx = torch.arange(cur_len, device=dev, dtype=torch.long)
                    # full keep -> cache becomes trivial
                    self._prune_idx_cache = idx.detach()
                    self._prune_idx_cache_N = int(N)
                    self._prune_idx_cache_hw = int(hw)
                else:
                    keep_patch_per_frame = max(1, int(round(self.retention_ratio * patch_cnt)))
                    kept_list = []

                    # reuse cached idx for old frames
                    if self._prune_idx_cache is not None and prev_N > 0:
                        kept_list.append(self._prune_idx_cache.to(device=dev, dtype=torch.long))

                    # compute idx only for the new frames
                    for f in range(int(new_start), int(N)):
                        base = f * int(hw)

                        # keep special tokens
                        if special_cnt > 0:
                            kept_list.append(
                                base + torch.arange(special_cnt, device=dev, dtype=torch.long)
                            )

                        # divprune patch tokens based on hidden_F
                        patch_feat = hidden[f, special_cnt:, :].to(torch.float32)
                        
                        sel_rel, _ = self.DivPrune(
                            patch_feat,
                            image_feature_length=int(patch_cnt),
                            threshold_terms=int(keep_patch_per_frame),
                        )
                        
                        kept_list.append(base + special_cnt + sel_rel)

                    idx = torch.cat(kept_list, dim=0).unique(sorted=True)

                    # safety clamp
                    idx = idx[(idx >= 0) & (idx < cur_len)]

                    # update cache (idx for frames [0, N))
                    self._prune_idx_cache = idx.detach()
                    self._prune_idx_cache_N = int(N)
                    self._prune_idx_cache_hw = int(hw)

            except Exception:
                idx = None

        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
                global_ = False
                kv_ = None
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)
                global_ = True
                kv_ = kv[int(i//2)]
            

            if use_cache and global_:
                hidden, kv_ = blk(hidden, xpos=pos, N=N, branch=branch, global_=global_, kvcache=kv_, use_cache=use_cache, idx=idx)
                kv[int(i//2)] = kv_
            else:
                hidden = blk(hidden, xpos=pos, N=N, branch=branch)

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        if use_cache:
            return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1), kv
        else:
            return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)
    


    def extract(self, hidden, pos=None, shape_=None, cam_only=False):
      with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        if shape_ is None:
            shape_ = self.shape_
        B,_,H,W,patch_h, patch_w = shape_

        BN,P,C = hidden.shape 
        B = 1
        N = BN #int(NP)
        if pos is None:
            if self.pos_type.startswith('rope'):
                pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

            if self.patch_start_idx > 0:
                # do not use position embedding for special tokens (camera and register tokens)
                # so set pos to 0 for the special tokens
                pos = pos + 1
                pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
                pos = torch.cat([pos_special, pos], dim=1)

        if not cam_only:
            point_hidden = self.point_decoder(hidden, xpos=pos)
            conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)


        with torch.amp.autocast(device_type='cuda', enabled=False):

            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)


            # local points
            if not cam_only:
                point_hidden = point_hidden.float()

                if self.use_conv_head:
                    xy, z = self.point_head(
                        point_hidden[:, self.patch_start_idx:],
                        patch_h=patch_h,
                        patch_w=patch_w,
                    )
                    xy = xy.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)
                    z = z.permute(0, 2, 3, 1).reshape(B, N, H, W, -1)
                    z = z.clamp(max=15.0)
                else:
                    ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
                    xy, z = ret.split([2, 1], dim=-1)
                
                z = torch.exp(z)
                local_points = torch.cat([xy * z, z], dim=-1)

                # confidence
                conf_hidden = conf_hidden.float()
                conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
                # unproject local points using camera poses
                points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]
            else:
                local_points = None
                conf = None
                points = None


        output = dict(points=points,
                    local_points=local_points,
                    conf=conf,
                    camera_poses=camera_poses,
                    )

        return output

    def KFT(self, img):
      with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):

        if img.dim() == 4:
            imgs = img[None]
        else:
            imgs = img[None, None]

        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        self.shape_ = (B,N,H,W,patch_h, patch_w)

        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden_I = self.encoder(imgs, is_training=False)

        if isinstance(hidden_I, dict):
            hidden_I = hidden_I["x_norm_patchtokens"] # 1,P,C

        hidden_F, pos = self.decode(N, H, W, hidden_I, hidden_F=None, use_cache=False)
        return hidden_F

    def frontendT(self, img):
      with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        if img.dim() == 4:
            imgs = img[None]
        else:
            imgs = img[None, None]

        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        self.shape_ = (B,N,H,W,patch_h, patch_w)

        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden_I = self.encoder(imgs, is_training=False)

        if isinstance(hidden_I, dict):
            hidden_I = hidden_I["x_norm_patchtokens"] # 1,P,C

        hidden_F, pos, kvcache = self.decode(N, H, W, hidden_I, hidden_F=None, use_cache=True)

        self.fkv = kvcache
        return hidden_F

    def backendT(self, hidden_F):
      with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):

        _,_,H,W,patch_h, patch_w = self.shape_
        BN,P,C2 = hidden_F.shape
        #P = patch_h*patch_w+5
        N = BN #SP//P
        hidden_B, pos, kvcache = self.decode(N, H, W, hidden_I=None, hidden_F=hidden_F[:,:,:int(C2//2)], use_cache=True)

        self.fkv = kvcache

        return hidden_B



