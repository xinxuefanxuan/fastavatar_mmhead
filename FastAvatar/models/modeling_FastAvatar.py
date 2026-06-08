import os
import time
import json
import logging
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from safetensors.torch import load_file

from FastAvatar.models.rendering.gs_renderer import GS3DRenderer, PointEmbed
from FastAvatar.models.alternating_cross_attn import AlternatingCrossAttn
from FastAvatar.models.framepack_utils import FramePackCompressor
from FastAvatar.models.motion_token_adapter import MotionTokenAdapter
from FastAvatar.models.encoders.dinov2_fusion_wrapper import Dinov2FusionWrapper
from diffusers.utils import is_torch_version

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _token_debug_enabled():
    return os.environ.get("FASTAVATAR_TOKEN_DEBUG", "0") == "1"


def _shape_summary(obj):
    if isinstance(obj, torch.Tensor):
        return list(obj.shape)
    if isinstance(obj, dict):
        return {k: _shape_summary(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_shape_summary(v) for v in obj]
    return type(obj).__name__


def _token_debug(label, **items):
    if _token_debug_enabled():
        summary = {k: _shape_summary(v) for k, v in items.items()}
        print(f"[FASTAVATAR_TOKEN_DEBUG] {label}: {summary}")


class ModelFastAvatar(nn.Module):
    def __init__(self,
                 transformer_dim: int = 1024,
                 transformer_layers: int = 10,
                 transformer_heads: int = 16,
                 aa_order: list = ["global", "frame"],
                 tf_grad_ckpt=True,
                 pretrained_model_path: str = None,
                 encoder_path: str = None,
                 encoder_grad_ckpt=True,
                 encoder_freeze: bool = True,
                 source_image_res: int = 512,
                 encoder_model_name: str = 'dinov2_vitl14_reg',
                 encoder_feat_dim: int = 1024,
                 pcl_dim: int=1024,
                 human_model_path="./model_zoo/human_parametric_models",
                 renderer_freeze: bool = True,
                 flame_subdivide_num=1,
                 gs_query_dim=1024,
                 gs_use_rgb=True,
                 gs_sh=3,
                 gs_mlp_network_config=None,
                 gs_xyz_offset_max_step=0.2,
                 gs_clip_scaling=0.01,
                 fix_opacity=False,
                 fix_rotation=False,
                 gs_fusion: bool = True,
                 num_base_frames: int = 16,
                 if_framepack: bool = False,
                 framepack_compression_level: int = 4,
                 vggt_path: str = None,
                 use_motion_token: bool = False,
                 motion_token_input_dim: int = 96,
                 motion_token_hidden_dim: int = None,
                 motion_token_mode: str = "add_query",
                 motion_token_scale: float = 1.0,
                 zero_flame_motion: bool = False,
                 freeze_backbone_for_motion_token: bool = False,
                 motion_token_source: str = "none",
                 motion_token_norm_stats: str = None,
                 motion_token_expr_dim: int = 50,
                 motion_token_pad_to_dim: int = 96,
                 motion_token_train_adapter_only_strict: bool = False,
                 motion_token_counterfactual_training: bool = False,
                 **kwargs,
                 ):
        super().__init__()
        self.gradient_checkpointing = tf_grad_ckpt
        self.encoder_gradient_checkpointing = encoder_grad_ckpt
        # attributes
        self.encoder_feat_dim = encoder_feat_dim
        self.gs_fusion = gs_fusion
        self.rendering_chunk_size_train = kwargs.get("rendering_chunk_size_train", 16)
        self.rendering_chunk_size_infer = kwargs.get("rendering_chunk_size_infer", 128)
        debug_max_query_points = kwargs.get("debug_max_query_points", None)
        self.debug_max_query_points = None if debug_max_query_points is None else int(debug_max_query_points)
        self.debug_skip_renderer = bool(kwargs.get("debug_skip_renderer", False))
        self.debug_latent_smoke_loss = bool(kwargs.get("debug_latent_smoke_loss", False))
        if self.debug_max_query_points is not None and not self.debug_skip_renderer:
            raise ValueError(
                "debug_max_query_points currently supports latent-only micro smoke runs. "
                "Set model.debug_skip_renderer=true or set debug_max_query_points=null for renderer runs."
            )
        self.num_base_frames = num_base_frames
        self.if_framepack = if_framepack
        self.framepack_compression_level = framepack_compression_level
        self.source_image_res = source_image_res
        self.use_motion_token = bool(use_motion_token)
        self.motion_token_input_dim = int(motion_token_input_dim)
        self.motion_token_hidden_dim = int(motion_token_hidden_dim or transformer_dim)
        self.motion_token_mode = motion_token_mode
        self.motion_token_scale = float(motion_token_scale)
        self.zero_flame_motion = bool(zero_flame_motion)
        self.freeze_backbone_for_motion_token = bool(freeze_backbone_for_motion_token)
        self.motion_token_source = motion_token_source
        self.motion_token_norm_stats = motion_token_norm_stats
        self.motion_token_expr_dim = int(motion_token_expr_dim)
        self.motion_token_pad_to_dim = int(motion_token_pad_to_dim)
        self.motion_token_train_adapter_only_strict = bool(motion_token_train_adapter_only_strict)
        self.motion_token_counterfactual_training = bool(motion_token_counterfactual_training)
        self.motion_token_norm_mean = None
        self.motion_token_norm_std = None
        if self.motion_token_norm_stats:
            with open(self.motion_token_norm_stats, "r") as f:
                norm_stats = json.load(f)
            self.motion_token_norm_mean = torch.tensor(norm_stats["mean"][:56], dtype=torch.float32)
            self.motion_token_norm_std = torch.tensor(norm_stats["std"][:56], dtype=torch.float32)
        if self.motion_token_source not in ("none", "frame_flame_gt", "zeros", "external"):
            raise ValueError(f"Unsupported motion_token_source={self.motion_token_source}")
        if self.motion_token_mode != "add_query":
            raise ValueError(f"P9.1 only supports motion_token_mode='add_query', got {self.motion_token_mode}")

        # FramePack compressor (only used when if_framepack=True)
        if if_framepack:
            self.framepack_compressor = FramePackCompressor(
                in_channels=encoder_feat_dim,
                inner_dim=encoder_feat_dim,
                compression_level=framepack_compression_level
            )
        else:
            self.framepack_compressor = None

        # image encoder
        self.encoder = Dinov2FusionWrapper(
            model_name=encoder_model_name,
            freeze=encoder_freeze,
            encoder_feat_dim=self.encoder_feat_dim,
        )

        # learnable points embedding
        self.pcl_embed = PointEmbed(dim=pcl_dim)

        # Optional P9.1 motion token adapter. Disabled by default to preserve exact behavior.
        self.motion_token_adapter = None
        if self.use_motion_token:
            self.motion_token_adapter = MotionTokenAdapter(
                input_dim=self.motion_token_input_dim,
                hidden_dim=self.motion_token_hidden_dim,
                output_dim=transformer_dim,
            )

        # Alternating cross Attention
        self.transformer = AlternatingCrossAttn(
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            inner_dim=transformer_dim,
            cond_dim=transformer_dim,
            gradient_checkpointing=self.gradient_checkpointing,
            aa_order=aa_order,
            patch_start_idx=0,
            if_framepack=if_framepack,
        )

        # renderer
        self.renderer = GS3DRenderer(human_model_path=human_model_path,
                                     subdivide_num=flame_subdivide_num,
                                     feat_dim=transformer_dim,
                                     query_dim=gs_query_dim,
                                     use_rgb=gs_use_rgb,
                                     sh_degree=gs_sh,
                                     mlp_network_config=gs_mlp_network_config,
                                     xyz_offset_max_step=gs_xyz_offset_max_step,
                                     clip_scaling=gs_clip_scaling,
                                     scale_sphere=kwargs.get("scale_sphere", False),
                                     fix_opacity=fix_opacity,
                                     fix_rotation=fix_rotation,
                                     skip_decoder=True,
                                     decode_with_extra_info=kwargs.get("decode_with_extra_info", None),
                                     gradient_checkpointing=self.gradient_checkpointing,
                                     add_teeth=kwargs.get("add_teeth", True),
                                     teeth_bs_flag=kwargs.get("teeth_bs_flag", False),
                                     oral_mesh_flag=kwargs.get("oral_mesh_flag", True),
                                     use_mesh_shading=kwargs.get('use_mesh_shading', False),
                                     render_rgb=kwargs.get("render_rgb", True),
                                     gs_pruning=kwargs.get("gs_pruning", False),
                                     )

        # Load pretrained model if available
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            logger.info(f"Loading pretrained model from {pretrained_model_path}")
            state_dict = load_file(pretrained_model_path)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if missing:
                logger.info(f"Missing keys ({len(missing)}): {missing[:10]}{'...' if len(missing) > 10 else ''}")
            if unexpected:
                logger.info(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
            logger.info(f"Pretrained model loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        elif encoder_path and os.path.exists(encoder_path):
            # Training from scratch: load DINOv2 encoder weights
            logger.info(f"Training from scratch: loading encoder weights from {encoder_path}")
            encoder_state_dict = load_file(encoder_path) if encoder_path.endswith('.safetensors') else torch.load(encoder_path, map_location='cpu')
            encoder_dict = {k.replace('encoder.model.', '').replace('model.', ''): v for k, v in encoder_state_dict.items() if 'fusion_head' not in k}
            self.encoder.model.load_state_dict(encoder_dict, strict=False)
            logger.info("Encoder weights loaded for training from scratch")

        # Set parameter requires_grad
        for name, param in self.renderer.named_parameters():
            if name.startswith('flame_model.'):
                param.requires_grad = False
            elif name.startswith('mlp_net.') or name.startswith('gs_net.'):
                param.requires_grad = not renderer_freeze
            else:
                param.requires_grad = False
        if self.freeze_backbone_for_motion_token:
            self.freeze_backbone_keep_motion_token_trainable()
        

    def freeze_backbone_keep_motion_token_trainable(self):
        for param in self.parameters():
            param.requires_grad = False
        if self.motion_token_adapter is None:
            raise RuntimeError("freeze_backbone_for_motion_token=True requires use_motion_token=True and a MotionTokenAdapter")
        for param in self.motion_token_adapter.parameters():
            param.requires_grad = True
        trainable = [(name, p.numel()) for name, p in self.named_parameters() if p.requires_grad]
        total = sum(n for _, n in trainable)
        print(f"[MotionTokenAdapter] freeze_backbone_for_motion_token=True; trainable parameter count={total}")
        for name, count in trainable:
            print(f"[MotionTokenAdapter] trainable: {name} ({count})")
        adapter_trainable = [name for name, _ in trainable if name.startswith("motion_token_adapter.")]
        if not adapter_trainable:
            raise RuntimeError("MotionTokenAdapter has no trainable parameters after freezing the backbone")
        if self.motion_token_train_adapter_only_strict:
            non_adapter = [name for name, _ in trainable if not name.startswith("motion_token_adapter.")]
            if non_adapter:
                raise RuntimeError(
                    "motion_token_train_adapter_only_strict=True but non-adapter parameters remain trainable: "
                    + ", ".join(non_adapter[:20])
                )

    def _prepare_motion_token_input(self, motion_token_input, batch_size: int, device, dtype):
        if not self.use_motion_token:
            return None
        if motion_token_input is None:
            if _token_debug_enabled():
                print(
                    f"[FASTAVATAR_TOKEN_DEBUG] use_motion_token=True but motion_token_input is missing; "
                    f"using zeros [B,{self.motion_token_input_dim}]"
                )
            motion_token_input = torch.zeros(batch_size, self.motion_token_input_dim, device=device, dtype=dtype)
        else:
            motion_token_input = motion_token_input.to(device=device, dtype=dtype)
        return motion_token_input

    def _condition_query_tokens(self, query_tokens, motion_token_input):
        if _token_debug_enabled():
            _token_debug(
                "motion_token/config",
                use_motion_token=self.use_motion_token,
                zero_flame_motion=self.zero_flame_motion,
                motion_token_scale=self.motion_token_scale,
            )
        if not self.use_motion_token:
            return query_tokens
        motion_token_input = self._prepare_motion_token_input(
            motion_token_input, query_tokens.shape[0], query_tokens.device, query_tokens.dtype
        )
        projected_motion_token = self.motion_token_adapter(motion_token_input)
        _token_debug(
            "motion_token/add_query",
            motion_token_input=motion_token_input,
            projected_motion_token=projected_motion_token,
            query_tokens_before=query_tokens,
            motion_token_scale=self.motion_token_scale,
        )
        query_tokens = query_tokens + self.motion_token_scale * projected_motion_token[:, None, None, :]
        _token_debug("motion_token/query_tokens_after", query_tokens=query_tokens)
        return query_tokens

    def _clone_and_zero_flame_motion(self, flame_params):
        if not self.zero_flame_motion:
            return flame_params
        zero_fields = ["expr", "jaw_pose", "neck_pose", "rotation"]
        # Keep shape/betas, camera, and translation unchanged. Translation can encode camera/framing alignment
        # in this codebase, so P9.1 does not treat it as removable motion conditioning.
        out = {}
        zeroed = []
        for key, value in flame_params.items():
            if key in zero_fields and isinstance(value, torch.Tensor):
                out[key] = torch.zeros_like(value)
                zeroed.append(key)
            else:
                out[key] = value
        if _token_debug_enabled():
            print(f"[FASTAVATAR_TOKEN_DEBUG] zero_flame_motion=True; zeroed_fields={zeroed}")
        return out

    def _flame_motion_norms(self, flame_params):
        norms = {}
        for key in ["expr", "neck_pose", "jaw_pose"]:
            value = flame_params.get(key) if isinstance(flame_params, dict) else None
            if isinstance(value, torch.Tensor):
                norms[key] = float(value.detach().float().norm().cpu())
            else:
                norms[key] = "missing"
        return norms

    def _frame_average(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 3:
            return value.mean(dim=1)
        if value.ndim == 2:
            return value
        raise ValueError(f"Expected FLAME tensor [B,N,D] or [B,D], got {list(value.shape)}")

    def _pad_or_truncate_last_dim(self, value: torch.Tensor, target_dim: int) -> torch.Tensor:
        if value.shape[-1] > target_dim:
            return value[..., :target_dim]
        if value.shape[-1] < target_dim:
            pad = torch.zeros(*value.shape[:-1], target_dim - value.shape[-1], device=value.device, dtype=value.dtype)
            return torch.cat([value, pad], dim=-1)
        return value

    def build_motion_token_input_from_flame(self, flame_params) -> torch.Tensor:
        required = ["expr", "neck_pose", "jaw_pose"]
        missing = [key for key in required if key not in flame_params or not isinstance(flame_params[key], torch.Tensor)]
        if missing:
            raise ValueError(f"motion_token_source='frame_flame_gt' requires FLAME tensor fields {required}; missing {missing}")

        expr = self._frame_average(flame_params["expr"])
        expr = self._pad_or_truncate_last_dim(expr[..., : self.motion_token_expr_dim], 50)
        neck = self._pad_or_truncate_last_dim(self._frame_average(flame_params["neck_pose"]), 3)
        jaw = self._pad_or_truncate_last_dim(self._frame_average(flame_params["jaw_pose"]), 3)
        motion56 = torch.cat([expr, neck, jaw], dim=-1)

        if self.motion_token_norm_mean is not None and self.motion_token_norm_std is not None:
            mean = self.motion_token_norm_mean.to(device=motion56.device, dtype=motion56.dtype)
            std = self.motion_token_norm_std.to(device=motion56.device, dtype=motion56.dtype).clamp_min(1e-8)
            motion56 = (motion56 - mean[None, :]) / std[None, :]

        token = self._pad_or_truncate_last_dim(motion56, self.motion_token_pad_to_dim)
        _token_debug("motion_token/frame_flame_gt", motion56=motion56, motion_token_input=token)
        return token

    def _resolve_motion_token_input(self, motion_token_input, flame_params):
        if not self.use_motion_token:
            return motion_token_input
        if motion_token_input is not None:
            return motion_token_input
        if self.motion_token_source == "none":
            return None
        if self.motion_token_source == "zeros":
            expr = flame_params.get("expr") if isinstance(flame_params, dict) else None
            if not isinstance(expr, torch.Tensor):
                raise ValueError("motion_token_source='zeros' requires flame_params['expr'] to infer batch/device")
            base = self._frame_average(expr)
            return torch.zeros(base.shape[0], self.motion_token_pad_to_dim, device=base.device, dtype=base.dtype)
        if self.motion_token_source == "frame_flame_gt":
            return self.build_motion_token_input_from_flame(flame_params)
        if self.motion_token_source == "external":
            raise ValueError("motion_token_source='external' requires an explicit motion_token_input tensor")
        raise ValueError(f"Unsupported motion_token_source={self.motion_token_source}")

    def forward_encode_image(self, image):
        """
        Encode image features, supporting both single and multi-frame inputs
        Args:
            image: [B, N_frames, C_img, H_img, W_img]
        Returns:
            image_feats: [B, N_output, H*W, C] - Base frames features
            compressed_cond: [B, 1, compressed_tokens, C] or None - Compressed frame features (if framepack enabled)
            base_indices: list or None - Indices of base frames in original input (if framepack enabled)
            spatial_compression: int or None - Spatial compression ratio used (if framepack enabled)
        """
        B, N_frames, C_img, H_img, W_img = image.shape
        _token_debug("forward_encode_image/input", image=image)
        image = image.view(B * N_frames, C_img, H_img, W_img)

        tgt_size = (self.source_image_res // 14) * 14
        if H_img != tgt_size or W_img != tgt_size:
            image = F.resize(image, (tgt_size, tgt_size), antialias=True)
        if self.training and self.encoder_gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            image_feats = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.encoder),
                image,
                **ckpt_kwargs,
            )
        else:
            image_feats = self.encoder(image)
        
        # image_feats: [B*N_frames, H*W, C]
        _, HW, C = image_feats.shape
        
        # Unified FramePack processing: Always provide compression for all frame counts
        if self.if_framepack:
            image_feats = image_feats.view(B, N_frames, HW, C)

            # Always use first min(N_frames, num_base_frames) as base frames
            actual_base_frames = min(N_frames, self.num_base_frames)
            base_image_feats = image_feats[:, :actual_base_frames]
            base_indices = list(range(actual_base_frames))

            # Prepare 3D features for compression
            H_feat = int(HW ** 0.5)
            image_feats_3d = image_feats.view(B, N_frames, H_feat, H_feat, C)

            if N_frames > self.num_base_frames:
                compressed_input_3d = image_feats_3d[:, actual_base_frames:]
            else:
                compressed_input_3d = image_feats_3d

            compressed_features, spatial_compression = self.framepack_compressor(compressed_input_3d)

            compressed_cond = compressed_features.reshape(B, 1, -1, C)
            _token_debug(
                "forward_encode_image/framepack_output",
                base_image_feats=base_image_feats,
                compressed_cond=compressed_cond,
                spatial_compression=torch.tensor(spatial_compression),
            )

            return base_image_feats, compressed_cond, base_indices, spatial_compression
        else:
            # No FramePack: simply slice all inputs to num_base_frames
            image_feats = image_feats.view(B, N_frames, HW, C)
            image_feats = image_feats[:, :self.num_base_frames]
            _token_debug("forward_encode_image/output", image_feats=image_feats)

            return image_feats, None, None, None

    def _query_point_subsample_indices(self, num_points: int, device):
        if self.debug_max_query_points is None or self.debug_max_query_points <= 0:
            return None
        if num_points <= self.debug_max_query_points:
            return None
        # Deterministic coverage across the full point set. This is debug-only and opt-in;
        # the default None preserves original full-point FastAvatar behavior.
        return torch.linspace(0, num_points - 1, steps=self.debug_max_query_points, device=device).long()

    def _subsample_query_points_for_debug(self, query_points, label: str):
        idx = self._query_point_subsample_indices(query_points.shape[-2], query_points.device)
        if idx is None:
            return query_points
        out = query_points.index_select(-2, idx)
        if _token_debug_enabled():
            print(
                f"[FASTAVATAR_TOKEN_DEBUG] debug_max_query_points/{label}: "
                f"before={list(query_points.shape)} after={list(out.shape)} max={self.debug_max_query_points}"
            )
        return out

    def forward_transformer(self, image_feats, query_points, query_feats=None, compressed_cond=None, spatial_compression=None, motion_token_input=None):
        """
        Args:
            image_feats: [B, N_input, H*W, C]
            query_points: [B, N_input, N_points, 3]
            query_feats: Optional query features
            compressed_cond: [B, 1, compressed_tokens, C] or None - Compressed frame features (if framepack enabled)
            spatial_compression: int or None - Spatial compression ratio used (if framepack enabled)
        Returns:
            latent_points: [B, N_input, N_points, C]
        """
        B, N_input = image_feats.shape[:2]
        _token_debug(
            "forward_transformer/input",
            image_feats=image_feats,
            query_points=query_points,
            query_feats=query_feats if query_feats is not None else "None",
            compressed_cond=compressed_cond if compressed_cond is not None else "None",
        )

        # Reshape query_points for pcl_embed
        x = self.pcl_embed(query_points.reshape(B, -1, 3))
        x = x.reshape(B, query_points.shape[1], query_points.shape[2], x.shape[-1])
        if query_feats is not None:
            x = x + query_feats.to(image_feats.dtype)
        x = self._condition_query_tokens(x, motion_token_input)

        # Prepare compressed frame query points if exists
        compressed_x = x[:, N_input:N_input+1] if compressed_cond is not None else None
        if compressed_x is not None:
            x = x[:, :N_input]

        _token_debug(
            "forward_transformer/tokens",
            point_tokens=x,
            compressed_point_tokens=compressed_x if compressed_x is not None else "None",
            context_tokens=image_feats,
            compressed_context=compressed_cond if compressed_cond is not None else "None",
        )
        latent_points = self.transformer(x, image_feats, compressed_x=compressed_x, compressed_cond=compressed_cond, spatial_compression=spatial_compression)
        _token_debug("forward_transformer/output", latent_points=latent_points)
        return latent_points

    @torch.compile
    def forward_latent_points(self, image, input_flame_params, motion_token_input=None):

        B, N_input = image.shape[:2]
        _token_debug("forward_latent_points/input", image=image, input_flame_params=input_flame_params)
        base_frames = min(self.num_base_frames, N_input)

        # Encode ALL frames + FramePack (compress non-base frames when if_framepack=True)
        base_feats, compressed_cond, _, spatial_compression = self.forward_encode_image(image)

        flame_for_query = self._clone_and_zero_flame_motion(input_flame_params.copy())
        if 'betas' not in flame_for_query and 'shape' in flame_for_query:
            flame_for_query['betas'] = flame_for_query['shape']
        query_points, _ = self.renderer.get_query_points(flame_for_query, device=image.device)
        _token_debug("forward_latent_points/query_points", query_points=query_points)

        # Prepare query points for transformer (base_frames + 1 compressed if exists)
        query_points_transformer = query_points[:, 0:1].repeat(1, base_frames, 1, 1)
        if compressed_cond is not None:
            query_points_transformer = torch.cat([query_points_transformer, query_points_transformer[:, 0:1]], dim=1)
        query_points_transformer = self._subsample_query_points_for_debug(query_points_transformer, "transformer")

        # Reconstruction Transformer
        latent_points = self.forward_transformer(
            base_feats,
            query_points_transformer,
            compressed_cond=compressed_cond,
            spatial_compression=spatial_compression,
            motion_token_input=motion_token_input
        )
        
        _token_debug("forward_latent_points/output", latent_points=latent_points, query_points_transformer=query_points_transformer)
        return latent_points, query_points_transformer

    def _render_multiple_frames(self, latent_points, query_points, inf_flame_params, c2ws, intrs, bg_colors, render_h, render_w, N_inf, chunk_size=16, input_indices=[0]):
        """
        Render multiple frames using the same latent points and query points with chunked rendering.

        Args:
            latent_points: [B, N_points, C] - Latent features for 3DGS
            query_points: [B, N_points, 3] - 3D query points
            inf_flame_params: Dict containing FLAME parameters for all frames
            c2ws: [B, N_inf, 4, 4] - Camera to world transformations
            intrs: [B, N_inf, 4, 4] - Camera intrinsics
            bg_colors: [B, N_inf, 3] - Background colors
            render_h, render_w: int - Render resolution
            N_inf: int - Number of frames to render
            chunk_size: int - Number of frames to render per chunk. If >= N_inf, render all frames at once

            rotation: [B, N_inf, 3] - Rotation parameters
            translation: [B, N_inf, 3] - Translation parameters
            expr: [B, N_inf, 100] - Expression parameters
            neck_pose: [B, N_inf, 3] - Neck pose parameters
            jaw_pose: [B, N_inf, 3] - Jaw pose parameters
            eyes_pose: [B, N_inf, 3] - Eyes pose parameters

        Returns:
            Dict containing concatenated render results for all frames
        """
        # `inf_flame_params` is already the renderer-ready FLAME dictionary from the caller.
        # In P9.2, forward()/infer_images() build the motion token from the original FLAME
        # tensors first, then pass the post-zeroing FLAME dictionary here. Keep a local
        # alias so all renderer uses are scoped and explicit.
        render_flame_params = inf_flame_params
        _token_debug(
            "render_multiple_frames/input",
            latent_points=latent_points,
            query_points=query_points,
            inf_flame_params=inf_flame_params,
            render_flame_params=render_flame_params,
            c2ws=c2ws,
            intrs=intrs,
            bg_colors=bg_colors,
        )
        if _token_debug_enabled():
            debug_zeroed_params = self._clone_and_zero_flame_motion(inf_flame_params)
            print(f"[FASTAVATAR_TOKEN_DEBUG] render_multiple_frames/zero_flame_motion={self.zero_flame_motion}")
            print(f"[FASTAVATAR_TOKEN_DEBUG] render_multiple_frames/inf_flame_keys={sorted(inf_flame_params.keys())}")
            print(f"[FASTAVATAR_TOKEN_DEBUG] render_multiple_frames/render_flame_keys={sorted(render_flame_params.keys())}")
            print(
                "[FASTAVATAR_TOKEN_DEBUG] render_multiple_frames/motion_norms "
                f"before_zero={self._flame_motion_norms(inf_flame_params)} "
                f"after_zero={self._flame_motion_norms(debug_zeroed_params)}"
            )
        # Calculate number of chunks
        if chunk_size >= N_inf:
            # Render all frames at once
            chunk_size = N_inf
        num_chunks = (N_inf + chunk_size - 1) // chunk_size

        render_res_list = []
        try:
            for chunk_idx in range(num_chunks):
                # Calculate frame indices for this chunk
                start_frame = chunk_idx * chunk_size
                end_frame = min((chunk_idx + 1) * chunk_size, N_inf)

                # Extract chunk-specific parameters
                chunk_flame_params = {}
                for k, v in render_flame_params.items():
                    if isinstance(v, torch.Tensor):
                        if k == "betas":
                            chunk_flame_params[k] = v[:, 0:1]  # [B, 1, ...]
                        else:
                            chunk_flame_params[k] = v[:, start_frame:end_frame]
                    else:
                        chunk_flame_params[k] = v

                # Convert all inputs to float32 for rasterization
                latent_points_f32 = latent_points.float()
                query_points_f32 = query_points.float()
                chunk_c2ws = c2ws[:, start_frame:end_frame].float()
                chunk_intrs = intrs[:, start_frame:end_frame].float()
                chunk_bg_colors = bg_colors[:, start_frame:end_frame].float()
                chunk_flame_params = {k: v.float() if isinstance(v, torch.Tensor) else v
                                    for k, v in chunk_flame_params.items()}

                # Render this chunk
                render_res = self.renderer(
                    gs_hidden_features=latent_points_f32,
                    query_points=query_points_f32,
                    flame_data=chunk_flame_params,
                    c2w=chunk_c2ws,
                    intrinsic=chunk_intrs,
                    height=render_h,
                    width=render_w,
                    background_color=chunk_bg_colors,
                    num_input_frames=len(input_indices),
                )
                render_res_list.append(render_res)

                # Clean up chunk-specific variables
                del render_res, chunk_flame_params, latent_points_f32, query_points_f32, chunk_c2ws, chunk_intrs, chunk_bg_colors

                # Force memory cleanup after each chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Combine results from all chunks
            out = {}  # Changed from defaultdict to regular dict
            for res in render_res_list:
                for k, v in res.items():
                    if k not in out:
                        out[k] = []
                    out[k].append(v)

            # Process each key in the output dictionary
            for k, v in out.items():
                if isinstance(v[0], torch.Tensor):
                    if k == "pruning_masks":
                        out[k] = torch.concat(v, dim=0)
                    else:  # Multi-dimensional tensors
                        out[k] = torch.concat(v, dim=1)
                        if k == "comp_rgb":
                            out[k] = out[k].permute(0, 1, 2, 3, 4) # [B, N_inf, H, W, C]
                    # Clean up the list immediately after concat to free memory
                    del v
                elif k == "gs_stats":
                    out[k] = v[0] if v else None
                else:
                    out[k] = v

            # Clean up render_res_list before returning
            del render_res_list
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            _token_debug("render_multiple_frames/output", **out)
            return out
        finally:
            # Additional cleanup if needed (render_res_list already cleaned in the loop)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def forward(self, input_image, target_image, input_c2ws, target_c2ws, input_intrs, target_intrs, input_bg_colors, target_bg_colors, landmarks, input_flame_params, inf_flame_params, uid, motion_token_input=None):
        B, N_input = input_image.shape[:2]
        N_target = target_image.shape[1]
        _token_debug(
            "forward/input",
            input_image=input_image,
            target_image=target_image,
            input_flame_params=input_flame_params,
            inf_flame_params=inf_flame_params,
            motion_token_input=motion_token_input if motion_token_input is not None else "None",
        )
        
        # Obtain rendering resolution from target image to guarantee match for losses
        render_h, render_w = target_image.shape[-2:]
        
        # Build token from original target FLAME before any explicit motion zeroing.
        motion_token_input = self._resolve_motion_token_input(motion_token_input, inf_flame_params)
        render_flame_params = self._clone_and_zero_flame_motion(inf_flame_params)

        # Forward: encoder + transformer, using GT FLAME params for query points
        latent_points, query_points = self.forward_latent_points(input_image, input_flame_params, motion_token_input=motion_token_input)
        if self.debug_skip_renderer:
            smoke_loss = latent_points.float().pow(2).mean()
            _token_debug(
                "forward/debug_latent_smoke",
                debug_skip_renderer=self.debug_skip_renderer,
                debug_latent_smoke_loss=self.debug_latent_smoke_loss,
                latent_points=latent_points,
                smoke_loss=smoke_loss,
            )
            return {
                "latent_smoke_loss": smoke_loss,
                "gs_stats": None,
            }
        
        del input_image, target_image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Ground Truth Camera scaling (e.g. from native to render_w)
        gt_native_w = target_intrs[..., 0, 2:3] * 2.0
        gt_native_h = target_intrs[..., 1, 2:3] * 2.0
        gt_scale_w = render_w / gt_native_w.clamp(min=1.0)
        gt_scale_h = render_h / gt_native_h.clamp(min=1.0)
        target_intrs_scaled = target_intrs.clone()
        target_intrs_scaled[..., 0, 0] *= gt_scale_w.squeeze(-1)
        target_intrs_scaled[..., 1, 1] *= gt_scale_h.squeeze(-1)
        target_intrs_scaled[..., 0, 2] *= gt_scale_w.squeeze(-1)
        target_intrs_scaled[..., 1, 2] *= gt_scale_h.squeeze(-1)

        # Single path: GT Camera + GT FLAME (from FLAME Tracking)
        latent_flat = latent_points.reshape(B, -1, latent_points.shape[-1])
        query_flat = query_points.detach().reshape(B, -1, 3)
        render_kwargs = dict(
            render_h=render_h, render_w=render_w,
            N_inf=N_target,
            chunk_size=self.rendering_chunk_size_train,
            input_indices=list(range(latent_points.shape[1]))
        )

        out = self._render_multiple_frames(
            latent_points=latent_flat,
            query_points=query_flat,
            inf_flame_params=render_flame_params,
            c2ws=target_c2ws,
            intrs=target_intrs_scaled,
            bg_colors=target_bg_colors,
            **render_kwargs
        )
        
        del input_c2ws, target_c2ws, input_intrs, target_intrs, landmarks, input_flame_params, latent_flat, query_flat
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        _token_debug("forward/output", **out)
        return out
    
    @torch.no_grad()
    def infer_images(self, image, input_c2ws, input_intrs, target_c2ws, target_intrs, target_bg_colors, input_flame_params, inf_flame_params=None, render_h=512, render_w=512, motion_token_input=None):
        B, N_input = image.shape[:2]
        N_target = target_c2ws.shape[1]
        _token_debug(
            "infer_images/input",
            image=image,
            input_flame_params=input_flame_params,
            inf_flame_params=inf_flame_params if inf_flame_params is not None else "None",
            target_c2ws=target_c2ws,
            target_intrs=target_intrs,
            motion_token_input=motion_token_input if motion_token_input is not None else "None",
        )
        
        modeling_time = time.time()
        
        motion_token_input = self._resolve_motion_token_input(motion_token_input, inf_flame_params)
        render_flame_params = self._clone_and_zero_flame_motion(inf_flame_params)
        latent_points, query_points = self.forward_latent_points(image, input_flame_params, motion_token_input=motion_token_input)
        if self.debug_skip_renderer:
            smoke_loss = latent_points.float().pow(2).mean()
            _token_debug(
                "infer_images/debug_latent_smoke",
                debug_skip_renderer=self.debug_skip_renderer,
                latent_points=latent_points,
                smoke_loss=smoke_loss,
            )
            return {"latent_smoke_loss": smoke_loss, "gs_stats": None}

        # Clean up input tensors immediately after forward_latent_points
        del image, input_c2ws, input_intrs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        modeling_time = time.time() - modeling_time

        # Use all input frames (no slicing)
        num_input_frames = latent_points.shape[1]

        # Reshape to concatenate frames along the points dimension (same as forward)
        B, num_frames, N_points, C = latent_points.shape
        latent_points_reshaped = latent_points.reshape(B, num_frames * N_points, C)
        query_points_reshaped = query_points.reshape(B, num_frames * N_points, 3)

        # Rendering
        render_start = time.time()
        out = self._render_multiple_frames(
            latent_points=latent_points_reshaped,
            query_points=query_points_reshaped,
            inf_flame_params=render_flame_params,
            c2ws=target_c2ws,
            intrs=target_intrs,
            bg_colors=target_bg_colors,
            render_h=render_h,
            render_w=render_w,
            N_inf=N_target,
            chunk_size=self.rendering_chunk_size_infer,  # Use inference chunk size
            input_indices=list(range(num_input_frames))
        )
        render_time = time.time() - render_start

        del latent_points_reshaped, query_points_reshaped, latent_points
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Store timing information in output
        out['modeling_time'] = modeling_time
        out['render_time'] = render_time
        
        # Reshape comp_rgb: [B, N_target, H, W, C] -> [N_target, H, W, 3]
        if "comp_rgb" in out and isinstance(out["comp_rgb"], torch.Tensor):
            if len(out["comp_rgb"].shape) == 5:
                out["comp_rgb"] = out["comp_rgb"][0].permute(0, 2, 3, 1)
        
        _token_debug("infer_images/output", **out)
        return out