import os
import math
import gc
from tqdm.auto import tqdm
import torch
from accelerate.logging import get_logger

from FastAvatar.runners.train.base_trainer import Trainer
from FastAvatar.utils.profile import DummyProfiler
from FastAvatar.runners import REGISTRY_RUNNERS

logger = get_logger(__name__)

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

@REGISTRY_RUNNERS.register(name="train.fastavatar")
class FastAvatarTrainer(Trainer):
    def __init__(self):
        super().__init__()
        self.model = self._build_model(self.cfg)
        self.optimizer = self._build_optimizer(self.model, self.cfg)
        self.train_loader, self.val_loader = self._build_dataloader(self.cfg)
        self.scheduler = self._build_scheduler(self.optimizer, self.cfg)
        self.pixel_loss_fn, self.perceptual_loss_fn, self.ssim_loss_fn, self.offset_loss_fn = self._build_loss_fn()
        
    def _build_model(self, cfg):
        assert cfg.experiment.type == 'fastavatar', \
            f"Config type {cfg.experiment.type} does not match with runner {self.__class__.__name__}"
        from FastAvatar.models.modeling_FastAvatar import ModelFastAvatar
        model = ModelFastAvatar(**cfg.model)
        
        if cfg.train.gradient_checkpointing:
            model.gradient_checkpointing = True
            model.encoder_gradient_checkpointing = True
        
        clear_memory()
        
        return model

    def _build_optimizer(self, model, cfg):
        all_params = []
        total_params = 0
        strict_adapter_only = bool(getattr(cfg.model, 'motion_token_train_adapter_only_strict', False))
        non_adapter_trainable = [
            n for n, p in model.named_parameters()
            if p.requires_grad and 'motion_token_adapter' not in n
        ]
        if strict_adapter_only and non_adapter_trainable:
            raise RuntimeError(
                "motion_token_train_adapter_only_strict=True but non-motion_token_adapter parameters are trainable before optimizer construction: "
                + ", ".join(non_adapter_trainable[:20])
            )
        
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            
            all_params.append(p)
            total_params += p.numel()
        
        lr = getattr(cfg.train.optim, 'lr', 5e-5)
        
        logger.info("======== Learning Rate Configuration ========")
        logger.info(f"  Total trainable parameters: {total_params:,}")
        logger.info(f"  Learning rate: {lr:.2e}")
        
        optimizer = torch.optim.AdamW(
            all_params,
            lr=lr,
            weight_decay=cfg.train.optim.weight_decay,
            betas=(cfg.train.optim.beta1, cfg.train.optim.beta2),
        )
        return optimizer
    
    def _build_scheduler(self, optimizer, cfg):
        def create_single_scheduler(opt, cfg_train):
            if cfg_train.scheduler.type == 'cosine':
                local_batches_per_epoch = math.floor(len(self.train_loader) / self.accelerator.num_processes)
                total_global_batches = cfg_train.epochs * math.ceil(local_batches_per_epoch / self.cfg.train.accum_steps)
                effective_warmup_iters = cfg_train.scheduler.warmup_real_iters
                from FastAvatar.utils.scheduler import CosineWarmupScheduler
                return CosineWarmupScheduler(
                    optimizer=opt,
                    warmup_iters=effective_warmup_iters,
                    max_iters=total_global_batches,
                )
            elif cfg_train.scheduler.type == 'constant':
                from FastAvatar.utils.scheduler import ConstantLR
                return ConstantLR(optimizer=opt)
            else:
                raise NotImplementedError(f"Scheduler type {cfg_train.scheduler.type} not implemented")

        return create_single_scheduler(optimizer, cfg.train)

    def _build_dataloader(self, cfg):
        from FastAvatar.datasets.mixer import MixerDataset

        # Always use datasets field - yaml controls everything
        if not hasattr(cfg.dataset, 'datasets') or not cfg.dataset.datasets:
            raise ValueError("dataset.datasets must be specified in config. Use datasets field to define your data sources.")

        shared_meta_path = getattr(cfg.dataset, 'meta_path', None)
        if shared_meta_path is None:
            raise ValueError("dataset.meta_path must be specified in config")

        subsets = []
        for name, dataset_cfg in cfg.dataset.datasets.items():
            subsets.append({
                'name': name,
                'root_dirs': dataset_cfg.root_dir,
                'meta_path': shared_meta_path,
                'val_id': getattr(dataset_cfg, 'val_id', None)
            })

        train_dataset = MixerDataset(
            split='train',
            subsets=subsets,
            input_frames=cfg.dataset.input_frames,
            frames_per_sample=cfg.dataset.input_frames + cfg.dataset.target_frames,
            render_image_res=cfg.dataset.render_image_res,
            source_image_res=cfg.dataset.source_image_res,
            disorder=True,
            val_num=cfg.dataset.val_num,
            is_val=False,
            use_teeth=getattr(cfg.model, 'add_teeth', True)
        )

        val_dataset = MixerDataset(
            split='val',
            subsets=subsets,
            input_frames=cfg.dataset.input_frames,
            frames_per_sample=cfg.dataset.input_frames + cfg.dataset.target_frames,
            render_image_res=cfg.dataset.render_image_res,
            source_image_res=cfg.dataset.source_image_res,
            disorder=True,
            is_val=True,
            val_num=cfg.dataset.val_num,
            use_teeth=getattr(cfg.model, 'add_teeth', True)
        )

        num_train_workers = int(cfg.dataset.num_train_workers)
        num_val_workers = int(cfg.dataset.num_val_workers)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=num_train_workers,
            pin_memory=cfg.dataset.pin_mem,
            persistent_workers=(num_train_workers > 0),
            drop_last=True
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.val.batch_size,
            shuffle=False,
            num_workers=num_val_workers,
            pin_memory=cfg.dataset.pin_mem,
            persistent_workers=(num_val_workers > 0),
            drop_last=False
        )

        return train_loader, val_loader

    def _build_loss_fn(self):
        from FastAvatar.losses import PixelLoss, LPIPSLoss, SSIMLoss
        from FastAvatar.losses.offset import ACAP_Loss
        from FastAvatar.losses.arcface import ArcFaceLoss

        pixel_loss_type = getattr(self.cfg.train.loss, 'pixel_loss_type', 'mse')
        pixel_loss_fn = PixelLoss(option=pixel_loss_type)
        with self.accelerator.main_process_first():
            perceptual_loss_fn = LPIPSLoss(device=self.device, prefetch=True)
            # Initialize ArcFace loss
            self.id_loss_fn = ArcFaceLoss(
                device=self.device, 
                arcface_checkpoint=getattr(self.cfg.train.loss, 'arcface_checkpoint', "./model_zoo/arcface/arcface_checkpoint.tar")
            )
            
        ssim_loss_fn = SSIMLoss()
        offset_loss_fn = ACAP_Loss()

        return pixel_loss_fn, perceptual_loss_fn, ssim_loss_fn, offset_loss_fn

    def register_hooks(self):
        pass

    def _build_flame_dicts(self, data):
        flame_keys = ['expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'teeth_bs', 'translation', 'shape', 'betas']
        input_flame_params = {
            k.replace('input_', ''): v for k, v in data.items()
            if k.startswith('input_') and k.replace('input_', '') in flame_keys
        }
        target_flame_params = {
            k.replace('target_', ''): v for k, v in data.items()
            if k.startswith('target_') and k.replace('target_', '') in flame_keys
        }
        return input_flame_params, target_flame_params

    def _move_batch_to_device(self, data):
        if data is None:
            return None
        if torch.is_tensor(data):
            return data.to(self.device, non_blocking=True)
        if isinstance(data, dict):
            return {k: self._move_batch_to_device(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._move_batch_to_device(v) for v in data]
        if isinstance(data, tuple):
            return tuple(self._move_batch_to_device(v) for v in data)
        return data

    def _forward_render(self, data, motion_token_input=None):
        uid = data['uid']
        input_image = data['rgbs']
        target_image = data['target_rgbs']
        input_lms = data['landmarks']
        target_lms = data['target_landmarks']
        landmarks = torch.cat([input_lms, target_lms], dim=1)
        input_flame_params, target_flame_params = self._build_flame_dicts(data)
        outputs = self.model(
            input_image=input_image,
            target_image=target_image,
            input_c2ws=data['c2ws'],
            target_c2ws=data['target_c2ws'],
            input_intrs=data['intrs'],
            target_intrs=data['target_intrs'],
            input_bg_colors=data['bg_colors'],
            target_bg_colors=data['target_bg_colors'],
            landmarks=landmarks,
            input_flame_params=input_flame_params,
            inf_flame_params=target_flame_params,
            uid=uid,
            motion_token_input=motion_token_input,
        )
        outputs['full_landmarks'] = landmarks
        return outputs, input_flame_params, target_flame_params

    def _renderer_losses(self, outputs, target_image):
        loss_renderer = 0.
        loss_r_pixel = None
        loss_r_perceptual = None
        loss_r_ssim = None
        loss_r_offset = None
        loss_r_pruning = None
        loss_r_id = None
        comp_rgb = outputs['comp_rgb']
        if self.cfg.train.loss.pixel_weight > 0.:
            loss_r_pixel = self.pixel_loss_fn(comp_rgb, target_image)
            loss_renderer += loss_r_pixel * self.cfg.train.loss.pixel_weight
        if self.cfg.train.loss.perceptual_weight > 0.:
            with torch.autocast("cuda", enabled=False):
                loss_r_perceptual = self.perceptual_loss_fn(comp_rgb, target_image)
                loss_renderer += loss_r_perceptual * self.cfg.train.loss.perceptual_weight
        if self.cfg.train.loss.ssim_weight > 0.:
            with torch.autocast("cuda", enabled=False):
                loss_r_ssim = self.ssim_loss_fn(comp_rgb, target_image)
                loss_renderer += loss_r_ssim * self.cfg.train.loss.ssim_weight
        if self.cfg.train.loss.offset_weight > 0. and 'offsets' in outputs:
            loss_r_offset = self.offset_loss_fn(outputs['offsets']) * self.cfg.train.loss.offset_weight
            loss_renderer += loss_r_offset
        if self.cfg.train.loss.pruning_weight > 0. and self.cfg.model.gs_pruning:
            pruning_importance = outputs['pruning_importance']
            if pruning_importance is not None:
                loss_r_pruning = torch.mean(torch.sigmoid(pruning_importance))
            else:
                loss_r_pruning = torch.tensor(0.0, device=target_image.device)
            loss_renderer += loss_r_pruning ** 2 * self.cfg.train.loss.pruning_weight
        if getattr(self.cfg.train.loss, 'identity_weight', 0.0) > 0.:
            loss_r_id = self.id_loss_fn(comp_rgb, target_image)
            loss_renderer += loss_r_id * self.cfg.train.loss.identity_weight
        gs_stats = outputs.get('gs_stats', None)
        avg_remaining_gs = gs_stats['avg_remaining_gs'] if gs_stats is not None else None
        prune_percentage = gs_stats['prune_percentage'] if gs_stats is not None else None
        return loss_renderer, {
            'loss_renderer': loss_renderer if isinstance(loss_renderer, torch.Tensor) else torch.tensor(loss_renderer, device=target_image.device),
            'r_pixel': loss_r_pixel, 'r_perceptual': loss_r_perceptual, 'r_ssim': loss_r_ssim,
            'r_offset': loss_r_offset, 'r_pruning': loss_r_pruning, 'r_id': loss_r_id,
            'avg_remaining_gs': torch.tensor(avg_remaining_gs, device=target_image.device) if avg_remaining_gs is not None else None,
            'prune_percentage': torch.tensor(prune_percentage, device=target_image.device) if prune_percentage is not None else None,
        }

    def _match_token_batch_size(self, token, batch_size):
        if token.shape[0] == batch_size:
            return token
        if token.shape[0] == 1:
            return token.expand(batch_size, -1).contiguous()
        if token.shape[0] > batch_size:
            return token[:batch_size].contiguous()
        repeats = math.ceil(batch_size / token.shape[0])
        return token.repeat(repeats, 1)[:batch_size].contiguous()

    def _build_counterfactual_tokens(self, target_flame_params, donor_data, target_image):
        cf_cfg = self.cfg.train.counterfactual
        wrong_token_mode = str(getattr(cf_cfg, 'wrong_token_mode', 'shuffle_batch'))
        if wrong_token_mode != 'shuffle_batch':
            raise NotImplementedError(
                f"Unsupported train.counterfactual.wrong_token_mode={wrong_token_mode!r}; only 'shuffle_batch' is implemented."
            )

        model = self.accelerator.unwrap_model(self.model)
        correct_token = model.build_motion_token_input_from_flame(target_flame_params)
        zero_token = torch.zeros_like(correct_token)
        if donor_data is None:
            wrong_token = torch.roll(correct_token, shifts=1, dims=0) if correct_token.shape[0] > 1 else zero_token
        else:
            donor_data = self._move_batch_to_device(donor_data)
            _, donor_target_flame = self._build_flame_dicts(donor_data)
            wrong_token = model.build_motion_token_input_from_flame(donor_target_flame)
            wrong_token = self._match_token_batch_size(wrong_token, correct_token.shape[0])
            wrong_token = wrong_token.to(device=correct_token.device, dtype=correct_token.dtype)
        return correct_token, wrong_token, zero_token

    def _counterfactual_enabled(self):
        cf_cfg = getattr(self.cfg.train, 'counterfactual', None)
        return bool(getattr(cf_cfg, 'enabled', False)) and bool(getattr(self.cfg.model, 'motion_token_counterfactual_training', False))

    def _forward_loss_counterfactual_step(self, data, donor_data=None):
        target_image = data['target_rgbs']
        outputs_c, _, target_flame_params = self._forward_render(data, motion_token_input=None)
        loss_c, loss_parts = self._renderer_losses(outputs_c, target_image)
        correct_token, wrong_token, zero_token = self._build_counterfactual_tokens(target_flame_params, donor_data, target_image)

        outputs_e, _, _ = self._forward_render(data, motion_token_input=wrong_token)
        loss_e, _ = self._renderer_losses(outputs_e, target_image)
        outputs_f, _, _ = self._forward_render(data, motion_token_input=zero_token)
        loss_f, _ = self._renderer_losses(outputs_f, target_image)

        cf_cfg = self.cfg.train.counterfactual
        margin = float(getattr(cf_cfg, 'margin', 0.01))
        alpha = float(getattr(cf_cfg, 'alpha', 0.2))
        beta = float(getattr(cf_cfg, 'beta', 0.1))
        if bool(getattr(cf_cfg, 'detach_negative_losses', False)):
            loss_e_rank = loss_e.detach()
            loss_f_rank = loss_f.detach()
        else:
            loss_e_rank = loss_e
            loss_f_rank = loss_f
        rank_wrong = torch.relu(torch.as_tensor(margin, device=loss_c.device, dtype=loss_c.dtype) + loss_c - loss_e_rank)
        rank_zero = torch.relu(torch.as_tensor(margin, device=loss_c.device, dtype=loss_c.dtype) + loss_c - loss_f_rank)
        total_loss = loss_c + alpha * rank_wrong + beta * rank_zero
        loss_dict = {
            'total_loss': total_loss,
            'loss_renderer': loss_c,
            'r_pixel': loss_parts.get('r_pixel'),
            'r_perceptual': loss_parts.get('r_perceptual'),
            'r_ssim': loss_parts.get('r_ssim'),
            'r_offset': loss_parts.get('r_offset'),
            'r_pruning': loss_parts.get('r_pruning'),
            'r_id': loss_parts.get('r_id'),
            'avg_remaining_gs': loss_parts.get('avg_remaining_gs'),
            'prune_percentage': loss_parts.get('prune_percentage'),
            'cf_loss_correct': loss_c,
            'cf_loss_wrong': loss_e,
            'cf_loss_zero': loss_f,
            'cf_rank_wrong': rank_wrong,
            'cf_rank_zero': rank_zero,
            'cf_correct_lt_wrong': (loss_c.detach() < loss_e.detach()).float(),
            'cf_correct_lt_zero': (loss_c.detach() < loss_f.detach()).float(),
        }
        return outputs_c, total_loss, loss_dict

    def forward_loss_local_step(self, data, counterfactual_donor_data=None):
        if self._counterfactual_enabled():
            return self._forward_loss_counterfactual_step(data, donor_data=counterfactual_donor_data)
        uid = data['uid']
        input_image = data['rgbs']
        target_image = data['target_rgbs']
        input_c2ws = data['c2ws']
        target_c2ws = data['target_c2ws']
        input_intrs = data['intrs']
        target_intrs = data['target_intrs']
        input_bg_colors = data['bg_colors']
        target_bg_colors = data['target_bg_colors']
        input_masks = data['masks']
        target_masks = data['target_masks']
        
        # Concatenate input and target landmarks for full sequence tracking
        input_lms = data['landmarks']
        target_lms = data['target_landmarks']
        landmarks = torch.cat([input_lms, target_lms], dim=1)
        
        input_flame_params = {k.replace('input_', ''): v for k, v in data.items() 
                            if k.startswith('input_') and k.replace('input_', '') in 
                            ['expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'teeth_bs', 'translation', 'shape', 'betas']}
        
        target_flame_params = {k.replace('target_', ''): v for k, v in data.items() 
                             if k.startswith('target_') and k.replace('target_', '') in 
                             ['expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'teeth_bs', 'translation', 'shape', 'betas']}

        # Single forward pass → produces dual render outputs
        outputs = self.model(
            input_image=input_image,
            target_image=target_image,
            input_c2ws=input_c2ws,
            target_c2ws=target_c2ws,
            input_intrs=input_intrs,
            target_intrs=target_intrs,
            input_bg_colors=input_bg_colors,
            target_bg_colors=target_bg_colors,
            landmarks=landmarks,
            input_flame_params=input_flame_params,
            inf_flame_params=target_flame_params,
            uid=uid
        )
        outputs['full_landmarks'] = landmarks

        if getattr(self.cfg.model, 'debug_latent_smoke_loss', False):
            if 'latent_smoke_loss' not in outputs:
                raise RuntimeError("model.debug_latent_smoke_loss=True requires model output key 'latent_smoke_loss'")
            total_loss = outputs['latent_smoke_loss']
            loss_dict = {
                'total_loss': total_loss,
                'loss_renderer': total_loss,
                'latent_smoke_loss': total_loss,
                'r_pixel': None, 'r_perceptual': None, 'r_ssim': None,
                'r_offset': None, 'r_pruning': None, 'r_id': None,
                'avg_remaining_gs': None, 'prune_percentage': None,
            }
            del uid, input_image, target_image, input_c2ws, target_c2ws, input_intrs, target_intrs, input_bg_colors, target_bg_colors, input_masks, target_masks, landmarks
            return outputs, total_loss, loss_dict

        loss_renderer = 0.
        loss_r_pixel = None
        loss_r_perceptual = None
        loss_r_ssim = None
        loss_r_offset = None
        loss_r_pruning = None
        loss_r_id = None
        
        comp_rgb = outputs['comp_rgb']

        if self.cfg.train.loss.pixel_weight > 0.:
            loss_r_pixel = self.pixel_loss_fn(comp_rgb, target_image)
            loss_renderer += loss_r_pixel * self.cfg.train.loss.pixel_weight

        if self.cfg.train.loss.perceptual_weight > 0.:
            with torch.autocast("cuda", enabled=False):
                loss_r_perceptual = self.perceptual_loss_fn(comp_rgb, target_image)
                loss_renderer += loss_r_perceptual * self.cfg.train.loss.perceptual_weight

        if self.cfg.train.loss.ssim_weight > 0.:
            with torch.autocast("cuda", enabled=False):
                loss_r_ssim = self.ssim_loss_fn(comp_rgb, target_image)
                loss_renderer += loss_r_ssim * self.cfg.train.loss.ssim_weight
        
        if self.cfg.train.loss.offset_weight > 0. and 'offsets' in outputs:
            offsets = outputs['offsets']
            loss_r_offset = self.offset_loss_fn(offsets) * self.cfg.train.loss.offset_weight
            loss_renderer += loss_r_offset
        
        if self.cfg.train.loss.pruning_weight > 0. and self.cfg.model.gs_pruning:
            pruning_importance = outputs['pruning_importance']
            if pruning_importance is not None:
                loss_r_pruning = torch.mean(torch.sigmoid(pruning_importance))
            else:
                loss_r_pruning = torch.tensor(0.0, device=target_image.device)
            loss_renderer += loss_r_pruning ** 2 * self.cfg.train.loss.pruning_weight

        if getattr(self.cfg.train.loss, 'identity_weight', 0.0) > 0.:
            loss_r_id = self.id_loss_fn(comp_rgb, target_image)
            loss_renderer += loss_r_id * self.cfg.train.loss.identity_weight

        total_loss = loss_renderer

        # Extract GS stats if available
        gs_stats = outputs.get('gs_stats', None)
        avg_remaining_gs = gs_stats['avg_remaining_gs'] if gs_stats is not None else None
        prune_percentage = gs_stats['prune_percentage'] if gs_stats is not None else None

        # Build loss dict for logging
        loss_dict = {
            'total_loss': total_loss,
            'loss_renderer': loss_renderer if isinstance(loss_renderer, torch.Tensor) else torch.tensor(loss_renderer, device=target_image.device),
            'r_pixel': loss_r_pixel, 'r_perceptual': loss_r_perceptual, 'r_ssim': loss_r_ssim,
            'r_offset': loss_r_offset, 'r_pruning': loss_r_pruning, 'r_id': loss_r_id,
            'avg_remaining_gs': torch.tensor(avg_remaining_gs, device=target_image.device) if avg_remaining_gs is not None else None,
            'prune_percentage': torch.tensor(prune_percentage, device=target_image.device) if prune_percentage is not None else None,
        }

        # Clean up intermediate variables
        del uid, input_image, target_image, input_c2ws, target_c2ws, input_intrs, target_intrs, input_bg_colors, target_bg_colors, input_masks, target_masks, landmarks

        return outputs, total_loss, loss_dict
    

    def train_epoch(self, loader: torch.utils.data.DataLoader, profiler: torch.profiler.profile):
        self.model.train()

        global_step_losses = []

        logger.debug(f"======== Starting epoch {self.current_epoch} ========")
        counterfactual_donor_iter = None
        if self._counterfactual_enabled():
            counterfactual_donor_iter = iter(loader)
            next(counterfactual_donor_iter, None)

        for data in loader:
            logger.debug(f"======== Starting global step {self.global_step} ========")

            with self.accelerator.accumulate(self.model):
                # Single forward pass or counterfactual triple-forward loss.
                counterfactual_donor_data = None
                if counterfactual_donor_iter is not None:
                    try:
                        counterfactual_donor_data = next(counterfactual_donor_iter)
                    except StopIteration:
                        counterfactual_donor_iter = iter(loader)
                        counterfactual_donor_data = next(counterfactual_donor_iter, None)
                outputs, total_loss, loss_dict = self.forward_loss_local_step(data, counterfactual_donor_data=counterfactual_donor_data)

                # Single backward + step
                self.accelerator.backward(total_loss)
                if getattr(self.cfg.model, 'debug_latent_smoke_loss', False) and self.accelerator.is_main_process:
                    grad_sq_sum = 0.0
                    grad_param_count = 0
                    for name, param in self.model.named_parameters():
                        if name.startswith('motion_token_adapter.') and param.grad is not None:
                            grad_sq_sum += float(param.grad.detach().float().pow(2).sum().cpu())
                            grad_param_count += param.numel()
                    grad_norm = math.sqrt(grad_sq_sum) if grad_sq_sum > 0.0 else 0.0
                    print(f"[P9.2 latent smoke] MotionTokenAdapter grad_norm={grad_norm:.6e} grad_param_count={grad_param_count}")

                if self.accelerator.sync_gradients and self.cfg.train.optim.clip_grad_norm > 0.:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.train.optim.clip_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Logging
            if self.accelerator.sync_gradients:
                profiler.step()
                self.scheduler.step()
                
                self.global_step += 1
                
                # Gather & Mean across processes
                log_kwargs = {}
                for k, v in loss_dict.items():
                    if v is None:
                        continue
                    v_detached = v.detach() if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device)
                    gathered = self.accelerator.gather(v_detached)
                    if gathered.numel() > 0:
                        log_kwargs[k] = gathered.mean().item()
                    else:
                        log_kwargs[k] = v_detached.item()

                self.log_scalar_kwargs(
                    step=self.global_step, split='train',
                    **log_kwargs
                )
                # Log learning rates
                self.accelerator.log({
                    'opt/lr': self.optimizer.param_groups[0]['lr'],
                }, self.global_step)
                
                # Helper
                def get_val(key): return log_kwargs.get(key, float('nan'))
                
                # Epoch summary tensor
                legacy_loss_tensor = torch.tensor([
                    get_val('total_loss'), get_val('r_pixel'), get_val('r_perceptual'), get_val('r_ssim'),
                    get_val('r_offset'), get_val('r_pruning'), 
                    get_val('avg_remaining_gs'), get_val('prune_percentage'),
                    get_val('r_id')
                ], device=self.device)
                global_step_losses.append(legacy_loss_tensor)

                # Step reporting
                step_info = f"Step[{self.global_step}/{self.N_max_global_steps}] "
                step_info += f"Loss:{get_val('total_loss'):.4f} "
                step_info += f"Px:{get_val('r_pixel'):.4f} "
                if not math.isnan(get_val('r_perceptual')):
                    step_info += f"Perc:{get_val('r_perceptual'):.4f} "
                if not math.isnan(get_val('r_ssim')):
                    step_info += f"SSIM:{get_val('r_ssim'):.4f} "
                if not math.isnan(get_val('r_id')):
                    step_info += f"Id:{get_val('r_id'):.4f} "
                if not math.isnan(get_val('cf_loss_correct')):
                    step_info += f"LC:{get_val('cf_loss_correct'):.4f} "
                if not math.isnan(get_val('cf_loss_wrong')):
                    step_info += f"LE:{get_val('cf_loss_wrong'):.4f} "
                if not math.isnan(get_val('cf_loss_zero')):
                    step_info += f"LF:{get_val('cf_loss_zero'):.4f} "
                if not math.isnan(get_val('cf_rank_wrong')):
                    step_info += f"RW:{get_val('cf_rank_wrong'):.4f} "
                if not math.isnan(get_val('cf_rank_zero')):
                    step_info += f"RZ:{get_val('cf_rank_zero'):.4f} "
                if not math.isnan(get_val('cf_correct_lt_wrong')):
                    step_info += f"C<W:{get_val('cf_correct_lt_wrong'):.2f} "
                if not math.isnan(get_val('cf_correct_lt_zero')):
                    step_info += f"C<Z:{get_val('cf_correct_lt_zero'):.2f} "
                # GS stats
                if not math.isnan(get_val('avg_remaining_gs')):
                    step_info += f"GS:{get_val('avg_remaining_gs')/1000:.1f}K "
                if not math.isnan(get_val('prune_percentage')):
                    step_info += f"Prune:{get_val('prune_percentage'):.1f}% "
                # LR
                step_info += f"lr:{self.optimizer.param_groups[0]['lr']:.2e}"
                
                if self.accelerator.is_main_process:
                    print(step_info)

                # periodic actions
                if self.global_step % self.cfg.saver.checkpoint_global_steps == 0:
                    self.save_checkpoint()
                if self.global_step % self.cfg.val.global_step_period == 0:
                    if getattr(self.cfg.val, 'skip_eval', False):
                        logger.info("Skip evaluation because val.skip_eval=true")
                    else:
                        self.evaluate()
                    self.model.train()
                
                del data
                clear_memory()

                # progress control
                if self.global_step >= self.N_max_global_steps:
                    self.accelerator.set_trigger()
                    break
    

        # track epoch
        self.current_epoch += 1
        epoch_losses = torch.stack(global_step_losses).mean(dim=0)
        epoch_loss, epoch_r_pixel, epoch_r_perceptual, epoch_r_ssim, epoch_r_offset, epoch_r_pruning, epoch_avg_remaining_gs, epoch_prune_percentage, epoch_r_id = epoch_losses.unbind()
        epoch_loss_dict = {
            'total_loss': epoch_loss.item(),
            'r_pixel': epoch_r_pixel.item(),
            'r_perceptual': epoch_r_perceptual.item(),
            'r_ssim': epoch_r_ssim.item() if not math.isnan(epoch_r_ssim.item()) else float('nan'),
            'r_offset': epoch_r_offset.item() if not math.isnan(epoch_r_offset.item()) else float('nan'),
            'r_pruning': epoch_r_pruning.item() if not math.isnan(epoch_r_pruning.item()) else float('nan'),
            'r_id': epoch_r_id.item() if not math.isnan(epoch_r_id.item()) else float('nan'),
        }
        self.log_scalar_kwargs(
            step=self.global_step, split='train',
            **epoch_loss_dict
        )
        logger.info(
            f'[TRAIN EPOCH] {self.current_epoch}/{self.cfg.train.epochs}: ' + \
                ', '.join(f'{k}={v:.4f}' for k, v in epoch_loss_dict.items() if not math.isnan(v))
        )
        
        clear_memory()

    def train(self):
        
        starting_local_step_in_epoch = self.global_step_in_epoch * self.cfg.train.accum_steps
        skipped_loader = self.accelerator.skip_first_batches(self.train_loader, starting_local_step_in_epoch)
        logger.info(f"======== Skipped {starting_local_step_in_epoch} local batches ========")

        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=10, warmup=10, active=100,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(
                self.cfg.logger.tracker_root,
                self.cfg.experiment.parent, self.cfg.experiment.child,
            )),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) if self.cfg.logger.enable_profiler else DummyProfiler()
        
        with profiler:

            self.optimizer.zero_grad()
            for _ in range(self.current_epoch, self.cfg.train.epochs):

                loader = skipped_loader or self.train_loader
                skipped_loader = None
                self.train_epoch(loader=loader, profiler=profiler)
                if self.accelerator.check_trigger():
                    break

            logger.info(f"======== Training finished at global step {self.global_step} ========")

            # final checkpoint and evaluation
            self.save_checkpoint()
            if getattr(self.cfg.val, 'skip_eval', False):
                logger.info("Skip evaluation because val.skip_eval=true")
            else:
                self.evaluate()
    
    @torch.no_grad()
    @torch.compiler.disable
    def evaluate(self, epoch: int = None):
        self.model.eval()

        max_val_batches = self.cfg.val.debug_batches or len(self.val_loader)
        running_losses = []
        batch_idx = 0

        for data in tqdm(self.val_loader, disable=(not self.accelerator.is_main_process), total=max_val_batches):

            if len(running_losses) >= max_val_batches:
                logger.info(f"======== Early stop validation at {len(running_losses)} batches ========")
                break

            outs, total_loss, loss_dict = self.forward_loss_local_step(data)
            
            # Save images for each batch
            if self.cfg.val.save_images:
                n = self.cfg.val.samples_per_save
                self.save_val_images(
                    epoch=epoch,
                    step=self.global_step if epoch is None else None,
                    batch_idx=batch_idx,
                    gts=data['target_rgbs'][:n].cpu(),
                    renders=outs['comp_rgb'][:n].cpu(),
                    uids=data['uid'],
                )

            # Build loss tensor for gathering (includes both GT Pose and Pred Pose photometric metrics)
            loss_keys = ['total_loss', 'r_pixel', 'r_perceptual', 'r_ssim', 'r_offset', 'r_pruning',
                        'avg_remaining_gs', 'prune_percentage', 'r_id', 'loss_renderer']
            loss_tensor = torch.stack([
                loss_dict[k].detach() if loss_dict.get(k) is not None and isinstance(loss_dict[k], torch.Tensor) 
                else (torch.tensor(loss_dict[k], device=self.device) if loss_dict.get(k) is not None 
                      else torch.tensor(float('nan'), device=self.device))
                for k in loss_keys
            ])
            running_losses.append(loss_tensor)
            batch_idx += 1
        
        if not running_losses:
            logger.warning("No validation batches were evaluated; skip loss aggregation.")
            self.model.train()
            clear_memory()
            return {}

        total_losses = self.accelerator.gather(torch.stack(running_losses)).mean(dim=0).cpu()
        total_vals = total_losses.unbind()
        
        del total_losses, running_losses
        
        total_loss_dict = {}
        for i, k in enumerate(loss_keys):
            v = total_vals[i].item()
            total_loss_dict[k] = v if not math.isnan(v) else float('nan')

        if epoch is not None:
            self.log_scalar_kwargs(
                epoch=epoch, split='val',
                **total_loss_dict,
            )
            logger.info(
                f'[VAL EPOCH] {epoch}/{self.cfg.train.epochs}: ' + \
                    ', '.join(f'{k}={v:.4f}' for k, v in total_loss_dict.items() if not math.isnan(v))
            )
        else:
            self.log_scalar_kwargs(
                step=self.global_step, split='val',
                **total_loss_dict,
            )
            logger.info(
                f'[VAL] Step {self.global_step}: loss={total_loss_dict["total_loss"]:.3f}'
            )
        
        def _safe(k): 
            v = total_loss_dict.get(k, float('nan'))
            return v if not math.isnan(v) else float('nan')
        logger.info(f'[VAL] pixel={_safe("r_pixel"):.4f}, perceptual={_safe("r_perceptual"):.4f}, ssim={_safe("r_ssim"):.4f}')
        
        clear_memory()



    @Trainer.control('on_main_process')
    def save_val_images(
        self, epoch: int = None, step: int = None, batch_idx: int = None,
        gts: torch.Tensor = None, renders: torch.Tensor = None,
        uids: list = None,
        ):
        """Save validation images: 2 rows — GT, Rendered"""
        import os
        from torchvision.utils import save_image
        
        step_dir = f"step_{step:06d}" if step is not None else f"epoch_{epoch:03d}" if epoch is not None else "val"
        
        save_dir = os.path.join(
            self.cfg.val.save_dir,
            self.cfg.experiment.parent,
            self.cfg.experiment.child,
            step_dir
        )
        os.makedirs(save_dir, exist_ok=True)
        
        M = gts.shape[1]   # Number of target frames
        num_samples = gts.shape[0]

        for sample_idx in range(num_samples):
            # 2 rows: GT | Rendered
            rows = [gts[sample_idx]]
            if renders is not None:
                rows.append(renders[sample_idx])

            merged = torch.stack(rows, dim=0).view(-1, *gts.shape[2:])  # [N_rows*M, C, H, W]
            nrow = M

            if merged.dtype != torch.float32:
                merged = merged.float()

            # Generate filename from uid
            path, frame_data = None, None
            if uids is not None:
                if isinstance(uids, list) and len(uids) == 2:
                    paths_list, frame_data_dict = uids
                    if isinstance(paths_list, list) and sample_idx < len(paths_list):
                        path = paths_list[sample_idx]
                        frame_data = frame_data_dict
                elif isinstance(uids, tuple) and len(uids) == 2:
                    paths_list, frame_data_list = uids
                    if isinstance(paths_list, (tuple, list)) and sample_idx < len(paths_list):
                        path = paths_list[sample_idx]
                        frame_data = frame_data_list[sample_idx] if sample_idx < len(frame_data_list) else None
                    elif sample_idx == 0:
                        path = paths_list
                        frame_data = frame_data_list
                elif isinstance(uids, list) and sample_idx < len(uids):
                    uid_sample = uids[sample_idx]
                    if isinstance(uid_sample, tuple) and len(uid_sample) == 2:
                        path, frame_data = uid_sample

            assert path is not None, f"Invalid path for sample {sample_idx}, uids structure: {type(uids)}, uids: {uids}"

            path_parts = path.split('/')
            person_id = path_parts[0]
            dataset_src = "Nersemble" if person_id.isdigit() else "VFHQ"
            input_frames_raw = frame_data.get('input_frames', M) if frame_data else M
            if hasattr(input_frames_raw, 'item'):
                input_frames = input_frames_raw.item()
            else:
                input_frames = int(input_frames_raw)
            batch_suffix = f"_batch{batch_idx:03d}" if batch_idx is not None else ""
            filename = f"{dataset_src}_{person_id}_input{input_frames}{batch_suffix}.png"

            save_image(merged, os.path.join(save_dir, filename), nrow=nrow, normalize=True)

        clear_memory()


    def log_optimizer(self, epoch: int = None, step: int = None, attrs: list = [], group_ids: list = []):
        assert self.optimizer is not None, 'Optimizer is not initialized'
        log_progress = step if step is not None else epoch
        for group_id, group in enumerate(self.optimizer.param_groups):
            for attr in attrs:
                if attr in group:
                    group_name = group.get('name', str(group_id))
                    self.accelerator.log({f'opt/{group_name}/{attr}': group[attr]}, log_progress)