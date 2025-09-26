"""Distributed training with DeepSpeed integration."""
import deepspeed
from transformers import TrainingArguments
import torch.distributed as dist

class DistributedTrainer:
    """Handle distributed training across multiple GPUs/nodes."""
    
    def __init__(self, config_path: str):
        self.ds_config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "fp16": {
                "enabled": "auto",
                "loss_scale": 0,
                "loss_scale_window": 1000,
            },
            "zero_optimization": {
                "stage": 3,  # ZeRO Stage 3 for maximum memory efficiency
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True
            },
            "gradient_clipping": 1.0,
            "steps_per_print": 10,
            "wall_clock_breakdown": False
        }
    
    def setup_distributed_environment(self):
        """Initialize distributed training environment."""
        if not dist.is_initialized():
            deepspeed.init_distributed()
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        logger.info(f"Initialized distributed training: Rank {rank}/{world_size}")
        
        return world_size, rank
    
    def create_training_arguments(self, base_args: TrainingArguments) -> TrainingArguments:
        """Enhance training arguments for distributed training."""
        base_args.deepspeed = self.ds_config
        base_args.local_rank = dist.get_rank() if dist.is_initialized() else -1
        base_args.ddp_find_unused_parameters = False
        base_args.gradient_checkpointing = True
        
        return base_args
