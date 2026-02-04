# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Optimizer utilities for the Trainer class.
"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import TYPE_CHECKING, Any

import torch
from packaging import version
from torch import nn

from .optimization import Adafactor
from .trainer_pt_utils import LayerWiseDummyOptimizer
from .trainer_utils import check_target_module_exists
from .training_args import OptimizerNames, ParallelMode
from .utils import (
    is_apollo_torch_available,
    is_bitsandbytes_available,
    is_galore_torch_available,
    is_grokadamw_available,
    is_lomo_available,
    is_schedulefree_available,
    is_torch_optimi_available,
    is_torchao_available,
    strtobool,
)


if TYPE_CHECKING:
    from .modeling_utils import PreTrainedModel
    from .training_args import TrainingArguments

logger = logging.getLogger(__name__)

# Optimizer groups for cleaner dispatch
_BITSANDBYTES_OPTIMIZERS = {
    OptimizerNames.ADAMW_BNB,
    OptimizerNames.ADAMW_8BIT,
    OptimizerNames.PAGED_ADAMW,
    OptimizerNames.PAGED_ADAMW_8BIT,
    OptimizerNames.ADEMAMIX,
    OptimizerNames.ADEMAMIX_8BIT,
    OptimizerNames.PAGED_ADEMAMIX,
    OptimizerNames.PAGED_ADEMAMIX_8BIT,
    OptimizerNames.LION,
    OptimizerNames.LION_8BIT,
    OptimizerNames.PAGED_LION,
    OptimizerNames.PAGED_LION_8BIT,
    OptimizerNames.RMSPROP_BNB,
    OptimizerNames.RMSPROP_8BIT,
    OptimizerNames.RMSPROP_32BIT,
}

_GALORE_OPTIMIZERS = {
    OptimizerNames.GALORE_ADAMW,
    OptimizerNames.GALORE_ADAMW_8BIT,
    OptimizerNames.GALORE_ADAFACTOR,
    OptimizerNames.GALORE_ADAMW_LAYERWISE,
    OptimizerNames.GALORE_ADAMW_8BIT_LAYERWISE,
    OptimizerNames.GALORE_ADAFACTOR_LAYERWISE,
}

_APOLLO_OPTIMIZERS = {
    OptimizerNames.APOLLO_ADAMW,
    OptimizerNames.APOLLO_ADAMW_LAYERWISE,
}

_SCHEDULE_FREE_OPTIMIZERS = {
    OptimizerNames.SCHEDULE_FREE_RADAM,
    OptimizerNames.SCHEDULE_FREE_ADAMW,
    OptimizerNames.SCHEDULE_FREE_SGD,
}

_TORCHAO_OPTIMIZERS = {
    OptimizerNames.ADAMW_TORCH_4BIT,
    OptimizerNames.ADAMW_TORCH_8BIT,
}


def _parse_optim_args(optim_args_str: str | None) -> dict[str, str]:
    """Parse optimizer arguments from a comma-separated string."""
    if not optim_args_str:
        return {}
    optim_args = {}
    for mapping in optim_args_str.replace(" ", "").split(","):
        key, value = mapping.split("=")
        optim_args[key] = value
    return optim_args


def _setup_low_rank_optimizer(
    args: TrainingArguments,
    model: PreTrainedModel,
    optimizer_name: str,
    optimizer_mapping: dict[str, Any],
    optim_kwargs: dict[str, Any],
    optimizer_kwargs: dict[str, Any],
    is_layerwise_supported: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """
    Helper function to set up low-rank optimizers like GaLore and Apollo.

    These optimizers apply low-rank projections to specific target modules (typically linear layers).
    """
    is_layerwise = optimizer_name.lower().endswith("layerwise")
    if is_layerwise and args.parallel_mode == ParallelMode.DISTRIBUTED and is_layerwise_supported:
        raise NotImplementedError(f"Layer-wise {optimizer_name} does not support DDP at this time")

    optimizer_cls = optimizer_mapping[optimizer_name]

    if args.optim_target_modules is None:
        raise ValueError(f"You need to define `optim_target_modules` to use {optimizer_name} optimizers")

    if not isinstance(args.optim_target_modules, (list, str)):
        raise TypeError(
            f"`optim_target_modules` must be a list of strings, a regex string, or 'all-linear'. "
            f"Got: {args.optim_target_modules}"
        )

    if model is None:
        raise ValueError(f"You need to pass a model to initialize {optimizer_name} optimizer.")

    all_linear = (
        isinstance(args.optim_target_modules, str) and args.optim_target_modules.replace("_", "-") == "all-linear"
    )

    target_params_names = []
    for module_name, module in model.named_modules():
        target_module_exists, is_regex = check_target_module_exists(
            args.optim_target_modules, module_name, return_is_regex=True
        )

        if not isinstance(module, nn.Linear):
            if target_module_exists and not is_regex:
                logger.warning(f"{module_name} matched but ignored. {optimizer_name} only supports linear layers.")
            continue

        if not target_module_exists and not all_linear:
            continue

        target_params_names.append(module_name + ".weight")

    if len(target_params_names) == 0:
        raise ValueError(f"No target modules found for {optimizer_name} ({args.optim_target_modules}).")

    target_params = [p for n, p in model.named_parameters() if n in target_params_names]
    non_target_params = [p for n, p in model.named_parameters() if n not in target_params_names]

    param_groups = [
        {"params": non_target_params},
        {"params": target_params, **optim_kwargs},
    ]

    if is_layerwise:
        if args.gradient_accumulation_steps != 1:
            raise ValueError(f"Layerwise {optimizer_name} does not support gradient accumulation!")

        optimizer_dict = {}
        for param in non_target_params:
            optimizer_dict[param] = optimizer_cls([{"params": [param]}], **optimizer_kwargs)
        for param in target_params:
            optimizer_dict[param] = optimizer_cls([{"params": [param], **optim_kwargs}], **optimizer_kwargs)

        def optimizer_hook(param):
            if param.grad is not None:
                optimizer_dict[param].step()
                optimizer_dict[param].zero_grad()

        for param in model.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(optimizer_hook)

        optimizer_cls = LayerWiseDummyOptimizer
        optimizer_kwargs.update({"optimizer_dict": optimizer_dict})

    optimizer_kwargs.update({"params": param_groups})
    return optimizer_cls, optimizer_kwargs


# =============================================================================
# Individual optimizer handlers
# =============================================================================


def _get_adafactor(optimizer_kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Get Adafactor optimizer."""
    optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
    return Adafactor, optimizer_kwargs


def _get_adamw_torch(
    optim: OptimizerNames,
    optimizer_kwargs: dict[str, Any],
    adam_kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Get PyTorch AdamW optimizer (regular or fused)."""
    from torch.optim import AdamW

    optimizer_kwargs.update(adam_kwargs)
    if optim == OptimizerNames.ADAMW_TORCH_FUSED:
        optimizer_kwargs.update({"fused": True})
    return AdamW, optimizer_kwargs


def _get_adamw_torch_xla(
    optimizer_kwargs: dict[str, Any],
    adam_kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Get Torch XLA syncfree AdamW optimizer."""
    try:
        from torch_xla.amp.syncfree import AdamW

        optimizer_kwargs.update(adam_kwargs)
        return AdamW, optimizer_kwargs
    except ImportError:
        raise ValueError("Trainer failed to import syncfree AdamW from torch_xla.")


def _get_adamw_torch_npu_fused(
    optimizer_kwargs: dict[str, Any],
    adam_kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Get NPU Fused AdamW optimizer."""
    try:
        from torch_npu.optim import NpuFusedAdamW

        optimizer_kwargs.update(adam_kwargs)
        return NpuFusedAdamW, optimizer_kwargs
    except ImportError:
        raise ValueError("Trainer failed to import FusedAdamW from torch_npu.")


def _get_adamw_apex_fused(
    optimizer_kwargs: dict[str, Any],
    adam_kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Get Apex Fused Adam optimizer."""
    try:
        from apex.optimizers import FusedAdam

        optimizer_kwargs.update(adam_kwargs)
        return FusedAdam, optimizer_kwargs
    except ImportError:
        raise ValueError("Trainer tried to instantiate apex FusedAdam but apex is not installed!")


def _get_bitsandbytes_optimizer(
    args: TrainingArguments,
    optimizer_kwargs: dict[str, Any],
    adam_kwargs: dict[str, Any],
    optim_args: dict[str, str],
) -> tuple[Any, dict[str, Any]]:
    """Get bitsandbytes optimizer (AdamW, Lion, RMSprop variants)."""
    if not is_bitsandbytes_available():
        raise ImportError(
            "You need to install `bitsandbytes` in order to use bitsandbytes optimizers: `pip install -U bitsandbytes`"
        )

    from bitsandbytes.optim import AdamW, Lion, RMSprop

    optim_name = args.optim
    is_paged = "paged" in optim_name
    optim_bits = 8 if "8bit" in optim_name else 32
    optimizer_cls = None
    additional_optim_kwargs = adam_kwargs

    if "adam" in optim_name:
        optimizer_cls = AdamW
    elif "lion" in optim_name:
        optimizer_cls = Lion
        additional_optim_kwargs = {"betas": (args.adam_beta1, args.adam_beta2)}
    elif "rmsprop" in optim_name:
        optimizer_cls = RMSprop
        additional_optim_kwargs = optim_args
    elif "ademamix" in optim_name:
        from bitsandbytes.optim import AdEMAMix

        optimizer_cls = AdEMAMix
        additional_optim_kwargs = {
            "betas": (
                float(optim_args.get("beta1", args.adam_beta1)),
                float(optim_args.get("beta2", args.adam_beta2)),
                float(optim_args.get("beta3", 0.9999)),
            ),
            "alpha": float(optim_args.get("alpha", 5.0)),
            "eps": float(optim_args.get("eps", args.adam_epsilon)),
        }
        if "t_alpha" in optim_args:
            additional_optim_kwargs["t_alpha"] = int(optim_args["t_alpha"])
        if "t_beta3" in optim_args:
            additional_optim_kwargs["t_beta3"] = int(optim_args["t_beta3"])

    bnb_kwargs = {"optim_bits": optim_bits}
    if "rmsprop" not in optim_name:
        bnb_kwargs["is_paged"] = is_paged

    optimizer_kwargs.update(additional_optim_kwargs)
    optimizer_kwargs.update(bnb_kwargs)
    return optimizer_cls, optimizer_kwargs


def _get_adamw_anyprecision(
    optimizer_kwargs: dict[str, Any],
    adam_kwargs: dict[str, Any],
    optim_args: dict[str, str],
) -> tuple[Any, dict[str, Any]]:
    """Get AnyPrecision AdamW optimizer."""
    try:
        from torchdistx.optimizers import AnyPrecisionAdamW

        optimizer_kwargs.update(adam_kwargs)
        optimizer_kwargs.update(
            {
                "use_kahan_summation": strtobool(optim_args.get("use_kahan_summation", "False")),
                "momentum_dtype": getattr(torch, optim_args.get("momentum_dtype", "float32")),
                "variance_dtype": getattr(torch, optim_args.get("variance_dtype", "float32")),
                "compensation_buffer_dtype": getattr(torch, optim_args.get("compensation_buffer_dtype", "bfloat16")),
            }
        )
        return AnyPrecisionAdamW, optimizer_kwargs
    except ImportError:
        raise ValueError("Please install https://github.com/pytorch/torchdistx")


def _get_sgd(optimizer_kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Get SGD optimizer."""
    return torch.optim.SGD, optimizer_kwargs


def _get_adagrad(optimizer_kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Get Adagrad optimizer."""
    return torch.optim.Adagrad, optimizer_kwargs


def _get_rmsprop(optimizer_kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Get RMSprop optimizer."""
    return torch.optim.RMSprop, optimizer_kwargs


def _get_galore_optimizer(
    args: TrainingArguments,
    model: PreTrainedModel,
    optimizer_kwargs: dict[str, Any],
    optim_args: dict[str, str],
) -> tuple[Any, dict[str, Any]]:
    """Get GaLore optimizer."""
    if not is_galore_torch_available():
        raise ImportError(
            "You need to install `galore_torch` in order to use GaLore optimizers. "
            "Install it with `pip install git+https://github.com/jiaweizzhao/GaLore`"
        )
    from galore_torch import GaLoreAdafactor, GaLoreAdamW, GaLoreAdamW8bit

    optimizer_mapping = {
        OptimizerNames.GALORE_ADAMW: GaLoreAdamW,
        OptimizerNames.GALORE_ADAMW_8BIT: GaLoreAdamW8bit,
        OptimizerNames.GALORE_ADAFACTOR: GaLoreAdafactor,
        OptimizerNames.GALORE_ADAMW_LAYERWISE: GaLoreAdamW,
        OptimizerNames.GALORE_ADAMW_8BIT_LAYERWISE: GaLoreAdamW8bit,
        OptimizerNames.GALORE_ADAFACTOR_LAYERWISE: GaLoreAdafactor,
    }

    galore_optim_kwargs = {
        "rank": int(optim_args.pop("rank", 128)),
        "update_proj_gap": int(optim_args.pop("update_proj_gap", 200)),
        "scale": float(optim_args.pop("scale", 0.25)),
        "proj_type": optim_args.pop("proj_type", "std"),
    }

    optimizer_cls, optimizer_kwargs = _setup_low_rank_optimizer(
        args, model, args.optim, optimizer_mapping, galore_optim_kwargs, optimizer_kwargs
    )
    if args.optim == OptimizerNames.GALORE_ADAFACTOR:
        optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
    return optimizer_cls, optimizer_kwargs


def _get_apollo_optimizer(
    args: TrainingArguments,
    model: PreTrainedModel,
    optimizer_kwargs: dict[str, Any],
    adam_kwargs: dict[str, Any],
    optim_args: dict[str, str],
) -> tuple[Any, dict[str, Any]]:
    """Get Apollo optimizer."""
    if not is_apollo_torch_available():
        raise ImportError(
            "You need to install `apollo_torch` in order to use APOLLO optimizers. "
            "Install it with `pip install git+https://github.com/zhuhanqing/APOLLO`"
        )
    from apollo_torch import APOLLOAdamW

    optimizer_mapping = {
        OptimizerNames.APOLLO_ADAMW: APOLLOAdamW,
        OptimizerNames.APOLLO_ADAMW_LAYERWISE: APOLLOAdamW,
    }

    apollo_optim_kwargs = {
        "rank": int(optim_args.pop("rank", 128)),
        "proj": optim_args.pop("proj", "random"),
        "scale_type": optim_args.pop("scale_type", "channel"),
        "update_proj_gap": int(optim_args.pop("update_proj_gap", 200)),
        "scale": float(optim_args.pop("scale", 1.0)),
        "proj_type": optim_args.pop("proj_type", "std"),
    }
    apollo_optim_kwargs.update(adam_kwargs)

    return _setup_low_rank_optimizer(args, model, args.optim, optimizer_mapping, apollo_optim_kwargs, optimizer_kwargs)


def _get_lomo_optimizer(
    args: TrainingArguments,
    model: PreTrainedModel,
    optimizer_kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Get LOMO optimizer."""
    if not is_lomo_available():
        raise ImportError(
            "You need to install `lomo_optim` in order to use LOMO optimizers. "
            "Install it with `pip install lomo-optim`"
        )

    if model is None:
        raise ValueError("You need to pass a `model` in order to correctly initialize a LOMO optimizer.")

    from lomo_optim import AdaLomo, Lomo

    optimizer_cls = AdaLomo if "ada" in args.optim else Lomo
    optimizer_kwargs.update({"model": model})
    return optimizer_cls, optimizer_kwargs


def _get_grokadamw(
    optimizer_kwargs: dict[str, Any],
    optim_args: dict[str, str],
) -> tuple[Any, dict[str, Any]]:
    """Get GrokAdamW optimizer."""
    if not is_grokadamw_available():
        raise ValueError("Please install grokadamw with `pip install grokadamw`")

    from grokadamw import GrokAdamW

    optimizer_kwargs.update(
        {
            "alpha_init": float(optim_args.get("alpha_init", 0.98)),
            "lamb": float(optim_args.get("lamb", 2.0)),
            "gamma": float(optim_args.get("gamma", 0.1)),
            "grokking_signal_decay_rate": float(optim_args.get("grokking_signal_decay_rate", 0.1)),
            "gradient_clipping": float(optim_args.get("gradient_clipping", 1.0)),
        }
    )
    return GrokAdamW, optimizer_kwargs


def _get_torchao_optimizer(
    args: TrainingArguments,
    optimizer_kwargs: dict[str, Any],
    adam_kwargs: dict[str, Any],
    optim_args: dict[str, str],
) -> tuple[Any, dict[str, Any]]:
    """Get TorchAO 4-bit or 8-bit optimizer."""
    if not is_torchao_available() or version.parse(importlib.metadata.version("torchao")) < version.parse("0.4.0"):
        raise ImportError(
            "You need to have `torchao>=0.4.0` in order to use torch 4-bit optimizers. "
            "Install it with `pip install torchao` or follow the instructions here: "
            "https://github.com/pytorch/ao"
        )
    if version.parse(importlib.metadata.version("torch")) <= version.parse("2.4"):
        raise ImportError(
            "You need to have `torch>2.4` in order to use torch 4-bit optimizers. "
            "Install it with `pip install --upgrade torch` it is available on pipy. "
            "Otherwise, you need to install torch nightly."
        )

    if version.parse(importlib.metadata.version("torchao")) >= version.parse("0.11.0"):
        from torchao.optim import AdamW4bit, AdamW8bit
    else:
        from torchao.prototype.low_bit_optim import AdamW4bit, AdamW8bit

    if args.optim == OptimizerNames.ADAMW_TORCH_4BIT:
        optimizer_cls = AdamW4bit
    else:
        optimizer_cls = AdamW8bit

    optimizer_kwargs.update(
        {
            "block_size": optim_args.get("block_size", 256),
            "bf16_stochastic_round": strtobool(optim_args.get("bf16_stochastic_round", "False")),
        }
    )
    optimizer_kwargs.update(adam_kwargs)
    return optimizer_cls, optimizer_kwargs


def _get_schedule_free_optimizer(
    args: TrainingArguments,
    optimizer_kwargs: dict[str, Any],
    adam_kwargs: dict[str, Any],
    optim_args: dict[str, str],
) -> tuple[Any, dict[str, Any]]:
    """Get ScheduleFree optimizer."""
    if not is_schedulefree_available():
        raise ImportError(
            "You need to install `schedulefree` in order to use schedulefree optimizers. "
            "Install it with `pip install schedulefree.`"
        )
    from schedulefree import AdamWScheduleFree, SGDScheduleFree

    additional_optim_kwargs = {}
    require_warmup = True

    if args.optim == OptimizerNames.SCHEDULE_FREE_RADAM:
        if not is_schedulefree_available("1.4.0"):
            raise ImportError(
                "You need to install `schedulefree>=1.4.0` in order to use RAdamScheduleFree optimizer. "
                "Install it with `pip install schedulefree.`"
            )
        from schedulefree import RAdamScheduleFree

        optimizer_cls = RAdamScheduleFree
        additional_optim_kwargs = adam_kwargs
        require_warmup = False
    elif args.optim == OptimizerNames.SCHEDULE_FREE_ADAMW:
        optimizer_cls = AdamWScheduleFree
        additional_optim_kwargs = adam_kwargs
    elif args.optim == OptimizerNames.SCHEDULE_FREE_SGD:
        optimizer_cls = SGDScheduleFree
    else:
        raise ValueError("Invalid schedulefree optimizer")

    additional_optim_kwargs["weight_decay"] = args.weight_decay
    if require_warmup:
        additional_optim_kwargs["warmup_steps"] = args.warmup_steps
    additional_optim_kwargs.update(
        {
            "weight_lr_power": float(optim_args.get("weight_lr_power", 2.0)),
            "r": float(optim_args.get("r", 0.0)),
        }
    )
    optimizer_kwargs.update(additional_optim_kwargs)
    return optimizer_cls, optimizer_kwargs


def _get_stable_adamw(
    args: TrainingArguments,
    optimizer_kwargs: dict[str, Any],
    adam_kwargs: dict[str, Any],
    optim_args: dict[str, str],
) -> tuple[Any, dict[str, Any]]:
    """Get StableAdamW optimizer from torch-optimi."""
    if not is_torch_optimi_available():
        raise ImportError(
            "You need to install `torch-optimi` in order to use stable_adamw optimizers. "
            "Install it with `pip install torch-optimi`."
        )
    from optimi import StableAdamW

    max_lr = optim_args.pop("max_lr", None)
    if max_lr is not None:
        max_lr = float(max_lr)

    kahan_sum = optim_args.pop("kahan_sum", None)
    if kahan_sum is not None:
        kahan_sum = bool(kahan_sum)

    adam_kwargs["weight_decay"] = args.weight_decay
    stable_adamw_kwargs = {
        "decouple_lr": bool(optim_args.pop("decouple_lr", False)),
        "max_lr": max_lr,
        "kahan_sum": kahan_sum,
    }

    optimizer_kwargs.update(adam_kwargs)
    optimizer_kwargs.update(stable_adamw_kwargs)
    return StableAdamW, optimizer_kwargs


# =============================================================================
# Main entry point
# =============================================================================


def get_optimizer_cls_and_kwargs(
    args: TrainingArguments,
    model: PreTrainedModel | None = None,
) -> tuple[Any, dict[str, Any]]:
    """
    Returns the optimizer class and optimizer parameters based on the training arguments.

    Args:
        args (`TrainingArguments`):
            The training arguments for the training session.
        model (`PreTrainedModel`, *optional*):
            The model being trained. Required for some optimizers (GaLore, Apollo, LOMO).

    Returns:
        A tuple containing the optimizer class and a dictionary of optimizer keyword arguments.
    """
    # Parse optimizer arguments
    optim_args = _parse_optim_args(args.optim_args)

    # Common kwargs
    optimizer_kwargs = {"lr": args.learning_rate}
    adam_kwargs = {
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
    }

    # Dispatch to appropriate handler
    if args.optim == OptimizerNames.ADAFACTOR:
        return _get_adafactor(optimizer_kwargs)

    if args.optim in (OptimizerNames.ADAMW_TORCH, OptimizerNames.ADAMW_TORCH_FUSED):
        return _get_adamw_torch(args.optim, optimizer_kwargs, adam_kwargs)

    if args.optim == OptimizerNames.ADAMW_TORCH_XLA:
        return _get_adamw_torch_xla(optimizer_kwargs, adam_kwargs)

    if args.optim == OptimizerNames.ADAMW_TORCH_NPU_FUSED:
        return _get_adamw_torch_npu_fused(optimizer_kwargs, adam_kwargs)

    if args.optim == OptimizerNames.ADAMW_APEX_FUSED:
        return _get_adamw_apex_fused(optimizer_kwargs, adam_kwargs)

    if args.optim in _BITSANDBYTES_OPTIMIZERS:
        return _get_bitsandbytes_optimizer(args, optimizer_kwargs, adam_kwargs, optim_args)

    if args.optim == OptimizerNames.ADAMW_ANYPRECISION:
        return _get_adamw_anyprecision(optimizer_kwargs, adam_kwargs, optim_args)

    if args.optim == OptimizerNames.SGD:
        return _get_sgd(optimizer_kwargs)

    if args.optim == OptimizerNames.ADAGRAD:
        return _get_adagrad(optimizer_kwargs)

    if args.optim == OptimizerNames.RMSPROP:
        return _get_rmsprop(optimizer_kwargs)

    if args.optim in _GALORE_OPTIMIZERS:
        return _get_galore_optimizer(args, model, optimizer_kwargs, optim_args)

    if args.optim in _APOLLO_OPTIMIZERS:
        return _get_apollo_optimizer(args, model, optimizer_kwargs, adam_kwargs, optim_args)

    if args.optim in (OptimizerNames.LOMO, OptimizerNames.ADALOMO):
        return _get_lomo_optimizer(args, model, optimizer_kwargs)

    if args.optim == OptimizerNames.GROKADAMW:
        return _get_grokadamw(optimizer_kwargs, optim_args)

    if args.optim in _TORCHAO_OPTIMIZERS:
        return _get_torchao_optimizer(args, optimizer_kwargs, adam_kwargs, optim_args)

    if args.optim in _SCHEDULE_FREE_OPTIMIZERS:
        return _get_schedule_free_optimizer(args, optimizer_kwargs, adam_kwargs, optim_args)

    if args.optim == OptimizerNames.STABLE_ADAMW:
        return _get_stable_adamw(args, optimizer_kwargs, adam_kwargs, optim_args)

    raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
