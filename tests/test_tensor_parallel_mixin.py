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

    Weight Loading Paths & Test Coverage
    ====================================

    There are two distinct loading paths through the dynamic weight loading system:

    PATH A: Direct Load (Dense models like Llama, Mistral)
    -------------------------------------------------------
    Checkpoint format == Model format (no conversion needed)

        Checkpoint File
             │
             ▼
        from_pretrained(tp_plan="auto")
             │
             ▼
        ┌─────────────────────────────┐
        │  For each weight:           │
        │  1. Match key (direct)      │
        │  2. [no conversion needed]  │
        │  3. Apply TP sharding       │
        │  4. Set parameter           │
        └─────────────────────────────┘
             │
             ▼
        TP-Sharded Model

    Tests: test_tp_forward_direct, test_tp_backward_direct, test_tp_generation_direct


    PATH B: Conversion + Load (MoE models like Mixtral, Qwen2-MoE)
    --------------------------------------------------------------
    but EXCEPTION for GPT_OSS as weight by default 3D 

    Checkpoint format != Model format (conversion mapping required)

        Original Checkpoint (unfused experts)
             │
             ▼
        from_pretrained(tp_plan="auto")
             │
             ▼
        ┌─────────────────────────────┐
        │  For each weight:           │
        │  1. Match key pattern       │
        │  2. Apply conversion ops    │  ← MergeModulelist + Concatenate
        │  3. Apply TP sharding       │
        │  4. Set parameter           │
        └─────────────────────────────┘
             │
             ▼
        TP-Sharded Model (fused experts)

    Tests: test_tp_generation_with_conversion, test_tp_conversion_integration
"""

import os
import tempfile
from abc import ABC, abstractmethod

from transformers import set_seed
from transformers.conversion_mapping import _MODEL_TO_CONVERSION_PATTERN
from transformers.testing_utils import (
    backend_device_count,
    get_torch_dist_unique_port,
    is_torch_available,
    torch_device,
)
from transformers.utils import is_torch_greater_or_equal


if is_torch_available():
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp


def _debug_log(rank, msg):
    """Print debug message only from rank 0."""
    if rank == 0:
        print(f"[TP Test Debug] {msg}")


def get_packed_grad_shard(grad, world_size, rank, dim):
    """Get the correct shard of a packed gradient (matching get_packed_weights interleaved logic).

    Packed weights like gate_up_proj are sharded with interleaving:
    Original: [G0 G1 G2 G3 | U0 U1 U2 U3]  (gate | up)
    Rank 0:   [G0 G1 | U0 U1]
    Rank 1:   [G2 G3 | U2 U3]
    """
    total_size = grad.shape[dim]
    # Packed weights have 2 blocks (gate and up)
    block_size = total_size // 2
    shard_block_size = block_size // world_size

    # Build interleaved indices
    indices = []
    for block_idx in range(2):  # gate block, then up block
        block_offset = block_idx * block_size
        start = block_offset + rank * shard_block_size
        stop = block_offset + (rank + 1) * shard_block_size
        indices.extend(range(start, stop))

    # Select along the sharded dimension
    return grad.index_select(dim, torch.tensor(indices, device=grad.device))


def _create_original_format_checkpoint(model, tmp_dir, model_type):
    """Create a checkpoint in original (unfused) format to trigger conversion mapping.

    This unfuses the internal gate_up_proj format back to the original checkpoint format
    that the model was originally trained with (e.g., separate w1/w3/w2 for Mixtral,
    or separate gate_proj/up_proj/down_proj for Qwen2Moe).
    """
    from safetensors.torch import save_file

    # Determine conversion pattern
    conversion_pattern = _MODEL_TO_CONVERSION_PATTERN.get(model_type)

    state_dict = model.state_dict()
    original_state_dict = {}

    for key, tensor in state_dict.items():
        if conversion_pattern == "mixtral":
            # Mixtral-style: .mlp.experts.X → .block_sparse_moe.experts.N.wX
            if ".mlp.experts.gate_up_proj" in key:
                # Unfuse gate_up_proj: [num_experts, 2*intermediate, hidden]
                num_experts = tensor.shape[0]
                intermediate_dim = tensor.shape[1] // 2
                gate, up = tensor.split(intermediate_dim, dim=1)

                base_key = key.replace(".mlp.experts.gate_up_proj", ".block_sparse_moe")
                for expert_idx in range(num_experts):
                    original_state_dict[f"{base_key}.experts.{expert_idx}.w1.weight"] = gate[expert_idx]
                    original_state_dict[f"{base_key}.experts.{expert_idx}.w3.weight"] = up[expert_idx]
            elif ".mlp.experts.down_proj" in key:
                # Unfuse down_proj: [num_experts, hidden, intermediate]
                num_experts = tensor.shape[0]
                base_key = key.replace(".mlp.experts.down_proj", ".block_sparse_moe")
                for expert_idx in range(num_experts):
                    original_state_dict[f"{base_key}.experts.{expert_idx}.w2.weight"] = tensor[expert_idx]
            elif ".mlp.router" in key:
                # Rename router: .mlp.router → .block_sparse_moe.gate
                new_key = key.replace(".mlp.router", ".block_sparse_moe.gate")
                original_state_dict[new_key] = tensor
            else:
                original_state_dict[key] = tensor

        elif conversion_pattern == "qwen2_moe":
            # Qwen2-style: .mlp.experts.X → .mlp.experts.N.X_proj
            if ".mlp.experts.gate_up_proj" in key:
                # Unfuse gate_up_proj: [num_experts, 2*intermediate, hidden]
                num_experts = tensor.shape[0]
                intermediate_dim = tensor.shape[1] // 2
                gate, up = tensor.split(intermediate_dim, dim=1)

                base_key = key.replace(".mlp.experts.gate_up_proj", ".mlp.experts")
                for expert_idx in range(num_experts):
                    original_state_dict[f"{base_key}.{expert_idx}.gate_proj.weight"] = gate[expert_idx]
                    original_state_dict[f"{base_key}.{expert_idx}.up_proj.weight"] = up[expert_idx]
            elif ".mlp.experts.down_proj" in key:
                # Unfuse down_proj: [num_experts, hidden, intermediate]
                num_experts = tensor.shape[0]
                base_key = key.replace(".mlp.experts.down_proj", ".mlp.experts")
                for expert_idx in range(num_experts):
                    original_state_dict[f"{base_key}.{expert_idx}.down_proj.weight"] = tensor[expert_idx]
            else:
                original_state_dict[key] = tensor
        else:
            # No conversion pattern - keep as-is
            original_state_dict[key] = tensor

    # Save checkpoint in safetensors format
    save_file(original_state_dict, os.path.join(tmp_dir, "model.safetensors"))

    # Save config
    model.config.save_pretrained(tmp_dir)


def _global_wrapper(rank, func, tp, port, func_args, func_kwargs):
    """Wrapper to set up distributed environment and run the test function."""

    def setup_dist_env(rank, world_size, port):
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)

    world_size = tp
    setup_dist_env(rank, world_size, port)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    func(rank, *func_args, **func_kwargs)

    dist.barrier()
    dist.destroy_process_group()


def _init_distributed(tp: int):
    """Decorator to initialize distributed environment and spawn processes."""

    def _init_distributed_inner(func):
        def wrapper(*args, **kwargs):
            world_size = tp
            port = get_torch_dist_unique_port()
            spawn_args = (func, tp, port, args, kwargs)
            mp.spawn(_global_wrapper, args=spawn_args, nprocs=world_size)

        return wrapper

    return _init_distributed_inner


class TensorParallelTesterMixin(ABC):
    """
    Mixin for tensor parallel tests. Add to model test classes alongside ModelTesterMixin.

    The model_tester (e.g., CausalLMModelTester) already provides:
      - get_config() -> tiny model config
      - causal_lm_class, base_model_class, etc.

    This mixin adds tensor parallel-specific tests using that infrastructure.
    """

    # ============================================================
    # Configuration (can be overridden per model)
    # ============================================================
    tensor_parallel_size: int = 2
    tensor_parallel_atol: float = 1e-5
    tensor_parallel_rtol: float = 1e-5

    @property
    @abstractmethod
    def model_tester(self):
        """The model tester instance (e.g., CausalLMModelTester)."""
        ...

    # ============================================================
    # Helper methods
    # ============================================================
    def _has_tp_plan(self) -> bool:
        """Check if model has a tensor parallel plan defined."""
        config = self.model_tester.get_config()
        return hasattr(config, "base_model_tp_plan") and config.base_model_tp_plan is not None

    def _load_tp_and_reference_models(self, model_path, model_class):
        """Load TP model and non-TP reference model for comparison.

        Returns:
            tuple: (model_tp, model_ref, device)
        """
        model_tp = model_class.from_pretrained(model_path, tp_plan="auto")
        dist.barrier()

        device = model_tp.device
        model_ref = model_class.from_pretrained(model_path)
        model_ref = model_ref.to(device)

        return model_tp, model_ref, device

    def _verify_tp_sharding(self, rank, model_tp, model_ref):
        """Verify TP sharding by comparing parameter shapes between TP and reference models.

        Returns:
            list: Names of sharded parameters
        """
        world_size = dist.get_world_size()
        sharded_params = []

        for (name, param), (_, param_full) in zip(model_tp.named_parameters(), model_ref.named_parameters()):
            if param.shape != param_full.shape:
                sharded_params.append(name)
                _debug_log(rank, f"TP sharded: {name} - full: {param_full.shape} -> sharded: {param.shape}")

                # Verify sharding is correct
                for dim in range(param.ndim):
                    if param.size(dim) != param_full.size(dim):
                        if "gate_up_proj" in name:
                            expected_size = param_full.size(dim) // world_size
                            assert param.size(dim) == expected_size, (
                                f"Packed weight {name} sharding incorrect: expected {expected_size}, got {param.size(dim)}"
                            )
                        else:
                            expected_size = (param_full.size(dim) + world_size - 1) // world_size
                            assert param.size(dim) <= expected_size, (
                                f"Weight {name} sharding incorrect: expected <= {expected_size}, got {param.size(dim)}"
                            )
                        break

        return sharded_params

    def _get_tp_model_class(self):
        """Get the model class to use for TP tests (prefers *ForCausalLM)."""
        # Prefer model classes with a head (for computing loss)
        if hasattr(self.model_tester, "causal_lm_class") and self.model_tester.causal_lm_class is not None:
            return self.model_tester.causal_lm_class
        # Fall back to first model class
        return self.all_model_classes[0]

    def _skip_if_not_supported(self):
        """Check and skip test if TP is not supported for this model/environment."""
        # Check PyTorch version
        if not is_torch_greater_or_equal("2.9"):
            self.skipTest("Tensor parallel tests require torch >= 2.9")

        # Check if model has TP plan
        if not self._has_tp_plan():
            self.skipTest("Model does not have a tensor parallel plan (base_model_tp_plan)")

        # Check device availability
        if backend_device_count(torch_device) < self.tensor_parallel_size:
            self.skipTest(
                f"Need at least {self.tensor_parallel_size} devices, "
                f"have {backend_device_count(torch_device)}"
            )

    # ============================================================
    # Test implementations (run inside distributed processes)
    # ============================================================
    def _test_tp_forward_impl(self, _rank, model_path, model_class, atol, rtol):
        """Implementation for comparing TP and non-TP model outputs."""
        set_seed(0)

        model_tp, model, device = self._load_tp_and_reference_models(model_path, model_class)
        model_tp.eval()
        model.eval()

        set_seed(42)
        input_ids = torch.randint(0, model.config.vocab_size, (2, 64)).to(device)

        with torch.no_grad():
            logits = model(input_ids).logits
            logits_tp = model_tp(input_ids).logits

        diff = (logits - logits_tp).abs()
        assert torch.allclose(logits, logits_tp, atol=atol, rtol=rtol), (
            f"TP and non-TP model outputs differ. "
            f"Max diff: {diff.max().item()} | Min diff: {diff.min().item()}"
        )

        dist.barrier()

    def _test_tp_backward_impl(self, rank, model_path, model_class, atol, rtol):
        """Implementation for comparing TP and non-TP model backward passes."""
        set_seed(0)

        model_tp, model, device = self._load_tp_and_reference_models(model_path, model_class)
        model_tp.train()
        model.train()

        vocab_size = model.config.vocab_size
        set_seed(42)
        input_ids = torch.randint(0, vocab_size, (2, 64)).to(device)
        set_seed(43)
        labels = torch.randint(0, vocab_size, (2, 64)).to(device)

        # Forward and backward for both models
        loss = model(input_ids, labels=labels).loss
        loss.backward()

        loss_tp = model_tp(input_ids, labels=labels).loss
        loss_tp.backward()

        # Compare losses
        assert torch.allclose(loss, loss_tp, atol=atol, rtol=rtol), (
            f"TP and non-TP model losses differ. "
            f"Non-TP loss: {loss.item()}, TP loss: {loss_tp.item()}, "
            f"Diff: {(loss - loss_tp).abs().item()}"
        )

        # Compare gradients for matching parameters
        world_size = dist.get_world_size()
        for (name, param), (_, param_tp) in zip(model.named_parameters(), model_tp.named_parameters()):
            if param.grad is not None and param_tp.grad is not None:
                grad = param.grad
                grad_tp = param_tp.grad

                # Slice reference gradient to match local shard if parameter is sharded
                if grad.shape != grad_tp.shape:
                    for dim in range(grad.ndim):
                        if grad.size(dim) != grad_tp.size(dim):
                            if "gate_up_proj" in name:
                                grad = get_packed_grad_shard(grad, world_size, rank, dim)
                            else:
                                shard_size = grad_tp.size(dim)
                                start = rank * shard_size
                                grad = grad.narrow(dim, start, shard_size)
                            break

                assert torch.allclose(grad.cpu(), grad_tp.cpu(), atol=atol, rtol=rtol), (
                    f"Gradients differ for parameter {name}. "
                    f"Max diff: {(grad.cpu() - grad_tp.cpu()).abs().max().item()}"
                )

        dist.barrier()

    def _test_tp_generation_impl(self, _rank, model_path, model_class, atol, rtol, max_new_tokens):
        """Implementation for comparing TP and non-TP model generation outputs (direct load path)."""
        set_seed(0)

        model_tp, model, device = self._load_tp_and_reference_models(model_path, model_class)
        model_tp.eval()
        model.eval()

        set_seed(42)
        input_ids = torch.randint(0, model.config.vocab_size, (1, 10)).to(device)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "output_scores": True,
            "return_dict_in_generate": True,
            "use_cache": True,
        }

        with torch.no_grad():
            output = model.generate(input_ids, **generation_kwargs)
            output_tp = model_tp.generate(input_ids, **generation_kwargs)

        # Compare logits/scores at each generation step
        scores = torch.stack(output.scores)
        scores_tp = torch.stack(output_tp.scores)

        diff = (scores - scores_tp).abs()
        assert torch.allclose(scores, scores_tp, atol=atol, rtol=rtol), (
            f"TP and non-TP model generation logits differ (direct load path). "
            f"Max diff: {diff.max().item()} | Mean diff: {diff.mean().item()}"
        )

        _debug_log(_rank, "Generation with direct load path PASSED")
        dist.barrier()

    def _test_tp_generation_with_conversion_impl(self, _rank, model_path, model_class, atol, rtol, max_new_tokens):
        """Implementation for testing TP generation with conversion mapping."""
        set_seed(0)

        model_tp, model, device = self._load_tp_and_reference_models(model_path, model_class)
        model_tp.eval()
        model.eval()

        # Verify conversion mapping was applied
        assert hasattr(model_tp, "_weight_conversions"), "Conversion mapping was not applied during load"
        assert model_tp._weight_conversions is not None, "Conversion mapping is None"
        _debug_log(_rank, f"Conversion mapping applied: {len(model_tp._weight_conversions)} conversions")

        # Verify TP sharding by comparing parameter shapes
        self._verify_tp_sharding(_rank, model_tp, model)

        # Test generation
        set_seed(42)
        input_ids = torch.randint(0, model.config.vocab_size, (1, 10)).to(device)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
            "output_scores": True,
            "return_dict_in_generate": True,
            "use_cache": True,
        }

        with torch.no_grad():
            output = model.generate(input_ids, **generation_kwargs)
            output_tp = model_tp.generate(input_ids, **generation_kwargs)

        scores = torch.stack(output.scores)
        scores_tp = torch.stack(output_tp.scores)

        diff = (scores - scores_tp).abs()
        assert torch.allclose(scores, scores_tp, atol=atol, rtol=rtol), (
            f"TP and non-TP model generation logits differ (with conversion mapping). "
            f"Max diff: {diff.max().item()} | Mean diff: {diff.mean().item()}"
        )

        _debug_log(_rank, "Generation with conversion mapping PASSED")
        dist.barrier()

    def _test_tp_conversion_integration_impl(self, rank, model_path, model_class):
        """Verify that conversion mapping + TP sharding both execute during load."""
        model_tp, model, device = self._load_tp_and_reference_models(model_path, model_class)

        # Verification 1: Conversion mapping was applied
        assert hasattr(model_tp, "_weight_conversions"), "Conversion mapping not applied"
        assert model_tp._weight_conversions is not None, "Conversion mapping is None"

        from transformers.core_model_loading import WeightConverter

        converters = [c for c in model_tp._weight_conversions if isinstance(c, WeightConverter)]
        assert len(converters) > 0, "No WeightConverter operations were applied"
        _debug_log(rank, f"Applied {len(converters)} WeightConverter operations")
        if rank == 0:
            for c in converters:
                print(f"  - {c.source_patterns} -> {c.target_patterns}")

        # Verification 2: TP sharding occurred
        sharded_params = self._verify_tp_sharding(rank, model_tp, model)
        assert len(sharded_params) > 0, "No parameters were sharded by TP"
        _debug_log(rank, f"{len(sharded_params)} parameters sharded:")
        if rank == 0:
            for name in sharded_params[:5]:
                print(f"  - {name}")

        # Verification 3: Forward pass works
        set_seed(42)
        input_ids = torch.randint(0, model_tp.config.vocab_size, (2, 32)).to(device)

        with torch.no_grad():
            output_tp = model_tp(input_ids)
            output_full = model(input_ids)

        assert torch.allclose(output_tp.logits, output_full.logits, atol=1e-4, rtol=1e-4), (
            "TP and non-TP outputs differ after conversion+sharding"
        )

        _debug_log(rank, "Forward pass verification PASSED")
        dist.barrier()

    # ============================================================
    # Public test methods - PATH A: Direct Load (Dense models)
    # ============================================================
    def test_tp_forward_direct(self):
        """Test TP forward pass with direct load path (no conversion mapping).

        Loading path: checkpoint → TP sharding → model
        Applies to: Dense models (Llama, Mistral, etc.) where checkpoint format == model format
        """
        self._skip_if_not_supported()

        config = self.model_tester.get_config()
        model_class = self._get_tp_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol

        # Save model to temp directory so we can load it with from_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save a model with the test config
            model = model_class(config)
            model.save_pretrained(tmp_dir)

            _init_distributed(tp=self.tensor_parallel_size)(self._test_tp_forward_impl)(
                tmp_dir, model_class, atol, rtol
            )

    def test_tp_backward_direct(self):
        """Test TP backward pass with direct load path (no conversion mapping).

        Loading path: checkpoint → TP sharding → model
        Applies to: Dense models (Llama, Mistral, etc.) where checkpoint format == model format
        """
        self._skip_if_not_supported()

        config = self.model_tester.get_config()
        model_class = self._get_tp_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol

        # Save model to temp directory so we can load it with from_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create and save a model with the test config
            model = model_class(config)
            model.save_pretrained(tmp_dir)

            _init_distributed(tp=self.tensor_parallel_size)(self._test_tp_backward_impl)(
                tmp_dir, model_class, atol, rtol
            )

    def test_tp_generation_direct(self):
        """Test TP generation with direct load path (no conversion mapping).

        Loading path: checkpoint → TP sharding → model → generate
        Applies to: Dense models (Llama, Mistral, etc.) where checkpoint format == model format
        """
        self._skip_if_not_supported()

        config = self.model_tester.get_config()
        model_class = self._get_tp_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol
        max_new_tokens = 10

        with tempfile.TemporaryDirectory() as tmp_dir:
            model = model_class(config)
            model.save_pretrained(tmp_dir)

            _init_distributed(tp=self.tensor_parallel_size)(self._test_tp_generation_impl)(
                tmp_dir, model_class, atol, rtol, max_new_tokens
            )

    # ============================================================
    # Public test methods - PATH B: Conversion + Load (MoE models)
    # ============================================================
    def test_tp_generation_with_conversion(self):
        """Test TP generation with conversion mapping path (MoE weight fusion).

        Loading path: original checkpoint → conversion mapping → TP sharding → model → generate
        Applies to: MoE models (Mixtral, Qwen2-MoE, etc.) where checkpoint has unfused experts

        This test creates a checkpoint in the original format (e.g., separate expert weights
        like w1/w3/w2 for Mixtral) and verifies that loading with tp_plan="auto" correctly
        applies the conversion mapping to fuse weights during tensor parallel loading.
        """
        self._skip_if_not_supported()

        # Only run for models with conversion mapping
        config = self.model_tester.get_config()
        model_type = getattr(config, "model_type", None)
        if model_type not in _MODEL_TO_CONVERSION_PATTERN:
            self.skipTest(f"Model type {model_type} has no conversion mapping defined")

        model_class = self._get_tp_model_class()
        atol = self.tensor_parallel_atol
        rtol = self.tensor_parallel_rtol
        max_new_tokens = 10

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create model and save in original (unfused) format
            model = model_class(config)
            _create_original_format_checkpoint(model, tmp_dir, model_type)

            _init_distributed(tp=self.tensor_parallel_size)(self._test_tp_generation_with_conversion_impl)(
                tmp_dir, model_class, atol, rtol, max_new_tokens
            )

    def test_tp_conversion_integration(self):
        """Test that conversion mapping + TP sharding integrate correctly during load.

        Loading path: original checkpoint → conversion mapping → TP sharding → model
        Applies to: MoE models (Mixtral, Qwen2-MoE, etc.) where checkpoint has unfused experts

        This test verifies that:
        1. WeightConverter operations are applied (conversion mapping)
        2. Parameters are sharded correctly (TP sharding)
        3. Forward pass produces correct outputs
        """
        self._skip_if_not_supported()

        config = self.model_tester.get_config()
        model_type = getattr(config, "model_type", None)
        if model_type not in _MODEL_TO_CONVERSION_PATTERN:
            self.skipTest(f"Model type {model_type} has no conversion mapping")

        model_class = self._get_tp_model_class()

        with tempfile.TemporaryDirectory() as tmp_dir:
            model = model_class(config)
            _create_original_format_checkpoint(model, tmp_dir, model_type)

            _init_distributed(tp=self.tensor_parallel_size)(self._test_tp_conversion_integration_impl)(
                tmp_dir, model_class
            )
