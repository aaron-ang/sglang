from __future__ import annotations

import gc
import logging
from typing import Callable, Optional

import torch

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.platforms import current_platform
from sglang.srt.utils import get_available_gpu_memory, init_custom_process_group
from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)


def init_weights_update_group(
    *,
    _model_update_group,
    tp_rank,
    master_address,
    master_port,
    rank_offset,
    world_size,
    group_name,
    backend="nccl",
):
    """Initialize the Torch process group for model parameter updates.

    `_model_update_group` is used in the RLHF workflow, where rank
    0 is the actor model in the training engine, and the other ranks are
    the inference engine, which is used for rollout.

    In the RLHF workflow, the training engine updates the model
    weights/parameters online, and broadcasts them to the inference
    engine through the `_model_update_group` process group.
    """
    assert (
        torch.distributed.is_initialized()
    ), "Default torch process group must be initialized"
    assert group_name != "", "Group name cannot be empty"

    rank = rank_offset + tp_rank

    logger.info(
        f"init custom process group: master_address={master_address}, master_port={master_port}, "
        f"rank_offset={rank_offset}, rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
    )

    try:
        na = NetworkAddress(master_address, master_port)
        _model_update_group[group_name] = init_custom_process_group(
            backend=backend,
            init_method=na.to_tcp(),
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        return True, "Succeeded to initialize custom process group."
    except Exception as e:
        message = f"Failed to initialize custom process group: {e}."
        logger.error(message)
        return False, message


def destroy_weights_update_group(*, _model_update_group, group_name):
    try:
        if group_name in _model_update_group:
            pg = _model_update_group.pop(group_name)
            torch.distributed.destroy_process_group(pg)
            return True, "Succeeded to destroy custom process group."
        else:
            return False, "The group to be destroyed does not exist."
    except Exception as e:
        message = f"Failed to destroy custom process group: {e}."
        logger.error(message)
        return False, message


def update_weights_from_disk(
    *,
    model_runner_ref,
    model_path: str,
    load_format: str,
    weight_name_filter: Optional[Callable[[str], bool]] = None,
    recapture_cuda_graph: bool = False,
) -> tuple[bool, str]:
    """Update engine weights in-place from the disk."""
    logger.info(
        f"Update engine weights online from disk begin. "
        f"avail mem={get_available_gpu_memory(model_runner_ref.device, model_runner_ref.gpu_id, empty_cache=False):.2f} GB"
    )

    target_device = torch.device(model_runner_ref.device)
    model_runner_ref.model_config.model_path = model_path
    load_config = LoadConfig(load_format=load_format)

    # Only support DefaultModelLoader for now
    loader = get_model_loader(load_config, model_runner_ref.model_config)
    if not isinstance(loader, DefaultModelLoader):
        message = f"Failed to get model loader: {loader}."
        return False, message

    def get_weight_iter(config):
        iter = loader._get_weights_iterator(
            DefaultModelLoader.Source.init_new(config, model_runner_ref.model)
        )
        if weight_name_filter is not None:
            iter = ((name, weight) for name, weight in iter if weight_name_filter(name))

        return iter

    def model_load_weights(model, iter):
        loader.load_weights_and_postprocess(model, iter, target_device)
        return model

    with set_default_torch_dtype(model_runner_ref.model_config.dtype):
        try:
            iter = get_weight_iter(model_runner_ref.model_config)
        except Exception as e:
            message = f"Failed to get weights iterator: {e}."
            return False, message
        try:
            model = model_load_weights(model_runner_ref.model, iter)
        except Exception as e:
            message = (
                f"Failed to update weights: {e}.\nRolling back to original weights."
            )
            del iter
            gc.collect()
            iter = get_weight_iter(model_runner_ref.model_config)
            model_runner_ref.model = model_load_weights(model_runner_ref.model, iter)
            return False, message

    model_runner_ref.model = model
    model_runner_ref.server_args.model_path = model_path
    model_runner_ref.server_args.load_format = load_format
    model_runner_ref.load_config = load_config

    if recapture_cuda_graph and (
        model_runner_ref.device == "cuda"
        or model_runner_ref.device == "musa"
        or (current_platform.is_out_of_tree() and current_platform.support_cuda_graph())
    ):
        model_runner_ref.init_device_graphs()

    logger.info("Update weights end.")
    return True, "Succeeded to update model weights."


from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket


def update_weights_from_distributed(
    *,
    model,
    _model_update_group,
    device,
    names,
    dtypes,
    shapes,
    group_name,
    load_format: Optional[str] = None,
):
    """
    Update specific parameter in the model weights online
    through `_model_update_group` process group.

    Args:
        name: the name of the parameter to be updated.
        dtype: the data type of the parameter to be updated.
        shape: the shape of the parameter to be updated.
    """

    assert group_name in _model_update_group, (
        f"Group {group_name} not in {list(_model_update_group.keys())}. "
        "Please call `init_weights_update_group` first."
    )

    if load_format == "flattened_bucket":
        return _update_bucketed_weights_from_distributed(
            model=model,
            _model_update_group=_model_update_group,
            device=device,
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
        )
    try:
        weights = []
        handles = []
        for name, dtype, shape in zip(names, dtypes, shapes):
            target_dtype = (
                dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
            )
            weight = torch.empty(shape, dtype=target_dtype, device=device)
            handles.append(
                torch.distributed.broadcast(
                    weight,
                    src=0,
                    group=_model_update_group[group_name],
                    async_op=True,
                )
            )
            weights.append((name, weight))
        for handle in handles:
            handle.wait()

        model.load_weights(weights)
        return True, "Succeeded to update parameter online."

    except Exception as e:
        error_msg = (
            f"Failed to update parameter online: {e}. "
            f"The full weights of the ModelRunner are partially updated. "
            f"Please discard the whole weights."
        )
        logger.error(error_msg)
        return False, error_msg


def _update_bucketed_weights_from_distributed(
    *, model, _model_update_group, device, names, dtypes, shapes, group_name
):
    try:
        named_tensors = []
        for name, dtype, shape in zip(names, dtypes, shapes):
            target_dtype = (
                dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
            )
            named_tensors.append(
                (name, torch.empty(shape, dtype=target_dtype, device=device))
            )
        bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        flattened_tensor = bucket.get_flattened_tensor()
        torch.distributed.broadcast(
            flattened_tensor,
            src=0,
            group=_model_update_group[group_name],
        )
        reconstructed_tensors = bucket.reconstruct_tensors()
        model.load_weights(reconstructed_tensors)
        return True, f"Succeeded to update parameter online."
    except Exception as e:
        error_msg = (
            f"Failed to update parameter online: {e}. "
            f"The full weights of the ModelRunner are partially updated. "
            f"Please discard the whole weights."
        )
        logger.error(error_msg)
        return False, error_msg


from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import MultiprocessingSerializer, dynamic_import
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)


def _model_load_weights_direct(model, named_tensors: List[Tuple[str, torch.Tensor]]):
    params_dict = dict(model.named_parameters())
    for name, tensor in named_tensors:
        default_weight_loader(params_dict[name], tensor)


def _unwrap_tensor(tensor, tp_rank, device):
    if isinstance(tensor, LocalSerializedTensor):
        tensor = tensor.get(tp_rank)
    return tensor.to(device)


@dataclass
class LocalSerializedTensor:
    """torch.Tensor that gets serialized by MultiprocessingSerializer (which only serializes a pointer and not the data).
    The i-th element in the list corresponds to i-th rank's GPU."""

    values: List[bytes]

    def get(self, rank: int):
        return MultiprocessingSerializer.deserialize(self.values[rank])


def _update_weights_from_flattened_bucket(
    *,
    model,
    flattened_tensor_bucket_dict,
):
    """Handle flattened bucket format for weight updates"""
    flattened_tensor = flattened_tensor_bucket_dict["flattened_tensor"]
    metadata = flattened_tensor_bucket_dict["metadata"]

    # Convert metadata dict to our format
    converted_metadata = []
    for meta in metadata:
        converted_meta = FlattenedTensorMetadata(
            name=meta.name,
            shape=meta.shape,
            dtype=meta.dtype,
            start_idx=meta.start_idx,
            end_idx=meta.end_idx,
            numel=meta.numel,
        )
        converted_metadata.append(converted_meta)

    # Create bucket and reconstruct tensors
    bucket = FlattenedTensorBucket(
        flattened_tensor=flattened_tensor, metadata=converted_metadata
    )
    reconstructed_tensors = bucket.reconstruct_tensors()

    # Load the reconstructed tensors using the standard method
    model.load_weights(reconstructed_tensors)

    return True, "Success"


def update_weights_from_tensor(
    *,
    model,
    tp_rank,
    device,
    custom_weight_loader,
    named_tensors: List[Tuple[str, Union[torch.Tensor, "LocalSerializedTensor"]]],
    load_format: Optional[str] = None,
):
    monkey_patch_torch_reductions()
    if load_format == "flattened_bucket":
        # Handle flattened bucket format
        return _update_weights_from_flattened_bucket(
            model=model, flattened_tensor_bucket_dict=named_tensors
        )

    # We need to get device after patch otherwise the device would be wrong
    device_module = torch.get_device_module(device)
    infered_device = device_module.current_device()

    named_tensors = [
        (name, _unwrap_tensor(tensor, tp_rank=tp_rank, device=infered_device))
        for name, tensor in named_tensors
    ]
    if load_format == "direct":
        _model_load_weights_direct(model, named_tensors)
    elif load_format in custom_weight_loader:
        custom_loader = dynamic_import(load_format)
        custom_loader(model, named_tensors)
    elif load_format is None:
        model.load_weights(named_tensors)
    else:
        raise NotImplementedError(f"Unknown load_format={load_format}")
    return True, "Success"
