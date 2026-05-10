from __future__ import annotations

import logging

import torch
import torch.distributed as dist

from sglang.srt.utils import init_custom_process_group
from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)


def init_weights_send_group_for_remote_instance(
    *,
    _weights_send_group,
    tp_rank,
    tp_size,
    gpu_id,
    master_address,
    ports,
    group_rank,
    world_size,
    group_name,
    backend="nccl",
):
    assert (
        torch.distributed.is_initialized()
    ), "Default torch process group must be initialized"
    assert group_name != "", "Group name cannot be empty"

    ports_list = ports.split(",")
    assert (
        len(ports_list) == tp_size
    ), f"Expected {tp_size} ports, but got {len(ports_list)} ports."
    group_port = ports_list[tp_rank]
    group_name = f"{group_name}_{group_port}_{tp_rank}"

    logger.info(
        f"init custom process group: tp_rank={tp_rank}, gpu_id={gpu_id}, master_address={master_address}, master_port={group_port}, "
        f"group_rank={group_rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
    )

    torch.cuda.empty_cache()
    success = False
    message = ""
    try:
        na = NetworkAddress(master_address, group_port)
        _weights_send_group[group_name] = init_custom_process_group(
            backend=backend,
            init_method=na.to_tcp(),
            world_size=world_size,
            rank=group_rank,
            group_name=group_name,
            device_id=torch.device("cuda", gpu_id),
        )
        dist.barrier(group=_weights_send_group[group_name])
        success = True
        message = f"Succeeded to init group through {na.to_host_port_str()} group."
    except Exception as e:
        message = f"Failed to init group: {e}."
        logger.error(message)

    torch.cuda.empty_cache()
    return success, message


def send_weights_to_remote_instance(
    *,
    model,
    _weights_send_group,
    tp_rank,
    tp_size,
    master_address,
    ports,
    group_name,
):
    assert (
        torch.distributed.is_initialized()
    ), "Default torch process group must be initialized"
    assert group_name != "", "Group name cannot be empty"

    ports_list = ports.split(",")
    assert (
        len(ports_list) == tp_size
    ), f"Expected {tp_size} ports, but got {len(ports_list)} ports."
    group_port = ports_list[tp_rank]
    group_name = f"{group_name}_{group_port}_{tp_rank}"

    if _weights_send_group[group_name] is not None:
        send_group = _weights_send_group[group_name]
    else:
        message = f"Group {group_name} not in _weights_send_group list. Please call `init_weights_send_group_for_remote_instance` first."
        logger.error(message)
        return False, message

    torch.cuda.empty_cache()
    success = False
    na = NetworkAddress(master_address, group_port)
    message = ""
    try:
        for _, weights in model.named_parameters():
            torch.distributed.broadcast(
                weights,
                src=0,
                group=send_group,
            )
        success = True
        message = (
            f"Succeeded to send weights through {na.to_host_port_str()} {group_name}."
        )
    except Exception as e:
        message = f"Failed to send weights: {e}."
        logger.error(message)

    # destroy the process group after sending weights
    del _weights_send_group[group_name]
    torch.distributed.distributed_c10d.destroy_process_group(send_group)
    torch.cuda.empty_cache()
    return success, message
