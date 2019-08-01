# Stubs for torch.distributed.distributed_c10d (Python 3)
#
# NOTE: This dynamically typed stub was automatically generated by stubgen.

from .rendezvous import register_rendezvous_handler, rendezvous
from typing import Any, Optional

class Backend:
    UNDEFINED: str = ...
    GLOO: str = ...
    NCCL: str = ...
    MPI: str = ...
    TCP: str = ...
    def __new__(cls, name: Any): ...
dist_backend = Backend

class reduce_op:
    __members__: Any = ...
    def __init__(self) -> None: ...
    def __getattribute__(self, key: Any): ...

class group:
    WORLD: Any = ...

class GroupMember:
    WORLD: Any = ...
    NON_GROUP_MEMBER: Any = ...

def is_mpi_available(): ...
def is_nccl_available(): ...
def is_gloo_available(): ...
def is_initialized(): ...
def get_backend(group: Any = ...): ...
def init_process_group(backend: Any, init_method: Optional[Any] = ..., timeout: Any = ..., world_size: int = ..., rank: int = ..., store: Optional[Any] = ..., group_name: str = ...) -> None: ...
def destroy_process_group(group: Any = ...) -> None: ...
def get_rank(group: Any = ...): ...
def get_world_size(group: Any = ...): ...
def isend(tensor: Any, dst: Any, group: Any = ..., tag: int = ...): ...
def irecv(tensor: Any, src: Any, group: Any = ..., tag: int = ...): ...
def send(tensor: Any, dst: Any, group: Any = ..., tag: int = ...) -> None: ...
def recv(tensor: Any, src: Optional[Any] = ..., group: Any = ..., tag: int = ...): ...
def broadcast_multigpu(tensor_list: Any, src: Any, group: Any = ..., async_op: bool = ..., src_tensor: int = ...): ...
def broadcast(tensor: Any, src: Any, group: Any = ..., async_op: bool = ...): ...
def all_reduce_multigpu(tensor_list: Any, op: Any = ..., group: Any = ..., async_op: bool = ...): ...
def all_reduce(tensor: Any, op: Any = ..., group: Any = ..., async_op: bool = ...): ...
def reduce_multigpu(tensor_list: Any, dst: Any, op: Any = ..., group: Any = ..., async_op: bool = ..., dst_tensor: int = ...): ...
def reduce(tensor: Any, dst: Any, op: Any = ..., group: Any = ..., async_op: bool = ...): ...
def all_gather_multigpu(output_tensor_lists: Any, input_tensor_list: Any, group: Any = ..., async_op: bool = ...): ...
def all_gather(tensor_list: Any, tensor: Any, group: Any = ..., async_op: bool = ...): ...
def gather(tensor: Any, gather_list: Any, dst: Any, group: Any = ..., async_op: bool = ...): ...
def scatter(tensor: Any, scatter_list: Any, src: Any, group: Any = ..., async_op: bool = ...): ...
def reduce_scatter_multigpu(output_tensor_list: Any, input_tensor_lists: Any, op: Any = ..., group: Any = ..., async_op: bool = ...): ...
def reduce_scatter(output: Any, input_list: Any, op: Any = ..., group: Any = ..., async_op: bool = ...): ...
def barrier(group: Any = ..., async_op: bool = ...): ...
def new_group(ranks: Optional[Any] = ..., timeout: Any = ..., backend: Optional[Any] = ...): ...