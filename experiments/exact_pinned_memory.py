"""Research-only CUDA host allocation without caching-allocator size rounding."""
import ctypes
from functools import lru_cache
import math
import sys
import weakref
import torch


_live={}
_events=[]
_serial=0


@lru_cache(maxsize=1)
def driver():
    library=ctypes.CDLL('libcuda.so.1')
    signatures={
        'cuCtxGetCurrent':[ctypes.POINTER(ctypes.c_void_p)],
        'cuCtxPushCurrent_v2':[ctypes.c_void_p],
        'cuCtxPopCurrent_v2':[ctypes.POINTER(ctypes.c_void_p)],
        'cuCtxSynchronize':[],
        'cuMemHostAlloc':[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t,ctypes.c_uint],
        'cuMemFreeHost':[ctypes.c_void_p],
    }
    for name,arguments in signatures.items():
        function=getattr(library,name);function.argtypes=arguments;function.restype=ctypes.c_int
    return library


def checked(function,*arguments):
    code=function(*arguments)
    if code:raise RuntimeError(f'{function.__name__} failed with CUDA status {code}')


def release(library,pointer,context,token):
    # ponytail: context-wide wait; only serialized single-context model use is proven.
    checked(library.cuCtxPushCurrent_v2,context)
    try:
        checked(library.cuCtxSynchronize)
        checked(library.cuMemFreeHost,pointer)
        size=_live.pop(token)
        _events.append(dict(kind='release',token=token,bytes=size))
    finally:
        previous=ctypes.c_void_p();checked(library.cuCtxPopCurrent_v2,ctypes.byref(previous))


def allocate(shape,*,dtype=torch.bfloat16,device='cuda'):
    """CPU Tensor storage owns its buffer until every tensor/view releases it.

    All asynchronous consumers must use the allocating CUDA context. The final
    buffer owner synchronizes that context before freeing the pinned range.
    """
    global _serial
    sizes={torch.uint8:1,torch.bfloat16:2,torch.float16:2,torch.float32:4}
    if (not isinstance(shape,(tuple,list)) or not shape or
        any(type(n) is not int or n<1 for n in shape) or dtype not in sizes):
        raise ValueError('Expected positive integer dimensions and a supported buffer dtype')
    size=math.prod(shape)*sizes[dtype]
    if size>sys.maxsize:raise ValueError('Pinned allocation exceeds supported buffer size')
    target=torch.device(device)
    if target.type!='cuda':raise ValueError('Exact pinned storage requires a CUDA owner context')
    library=driver();context=ctypes.c_void_p();pointer=ctypes.c_void_p()
    with torch.cuda.device(target):
        torch.cuda.current_stream(target)
        checked(library.cuCtxGetCurrent,ctypes.byref(context))
        if not context.value:
            anchor=torch.empty(1,device=target)
            checked(library.cuCtxGetCurrent,ctypes.byref(context))
            del anchor
        if not context.value:raise RuntimeError('CUDA owner context is unavailable')
        checked(library.cuMemHostAlloc,ctypes.byref(pointer),size,0)
    _serial+=1;token=_serial;_live[token]=size
    _events.append(dict(kind='allocate',token=token,bytes=size))
    try:buffer=(ctypes.c_ubyte*size).from_address(pointer.value)
    except BaseException:
        release(library,pointer,context,token);raise
    finalizer=weakref.finalize(buffer,release,library,pointer,context,token)
    finalizer.atexit=False  # Live ranges are reclaimed by the driver at process exit.
    tensor=torch.frombuffer(buffer,dtype=dtype).reshape(shape)
    if not tensor.is_pinned():raise RuntimeError('CUDA host range is not recognized as pinned')
    return tensor


def stats():
    return dict(live_allocations=len(_live),live_requested_bytes=sum(_live.values()),
        events=[dict(event) for event in _events])
