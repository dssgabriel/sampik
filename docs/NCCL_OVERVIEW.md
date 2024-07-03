# NVIDIA NCCL overview


## Features

- NCCL implements collectives in single kernels handling both computation and communication.
- Multiple GPU support: both within and across nodes.
- Supports PCIe, NVLINK, InfiniBand, IP sockets.
- Simple C API (mostly for FFI reasons/ease of programming).
- In contrast to MPI, adds a `stream` argument.


## Communicators

### Initialization

Use `ncclCommInitRank`, `ncclCommInitRankConfig` or `ncclCommInitAll` to create _N_ communicator objects, each
associated to a fixed rank and a CUDA device.

One must first create a unique object using `ncclGetUniqueId` and broadcast it to all participating threads/processes.

Configuration includes the following parameters:
- `blocking` (0/1, defaults to 1): non-blocking/blocking communicator behavior.
- `cgaClusterSize` (0..8, defaults to `arch >= sm90 ? 4 : 0`): Cooperative Group Array (CGA <==> "Clusters" introduced
  in Hopper architectures) size of kernels launched by NCCL.
- `minCTAs` (1..32, defaults to 1): Minimal number of Cooperative Thread Arrays (CTAs <==> "Blocks"). 
- `maxCTAs` (1..32, defaults to 32): Maximal number of CTAs.
- `netName` (string, default is undefined): Network module name. NCCL provides internal names that must exactly match
  one of: "IB" (InfiniBand verbs), "Socket" (TCP/IP socket). External network plugins define their own names. NCCL
  chooses the module automatically.
- `splitShare` (0/1, defaults to 0): Share resources with child communicator during split.

Create more communicators using `ncclCommSplit` (works similarly to `MPI_Comm_split`). All ranks must call it, ranks not
taking part in the created sub-group must specify `NCCL_SPLIT_NOCOLOR`.

Duplicate like so:
```cpp
int rank;
ncclCommUserRank(comm, &rank);
ncclCommSplit(
  comm,     /* ncclComm_t: comm */
  0,        /* int: key */
  rank,     /* int: color */
  &newcomm, /* ncclComm_t*: newcomm */
  nullptr   /* ncclConfig_t*: config */
);
```

**NCCL does not specifically need to be initialized using MPI. Any method that can broadcast the unique ID to all
participating threads/processes is sufficient (e.g. pipes, sockets, etc...).**

### Finalization
Finalize with `ncclCommFinalize`. While completing all background operations, the communicator will switch to the
`ncclInProgress` state. When done, it will switch back to `ncclSuccess`. Users may retrieve the state using the
`ncclCommGetAsyncError` function, which may be blocking or non-blocking depending on the communicator's configuration.

Once finalized, destroy with `ncclCommDestroy`. If in the `ncclSuccess` state, this call is guaranteed to be
non-blocking.


## Error handling

All NCCL calls return an error code. If `NCCL_DEBUG` is set to `WARN`, human-readable error messages will be displayed.
If set to `INFO`, a call stack will also be printed.

| **Error**                | **Description**                           | **Resolution**                                  | **Error handling**     | **Group behavior** |
| ------------------------ | ----------------------------------------- | ----------------------------------------------- | ---------------------- | ------------------ |
| `ncclSuccess`            | No error                                  | None                                            | None                   | None               |
| `ncclUnhandledCudaError` | Error during a CUDA call (1)              | CUDA configuration / usage (1)                  | Communicator abort (5) | Global (6)         |
| `ncclSystemError`        | Error during a system call (1)            | System configuration / usage (1)                | Communicator abort (5) | Global (6)         |
| `ncclInternalError`      | Error inside NCCL (2)                     | Fix in NCCL (2)                                 | Communicator abort (5) | Global (6)         |
| `ncclInvalidArgument`    | An argument to a NCCL call is invalid (3) | Fix in the application (3)                      | None (3)               | Individual (3)     |
| `ncclInvalidUsage`       | The usage of NCCL calls is invalid (4)    | Fix in the application (4)                      | Communicator abort (5) | Global (6)         |
| `ncclInProgress`         | The NCCL call is still in progress        | Poll for completion using ncclCommGetAsyncError | None                   | None               |

1. Error in an external component, causing NCCL to fail too. 

2. NCCL bug, report to maintainers with `NCCL_DEBUG=INFO`.

3. Argument value is incorrect, e.g. a null-pointer or an out-of-bounds access. Does not affect NCCL, the called
   communicator functions normally.

4. Incorrect usage of NCCL's API.

5. Fatal error to the communicator, abort with `ncclCommAbort` and recreate it.

6. Dynamic errors for operations in a group are reported by `ncclGroupEnd`. Application must call abort on all
   communicators withing the group.

Errors can be caught asynchronously using `ncclCommGetAsyncError`, unlike "standard" CUDA errors.

### Thread safety

NCCL primitives are not thread-safe and reentrant. Multiple threads should use separate communicators.


## Data pointers

NCCL accepts any pointers that are accessible from a CUDA device associated to the communicator object:
- device memory local to the associated CUDA device.
- host memory registered through `cudaHostRegister` or `cudaGetDevicePointer`.
- managed and unified memory.

Accessing device memory located on another device but using peer access produces an error, which can be silenced with
`NCCL_CHECK_POINTERS=0` (since v2.2.12).


## Group calls

Group functions (`ncclGroupStart` and `ncclGroupEnd`) may be used to merge multiple calls into one. This allows:
1. managing multiple GPUs from one thread (deadlocks).
2. aggregating operations for performance.
3. multiple point-to-point operations.

All three can be used in conjunction (at the exception of calling `ncclCommInitRank`).

### Manage multiple GPUs from one thread

The following code tells NCCL to treat all calls between `ncclGroupStart` and `ncclGroupEnd` as a single call to many
devices:
```cpp
ncclGroupStart();
for (int i = 0; i < nb_local_devices; ++i) {
  ncclOp(..., comm[i], stream[i]);
}
ncclGroupEnd();
```

Note that inside a group section, calls may return before having enqueued the operation on the stream. Hence, calls to
`cudaStreamSynchronize` must be done after the end of the group.

Groups are also necessary when creating a communicator when a thread manages more than one device:
```cpp
ncclGroupStart();
for (int i = 0; i < nb_local_devices; ++i) {
  cudaSetDevice(device[i]); // only required because calling `ncclCommInitRank` (since NCCL v1.x)
  ncclCommInitRank(&comm[i], nranks, comm_id, rank[i]);
}
ncclGroupEnd();
```

### Aggregated operations

Groups are useful for reducing kernel launch overhead (latency) as it only occurs for once for multiple operations.
Note that initialization function cannot be aggregated together, nor with communications.

```cpp
ncclGroupStart();
ncclBroadcast(sbuf0, rbuf0, cnt0, datatype, root, comm, stream);
ncclAllReduce(sbuf1, rbuf1, cnt1, datatype, comm, stream);
ncclAllReduce(sbuf2, rbuf2, cnt2, datatype, comm, stream);
ncclGroupEnd();
```

Additionally, it is possible to combine this with multi-GPU:
```cpp
ncclGroupStart();
for (int l = 0; l < nb_layers; ++l) {
  for (int d = 0; d < nb_devices; ++d) {
    ncclAllReduce(
      sbufs[d] + offsets[l],
      rbufs[d] + offsets[l],
      counts[l],
      datatype[l],
      comms[d],
      streams[d]
    );
  }
}
ncclGroupEnd();
```

### Non-blocking group operations

If a communicator is configured to be non-blocking, group functions are correspondingly non-blocking, i.e. returning
from `ncclGroupEnd` does not guarantee that all operations have been issued to the CUDA streams. The returned result
will let the user know if they are still in-progress (therefore must manually wait using `cudaStreamSynchronize` before
making any related CUDA calls), or have been successfully enqueued.


## Semantics


### Streams

NCCL communication calls are associated to a CUDA stream. Calls return once the operation has been enqueud to the
stream. Operations are then executed _asynchronously_ on the device. Its status can be queried, e.g. via
`cudaStreamSynchronize` or CUDA events.

It is possible to mix multiple streams within a group call. This enforces a dependency on all streams before the kernel
starts and blocks all streams until it completes, causing a global synchronization (as if it had been posted on every
stream).

### Differences with MPI

#### `ReduceScatter` operation
The `ncclReduceScatter` operation is similar to the `MPI_Reduce_scatter_block` operation, not `MPI_Reduce_scatter`. The
latter is intrinsically a "vector" function, while `MPI_Reduce_scatter_block` (later defined to fill the missing
semantics) provides regular counts similarly to the mirror function `MPI_Allgather`. This is an oddity of MPI which has
not been fixed for legitimate retro-compatibility reasons and that NCCL does not follow.

#### Send/Receive counts
While MPI allows for different send and receive counts and types, as long as `scnt * sizeof(stype) == rcnt * sizeof(rtype)`.
NCCL does not allow that, defining a single count and a single data type.

#### Message matching
`ncclRecv` does not support the equivalent of `MPI_ANY_SOURCE`; a specific source rank must always be provided.
Similarly, the provided receive count must match the send count. Further, there is no concept of message tags.

#### In-place operations
NCCL does not support an equivalent of `MPI_IN_PLACE`. However, in-place operations are still possible and are optimized
by NCCL if it detects that the send and receive buffers point to the same location.

For `ncclReduceScatter` and `ncclAllGather`, in place operations are done when the per-rank pointer is located at the
rank offset of the global buffer. More precisely, these calls are considered in place :
```cpp
ncclReduceScatter(
  data,
  data + rank * rcnt,
  rcnt,
  datatype,
  op,
  comm,
  stream
);

ncclAllGather(
  data + rank * scnt,
  data,
  scnt,
  datatype,
  op,
  comm,
  stream
);
```

### Interoperability with MPI

#### Progress
MPI defines a notion of progress which means that MPI operations need the program to call MPI functions (potentially
multiple times) to make progress and eventually complete.
In some implementations, progress on one rank may need MPI to be called on another rank. While this is usually bad for
performance, it can be argued that this is a valid MPI implementation. As a result, blocking on a NCCL collective
operation, for example calling `cudaStreamSynchronize`, may create a deadlock in some cases because not calling MPI on
one rank could block other ranks, preventing them from reaching the NCCL call that would unblock the NCCL collective on
the first rank.

In that case, the `cudaStreamSynchronize` call should be replaced by a loop like the following:
```cpp
cudaError_t err = cudaErrorNotReady;
int flag;
while (err == cudaErrorNotReady) {
  err = cudaStreamQuery(args->streams[i]);
  MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE); // Consider using `Improbe`?
}
```

#### Inter-GPU communication with CUDA-aware MPI
Using NCCL to perform inter-GPU communication concurrently with CUDA-aware MPI may create deadlocks. NCCL creates
inter-device dependencies, meaning that after it has been launched, a NCCL kernel will wait (and potentially block the
CUDA device) until all ranks in the communicator launch their NCCL kernel. CUDA-aware MPI may also create such
dependencies between devices depending on the MPI implementation.

Using both MPI and NCCL to perform transfers between the same sets of CUDA devices concurrently is therefore not
guaranteed to be safe.


## Provided primitives

### Collectives

Collective operations must be called on every rank (hence, CUDA device), or other ranks will wait indefinitely. 

- `Reduce`
- `ReduceScatter`
- `AllReduce`
- `AllGather`
- `Broadcast`

### Point-to-point (since 2.7)

- `Send`
- `Recv`

A point-to-point communications consists of a call to `ncclSend` on one rank, and a corresponding `ncclRecv` on the
other rank. Count and data type must match.

Other collectives such as `all-to-all`, `gather` or `scatter` can be implemented in terms of point-to-point operations
fused together in a group. Notably, this also enables implementing neighbor collectives.
Point-to-point calls within a group are blocking until the group completes, but progress independently and should never
block each other.

**Merge call that need to progress concurrently to avoid deadlocks.**

### Example implementation of collectives using P2P calls

**Scatter:**
```cpp
ncclGroupStart();
if (rank == root) {
  for (int r = 0; r < nb_ranks; ++r) {
    ncclSend(sbuf[r], size, type, r, comm, stream);
  }
}
ncclRecv(recvbuff, size, type, root, comm, stream);
ncclGroupEnd();
```

**Gather:**
```cpp
ncclGroupStart();
if (rank == root) {
  for (int r = 0; r < nb_ranks; ++r) {
    ncclRecv(rbuf[r], size, type, r, comm, stream);
  }
}
ncclSend(sbuf, size, type, root, comm, stream);
ncclGroupEnd();
```

**All-to-all:**
```cpp
ncclGroupStart();
for (int r = 0; r < nb_ranks; ++r) {
  ncclSend(sbuf[r], scnt, stype, r, comm, stream);
  ncclRecv(rbuf[r], rcnt, rtype, r, comm, stream);
}
ncclGroupEnd();
```


**Neighbor exchange (N-dimensions space):**
```cpp
ncclGroupStart();
for (int d = 0; d < nb_dims; ++d) {
  ncclSend(sbuf[d], scnt, stype, next[d], comm, stream);
  ncclRecv(rbuf[d], rcnt, rtype, prev[d], comm, stream);
}
ncclGroupEnd();
```
