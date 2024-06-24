# NVIDIA NCCL overview


## Features

- NCCL implements collectives in a single kernel handling both computation and communication.
- Multiple GPU support: both within and across nodes.
- Supports PCIe, NVLINK, InfiniBand, IP sockets.
- Simple C API (mostly for FFI reasons/ease of programming).
- In contrast to MPI, adds a `stream` argument.


## Provided primitives

### Collectives

- `Reduce`
- `ReduceScatter`
- `AllReduce`
- `AllGather`
- `Broadcast`

### Point-to-point

- `Send`
- `Recv`

Other collectives such as `all-to-all`, `gather` or `scatter` can be implemented in terms of point-to-point operations.


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
```cu
int rank;
ncclCommUserRank(comm, &rank);
ncclCommSplit(/*ncclComm_t:*/comm, /*int: key*/ 0, /*int: color*/ rank, /*ncclComm_t*:*/&newcomm, /*ncclConfig_t*:*/NULL);
```

### Finalization
Finalize with `ncclCommFinalize`. While completing all background operations, the communicator will switch to the
`ncclInProgress` state. When done, it will switch back to `ncclSuccess`. Users may retrieve the state using the
`ncclCommGetAsyncError` function, which may be blocking or non-blocking depending on the communicator's configuration.

Once finalized, destroy with `ncclCommDestroy`. If in the `ncclSuccess` state, this call is guaranteed to be
non-blocking.


## Error handling

All NCCL calls return an error code. If `NCCL_DEBUG` is set to `WARN`, human-readable error messages will be displayed.
If set to `INFO`, a call stack will also be printed.

+------------------------+-------------------------------------------+-------------------------------------------------+------------------------+----------------+
| Error                  | Description                               | Resolution                                      | Error handling         | Group behavior |
+------------------------+-------------------------------------------+-------------------------------------------------+------------------------+----------------+
| ncclSuccess            | No error                                  | None                                            | None                   | None           |
| ncclUnhandledCudaError | Error during a CUDA call (1)              | CUDA configuration / usage (1)                  | Communicator abort (5) | Global (6)     |
| ncclSystemError        | Error during a system call (1)            | System configuration / usage (1)                | Communicator abort (5) | Global (6)     |
| ncclInternalError      | Error inside NCCL (2)                     | Fix in NCCL (2)                                 | Communicator abort (5) | Global (6)     |
| ncclInvalidArgument    | An argument to a NCCL call is invalid (3) | Fix in the application (3)                      | None (3)               | Individual (3) |
| ncclInvalidUsage       | The usage of NCCL calls is invalid (4)    | Fix in the application (4)                      | Communicator abort (5) | Global (6)     |
| ncclInProgress         | The NCCL call is still in progress        | Poll for completion using ncclCommGetAsyncError | None                   | None           |
+------------------------+-------------------------------------------+-------------------------------------------------+------------------------+----------------+

1. Error in an external component, causing NCCL to fail too. 

2. NCCL bug, report to maintainers with `NCCL_DEBUG=INFO`.

3. Argument value is incorrect, e.g. a null-pointer or an out-of-bounds access. Does not affect NCCL, the called
   communicator functions normally.

4. Incorrect usage of NCCL's API.

5. Fatal error to the communicator, abort with `ncclCommAbort` and recreate it.

6. Dynamic errors for operations in a group are reported by `ncclGroupEnd`. Application must call abort on all
   communicators withing the group.

Errors can be caught asynchronously using `ncclCommGetAsyncError`, unlike "standard" CUDA errors.
