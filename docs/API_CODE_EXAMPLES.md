# API design

This document presents some minimal code examples using SAMPIK to give an idea of what it is to use it for message passing interop with Kokkos.


## Programming model

### Communication Spaces

A communication space

### Rules

## Data types and Concepts

A `CommunicationSpace` concept that expresses a communication backend (e.g. MPI, NCCL, etc...).
```cpp
concept CommunicationSpace = ...;
```

A rank ID strong type, representing a process's identifier inside of the communication space communicator:
```cpp
using RankId = int;
```

A handle wrapper for the underlying communication backend "communicator":
```cpp
template <CommunicationSpace CommSpace>
class Handle {
 public:
  auto rank(void) -> RankId;
  auto size(void) -> int;

  ...
};
```

A request wrapper for waiting/testing the progression of the communication:
```cpp
template <CommunicationSpace CommSpace>
class Request {
  ...
};
```


## Initialization and Finalization

Initialization and finalization functions for setting/cleaning up the libraries environment:
```cpp
auto initialize(int const& argc, char* argv[]) -> void;
```
```cpp
auto finalize(void) -> void;
```


## Point-to-point

Non-blocking send & receive point-to-point operations.
Both take:
- an execution space: where will the communication happen;
- a handle: wrapping the communication space's communicator associated to the View exchange;
- a Kokkos View: the data to send/recv;
- a RankId: the target of the P2P communication.

```cpp
template <KokkosExecutionSpace ExecSpace, CommunicationSpace CommSpace>
auto send(
  ExecSpace const& exec,
  Handle<CommSpace> const& handle,
  KokkosView v,
  RankId target
) -> std::expected<Request, Error>;
```
```cpp
template <KokkosExecutionSpace ExecSpace, CommunicationSpace CommSpace>
auto recv(
  ExecSpace const& exec,
  Handle<CommSpace> const& handle,
  KokkosView v,
  RankId target
) -> std::expected<Request, Error>;
```


## Collectives

Non-blocking NCCL's basic collectives (apart from reduce-scatter). The execution space and communication space handle have the same behavior as the P2P functions.

```cpp
template <KokkosExecutionSpace ExecSpace, CommunicationSpace CommSpace>
auto broadcast(
  ExecSpace const& exec,
  Handle<CommSpace> const& handle,
  KokkosView input,
  KokkosView output,
  RankId root
) -> std::expected<Request, Error>;
```
```cpp
template <KokkosExecutionSpace ExecSpace, CommunicationSpace CommSpace>
auto reduce(
  ExecSpace const& exec,
  Handle<CommSpace> const& handle,
  KokkosView input,
  KokkosView output,
  ReduceOp op,
  RankId root
) -> std::expected<Request, Error>;
```
```cpp
template <KokkosExecutionSpace ExecSpace, CommunicationSpace CommSpace>
auto all_reduce(
  ExecSpace const& exec,
  Handle<CommSpace> const& handle,
  KokkosView input,
  KokkosView output,
  ReduceOp op
) -> std::expected<Request, Error>;
```
```cpp
template <KokkosExecutionSpace ExecSpace, CommunicationSpace CommSpace>
auto all_gather(
  ExecSpace const& exec,
  Handle<CommSpace> const& handle,
  KokkosView input,
  KokkosView output
) -> std::expected<Request, Error>;
```


## Utilities

Wait and test functions for checking/completing communications:
```cpp
auto wait(Request& req) -> std::expected<T, E>;
```
```cpp
auto wait_all(std::span<Request&> req) -> std::expected<T, E>;
```

```cpp
auto test(Request& req) -> std::optional<T>;
```
```cpp
auto test_all(std::span<Request&> req) -> std::optional<T>;
```

Note: the exact return type is still to be determined.


## Code example

Basic ping-pong between two processes:
```cpp
#include <Kokkos_Core.hpp>
#include <sampik/sampik.hpp>

auto main() -> int {
  Sampik::initialize();
  Kokkos::initialize();
  {

  int sampik_comm_size = Sampik::get_comm_size();
  assert(sampik_comm_size == 2);

  int my_rank = Sampik::get_self_rank();
  int other_rank = (my_rank + 1) % sampik_comm_size;

  Kokkos::View<...> view = ...;

  if (my_rank == 0) {
    Sampik::Request send_req = Sampik::send(view, other_rank);
    // do some calculations...
    Sampik::wait(send_req);
  } else {
    Sampik::Request recv_req = Sampik::recv(view, other_rank);
    Sampik::wait(recv_req); // try to receive immediately
    // do some calculations...
  }

  }
  Kokkos::finalize();
  Sampik::finalize();
}
```
