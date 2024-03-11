# Technical design decisions

This document describe the technical design decisions for the SAMPIK project for Kokkos + MPI interoperability.


## Language

The project is written in ISO C++20. Along with `if constexpr`, templated helper classes and functions, and other metaprogramming features available in previous standard versions, C++20 offers `concept`s and `requires` clauses that let us avoid SFINAE coding patterns altogether.

We also plan on supporting C++23's `mdspan`.


## Project layout

The project layout is based on a mix of the [Canonical Project Structure](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1204r0.html) proposed by WG21 (the C++ standards committee) and the generic layout of Rust crates:
```
sampik
├── src
│   ├── sampik
│   │   ├── impl
│   │   │   └── ...
│   │   └── sampik.hpp
│   └── unit_tests
├── docs
├── ...
├── integration_tests
└── perf_tests
```

This allows headers to be included using the `<>` style instead of quotes and contains the project name as a directory prefix:
```cpp
#include "sampik.hpp"                // Bad
#include <sampik.hpp>                // Bad
#include "../src/sampik/sampik.hpp"  // Bad

#include <sampik/sampik.hpp>         // Good
```


## Philosophy

Kokkos + MPI interoperability is what Kokkos users are fundamentally asking for. However, it may be wise not to entirely associate this effort with "MPI" specifically but instead encompass it more broadly under "Message Passing". This will allow the implementation of dedicated backends other than MPI, e.g. NCCL or RCCL.


## Semantics

We want to steer away from a mere 1:1 wrapper of the MPI API (which includes way too many functions). Instead, we provide less routines but with clearly defined semantics.

### Non-blocking API

Following Kokkos' approach, all communications initiated with SAMPIK have non-blocking/immediate semantics. This maps to non-blocking MPI functions (starting with `MPI_I*`). Other patterns that do not strictly follow the general "immediate" semantic can be optionnally opted-in using a template parameter (e.g. "buffered", "synchronous", etc.).

### Request completion

All SAMPIK communication functions return a `Sampik::Request` object that must be explicitly waited on or tested in order to complete the communication request. This can be done using `Sampik::wait*` and `Sampik::test*` respectively. Both of these routines (and their variants) behave similarly to `MPI_Wait*` and `MPI_Test*`.
See code examples in the [API pseudo-codes](./API_CODE_EXAMPLES.md) document.

### Calls from parallel regions

Calls from parallel regions are only allowed when in the `Kokkos::OpenMP` execution space.

### Thread ready and probing

SAMPIK will provide `Sampik::matching_probe` and `Sampik::matching_recv`, similar to `MPI_Improbe` and `MPI_Imrecv` respectively, for knowing message sizes and creating views accordingly. This will allow "on-the-fly" view creation with correct dimensions using the explicit `Sampik::create_view_from_recv`.

WIP.

### Interoperability with unrelated uses

WIP.


## API overview

### Helpers

Initialize SAMPIK:
```cpp
auto Sampik::initialize() -> Sampik::ErrCode;
```
Terminate SAMPIK:
```cpp
auto Sampik::finalize() -> Sampik::ErrCode;
```

Create a SAMPIK communicator:
```cpp
auto Sampik::create_communicator() -> Sampik::Communicator
```

Probe for a matching request (still WIP):
```cpp
template <Sampik::Tag tag>
auto Sampik::matching_probe(Sampik::Communicator comm, int src, Sampik::Request& req) -> Sampik::Status;
```

Wait on a request:
```cpp
auto Sampik::wait(Sampik::Request& req) -> Sampik::Status;
```

Test a request:
```cpp
auto Sampik::test(Sampik::Request& req) -> std::optional<Sampik::Status>;
```

### Point-to-point (P2P)

Sends a Kokkos view:
```cpp
template <Sampik::Tag tag, class SV, class... SP>
auto Sampik::send(Sampik::Communicator comm, Kokkos::View view<SV, SP...>, int dst) -> Sampik::Request
```

Receives a Kokkos view:
```cpp
template <Sampik::Tag tag, class SV, class... SP>
auto Sampik::recv(Sampik::Communicator comm, Kokkos::View<SV, SP...> view, int src) -> Sampik::Request
```

Receive a view from a previous request match (still WIP):
```cpp
template <Sampik::Tag tag, class SV, class... SP>
auto Sampik::matching_recv(Sampik::Communicator comm, Kokkos::View view<SV, SP...>, int src) -> Sampik::Request
```

Create a view from a previous request match and receive immediately (?):
```cpp
template <Sampik::Tag tag>
auto Sampik::create_view_from_recv(Sampik::Communicator comm, int src) -> Kokkos::View
```
