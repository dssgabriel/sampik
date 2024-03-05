# SAMPIK - Simple API for MPI + Kokkos


## About - WIP

SAMPIK is a work-in-progress API for enabling interoperability between Kokkos and MPI. It enables users to transparently communicate Kokkos views using MPI.

### Support features

- Point-to-point communications:
  - `send` and `receive` views

Assumptions:
- Views must be contiguous (but may be multi-dimensional)
- Views must reside on the `HostSpace` memory space

### Planned features

- Support non-contiguous views:
  - Pack/unpack?
  - Transfer in multiple communications?
- Support any memory space:
  - CUDA/GPU-aware MPI implementations?
  - May use GPUDirect comms?
- Support collectives (what would be the semantics? e.g. `MPI_Reduce` on multi-dimensional views?)
- Support for C++23 `mdspan`


## Quickstart

### Pre-requisites

Before starting, make sure the following software is installed on your machine:
- CMake 3.16+
- C++17 conforming compiler
- Kokkos (packaged as a submodule of this repo)
- MPI implementation

### Build

```sh
cmake -S . -B build <KOKKOS_FLAGS...>
cmake --build build
```

### Run

Run the test program:
```sh
cmake --build build --target send_recv
mpirun -n 2 build/tests/send_recv
```


## Contributing

Contributions are welcome and accepted as pull requests on [GitHub](https://github.com/dssgabriel/sampik).

You may also ask questions or file bug reports on the [issue tracker](https://github.com/dssgabriel/sampik/issues).


## License

Licensed under:
- MIT License ([LICENSE-MIT](https://github.com/dssgabriel/sampik/blob/master/LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))
at your option.  

The [SPDX](https://spdx.dev/) license identifier for this project is MIT.
