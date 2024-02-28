# SAMPIK - Synergistic API for MPI + Kokkos interop

## About - WIP

SAMPIK is a simple work-in-progress API for enabling interoperability between Kokkos and MPI. It enables users to transparently distribute Kokkos views through MPI as if they were normal C-style buffers.

### Current support
- Views must be contiguous (may be multi-dimensional)
- Views must reside on the `HostSpace` memory space

### Planned features
- Support non-contiguous views:
  - Pack/unpack?
  - Transfer in multiple communications?
- Support any memory space: CUDA/GPU-aware MPI implementations? May use GPU-direct comms?
- Add missing point-to-point communications
- Add collectives (what would be the semantics for, e.g., `MPI_Reduce`?)
- Support for C++23 `mdspan`


## Quickstart

### Pre-requisites
Before starting, make sure the following software is installed on your machine:
- CMake 3.16+
- C++17 conforming compiler
- Kokkos (packaged as a submodule of this repo)
- MPI implementation (assuming OpenMPI for now)

### Build
```sh
cmake -S . -B build <KOKKOS_FLAGS...>
cmake --build build
```

### Run
Run the test program:
```sh
cmake --build build --target test_sendrecv
mpirun -n 2 build/test/test_sendrecv
```


## Contributing

Contributions are welcome and accepted as pull requests on [GitHub](https://github.com/dssgabriel/sampik).

You may also ask questions or file bug reports on the [issue tracker](https://github.com/dssgabriel/sampik/issues).


## License

Licensed under:
- MIT License ([LICENSE-MIT](https://github.com/dssgabriel/sampik/blob/master/LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))
at your option.  

The [SPDX](https://spdx.dev/) license identifier for this project is MIT.
