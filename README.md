# SAMPIK - Simple API for Message Passing Interoperability with Kokkos


## About - WIP

SAMPIK is a work-in-progress API for enabling interoperability between Kokkos and Message Passing (e.g. MPI). It enables users to transparently communicate Kokkos views using MPI as a backedn.

### Support features

- Point-to-point communications:
  - `send` and `receive` views

Assumptions:
- Views must be contiguous (but may be multi-dimensional)
- Views must reside on the `HostSpace` memory space

### Planned features

See the [Roadmap](./docs/ROADMAP.md).


## Quickstart

### Pre-requisites

Before starting, make sure the following software is installed on your machine:
- CMake 3.16+
- C++20 conforming compiler
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
Or directly using `ctest`:
```sh
cmake --build build --target test
```


## Contributing

Contributions are welcome and accepted as pull requests on [GitHub](https://github.com/dssgabriel/sampik).

You may also ask questions or file bug reports on the [issue tracker](https://github.com/dssgabriel/sampik/issues).


## License

Licensed under:
- MIT License ([LICENSE-MIT](https://github.com/dssgabriel/sampik/blob/master/LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))
at your option.  

The [SPDX](https://spdx.dev/) license identifier for this project is MIT.
