/**
 * Copyright (C) 2024, CEA
 *
 * Licensed under the MIT License ("the License" hereafter);
 * You may not use this file except in compliance with the License.
 * You should have received a full copy of the License along with this program;
 * if not, you can obtain a copy at:
 *
 *    https://opensource.org/license/MIT
 *
 * The software is provided “as is”, without warranty of any kind, express
 * or implied, including but not limited to the warranties of merchantability,
 * fitness for a particular purpose and noninfringement. In no event shall
 * the authors or copyright holders be liable for any claim, damages or other
 * liability, whether in an action of contract, tort or otherwise, arising from,
 * out of or in connection with the software or the use or other dealings
 * in the software.
 *
 * Author: Gabriel Dos Santos <gabriel.dossantos@cea.fr>
 **/

// Distributed matrix-vector product, with 1D row-wise partitioning of the matrix:
//
//                              x
//                            ┌───┐
//                            │   │
//                            │ B │
//                            │ C │
// ╔══════════════════════════│ A │
// ║                          │ S │
// ║                          │ T │
// ║                          │   │
// ║                A         └───┘
// ║        ┌───────────────┐ ┌───┐
// ║   ╔═══▷│       0       │→│ 0 │
// ║   ║    ├───────────────┤ ├───┤
// ║   ╠═══▷│       1       │→│ 1 │
// ╚═══╣    ├───────────────┤ ├───┤
//     ╠═══▷│       2       │→│ 2 │
//     ║    ├───────────────┤ ├───┤
//     ╚═══▷│       3       │→│ 3 │
//          └───────────────┘ └───┘
//                              y

#include <sampik/sampik.hpp>

#include <Kokkos_Core.hpp>
// #include <KokkosBlas2_gemv.hpp>
#include <mpi.h>

#include <cstdint>

using ScalarType  = double;
using Layout      = Kokkos::LayoutRight;
using ExecSpace   = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecSpace::memory_space;

using Matrix = Kokkos::View<ScalarType**, Layout, MemorySpace>;
using Vector = Kokkos::View<ScalarType*,  Layout, MemorySpace>;

// Shared memory matrix-vector product. Assumes that the matrix is non-transposed ("N" in the BLAS API).
// No optimizations are done when `alpha` or `beta` are 0 or 1.
auto local_gemv(ScalarType alpha, Matrix const A, Vector const x, ScalarType beta, Vector y) -> void {
  uint64_t m = A.extent(0);
  uint64_t n = A.extent(1);

  Kokkos::parallel_for("gemv", m,  KOKKOS_LAMBDA(int const i) {
      ScalarType acc = 0;
      for (uint64_t j = 0; j < n; ++j) {
        acc += A(i, j) * x(j);
      }
      y(i) = alpha * acc + beta * y(i);
    }
  );
}

auto main(int argc, char* argv[]) -> int {
  // Initialize sampik: initializes both Kokkos and the enabled communication space (here, MPI)
  sampik::initialize(argc, argv);

  auto space = ExecSpace();
  // Initialize a handle from a raw `MPI_Comm`
  auto handle = sampik::CommHandle(MPI_COMM_WORLD, space);

  // Retrieve process rank and total rank count from the sampik handle
  auto rank = handle.rank();
  auto size = handle.rank_count();

  // Problem dimensions
  int global_m = 100'000;
  int global_n = 100'000;
  int local_m  = global_m / size + (rank < global_m % size ? 1 : 0);

  // Create global views
  Vector x("x", global_n);
  // Initialize `x` from the root process, and broadcast it
  if (rank == 0) {
    Kokkos::parallel_for("init_x", global_n, KOKKOS_LAMBDA(int const i) {
      x(i) = 3.0; // Example initialization
    });
  }
  auto bcast_req = sampik::broadcast(space, handle, x, 0);

  // Create local views
  Matrix A_local("A_local", local_m, global_n);
  Vector y_local("y_local", local_m);
  // Initialize local `A` on each process
  Kokkos::parallel_for("init_A", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {local_m, global_n}),
    KOKKOS_LAMBDA(int const i, int const j) {
      A_local(i, j) = 1.0; // Example initialization
    }
  );

  // Wait for broadcast of `x` to finish
  sampik::wait(bcast_req);

  // Perform local matrix-vector multiplication
  ScalarType alpha = 0.314;
  ScalarType beta  = 0.168;
  local_gemv(alpha, A_local, x, beta, y_local);

  sampik::finalize();
  return 0;  
}
