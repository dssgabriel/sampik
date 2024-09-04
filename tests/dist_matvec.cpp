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
 * Author: Gabriel Dos Santos <gabriel.dossantos@cea.fr, dss.gabriel@protonmail.com>
 **/

#include <sampik/sampik.hpp>

#include <Kokkos_Core.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <mpi.h>

using ScalarType  = double;
using Layout      = Kokkos::LayoutRight;
using ExecSpace   = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecSpace::memory_space;

using Matrix = Kokkos::View<ScalarType**, Layout, MemorySpace>;
using Vector = Kokkos::View<ScalarType*,  Layout, MemorySpace>;

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
  int global_m = 10'000;
  int global_n = 10'000;
  int local_m  = global_m / size + (rank < global_m % size ? 1 : 0);

  // Create local views
  Matrix A_local("A_local", local_m, global_n);
  Vector y_local("y_local", local_m);

  // Create global views
  Matrix A("A", global_m, global_n);
  Vector x("x", global_n);

  // Initialize `A` and `x` from the root process, and distribute them
  if (rank == 0) {
    Kokkos::parallel_for("init_A", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {global_m, global_n}),
      KOKKOS_LAMBDA(int const i, int const j) {
        A(i, j) = 1.0; // Example initialization
      }
    );

    Kokkos::parallel_for("init_x", n, KOKKOS_LAMBDA(int const i) {
      x(i) = 3.0; // Example initialization
    });

    // Scatter `A` to all processes
    std::vector<sampik::Request> reqs(size - 1);
    for (int i = 1; i < size; ++i) {
      auto start = i * local_m;
      auto end   = start + local_m - 1;
      auto slice = Kokkos::subview(A, std::make_pair(start, end), Kokkos::ALL);
      reqs[i] = sampik::send(space, handle, slice, i);
    }
    sampik::wait_all(reqs);
  } else {
    auto req = sampik::recv(space, handle, A_local, 0);
    sampik::wait(req);
  }

  // Broadcast `x` to all processes
  auto bcast_req = sampik::broadcast(space, handle, x, 0);
  sampik::wait(bcast_req);

  // Perform local matrix-vector multiplication using Kokkos Kernels
  ScalarType alpha = 1.0;
  ScalarType beta  = 0.0;
  KokkosBlas::gemv("N", alpha, A_local, x, beta, y_local);

  // Gather results (sampik does not provide a simple `gather` so we use `all_gather` instead)
  auto gather_req = sampik::all_gather(space, handle, A_local, A);
  sampik::wait(gather_req);

  sampik::finalize();
  return 0;  
}
