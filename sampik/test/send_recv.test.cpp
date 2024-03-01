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

#include <sampik/core.hpp>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cassert>
#include <cstdint>
#include <iostream>

constexpr int64_t N = 65'536;
constexpr int64_t M = 1'024;

using ScalarType = double;
using Layout = Kokkos::LayoutRight;
using MemorySpace = Kokkos::HostSpace;
using ViewType = Kokkos::View<ScalarType**, Layout, MemorySpace>;

struct Functor {
  ViewType v;
  ScalarType& tmp;

  Functor(ViewType _v, ScalarType& _tmp) : v(_v), tmp(_tmp) {}

  KOKKOS_INLINE_FUNCTION
  auto operator()(int const j, int const i) const -> void { tmp += v(i, j); }
};

auto main(int argc, char* argv[]) -> int {
  int ret = 0;

  int lvl_reqst = MPI_THREAD_MULTIPLE;
  int lvl_avail;
  MPI_Init_thread(&argc, &argv, lvl_reqst, &lvl_avail);
  assert(lvl_reqst == lvl_avail && "MPI Thread level request is not available");

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 2) {
    std::cerr << "world size must be 2 but is " << size << "\n";
    return -1;
  }

  Kokkos::initialize(argc, argv);
  {
    ViewType v("v", N, M);
    ScalarType res_local{};
    ScalarType res_other{};

    if (rank == 0) {
      // Initialize view with all 1s
      Kokkos::parallel_for(
        "init",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, M}),
        KOKKOS_LAMBDA(int const i, int const j) { v(i, j) = 1; }
      );

      // Send initialized view
      Sampik::send(v, 1, 0, MPI_COMM_WORLD);

      // Perform a parallel reduction using Kokkos on the sent view
      ScalarType tmp{};
      Kokkos::parallel_reduce(
        "reduce rank 0",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, M}),
        KOKKOS_LAMBDA(int const i, int const j, ScalarType& tmp) { tmp += v(i, j); },
        res_local
      );

      // Receive result from rank 1
      MPI_Recv(
        &res_other, 1, Sampik::Impl::mpi_type_v<ScalarType>, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE
      );
      // Send local result to rank 1
      MPI_Send(&res_local, 1, Sampik::Impl::mpi_type_v<ScalarType>, 1, 2, MPI_COMM_WORLD);
    } else {
      // Receive initialized view from rank 0
      Sampik::recv(v, 0, 0, MPI_COMM_WORLD);

      // Perform a parallel reduction using Kokkos on the received view
      Kokkos::parallel_reduce(
        "reduce rank 1",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, M}),
        KOKKOS_LAMBDA(int const i, int const j, ScalarType& tmp) { tmp += v(i, j); },
        res_local
      );

      // Send local reduction result to rank 0
      MPI_Send(&res_local, 1, Sampik::Impl::mpi_type_v<ScalarType>, 0, 1, MPI_COMM_WORLD);
      // Receive result from rank 0
      MPI_Recv(
        &res_other, 1, Sampik::Impl::mpi_type_v<ScalarType>, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE
      );
    }

    // Check same result between ranks
    if (rank == 0) {
      if (res_local == res_other) {
        std::cout << "PASSED\n";
        ret = 0;
      } else {
        std::cerr << "FAILED\n";
        ret = 1;
      }
    }
  }
  Kokkos::finalize();
  MPI_Finalize();

  return ret;
}
