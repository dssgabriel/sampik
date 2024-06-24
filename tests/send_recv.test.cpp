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

#include <Kokkos_Core_fwd.hpp>
#include <sampik/sampik.hpp>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cassert>
#include <cstdint>
#include <iostream>

constexpr int64_t N = 65'536;

using ScalarType = double;
using Layout = Kokkos::LayoutRight;
using MemorySpace = Kokkos::HostSpace;
using ViewType = Kokkos::View<ScalarType*, Layout, MemorySpace>;

auto main(int argc, char* argv[]) -> int {
  int ret = 0;

  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  {
    sampik::Context ctx{Kokkos::DefaultExecutionSpace()};
    int rank = ctx.rank();
    int size = ctx.size();
    if (size != 2) {
      std::cerr << "world size must be 2 but is " << size << "\n";
      return -1;
    }

    ViewType v("v", N);
    ScalarType res_local{};
    ScalarType res_other{};

    if (rank == 0) {
      // Initialize view with all 1s
      Kokkos::parallel_for("init", N, KOKKOS_LAMBDA(int const i) { v(i) = 1; });

      // Send initialized view
      auto send_req = sampik::send(ctx, v, 1);

      // Perform a parallel reduction using Kokkos on the sent view
      ScalarType tmp{};
      Kokkos::parallel_reduce(
        "reduce rank 0", N, KOKKOS_LAMBDA(int const i, ScalarType& tmp) { tmp += v(i); }, res_local
      );

      // Wait for send to finish
      send_req.wait();

      // Receive result from rank 1
      MPI_Recv(
        &res_other, 1, sampik::Impl::mpi_type_v<ScalarType>, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE
      );
      // Send local result to rank 1
      MPI_Send(&res_local, 1, sampik::Impl::mpi_type_v<ScalarType>, 1, 2, MPI_COMM_WORLD);
    } else {
      // Receive initialized view from rank 0
      sampik::recv(ctx, v, 0).wait();

      // Perform a parallel reduction using Kokkos on the received view
      Kokkos::parallel_reduce(
        "reduce rank 1", N, KOKKOS_LAMBDA(int const i, ScalarType& tmp) { tmp += v(i); }, res_local
      );

      // Send local reduction result to rank 0
      MPI_Send(&res_local, 1, sampik::Impl::mpi_type_v<ScalarType>, 0, 1, MPI_COMM_WORLD);
      // Receive result from rank 0
      MPI_Recv(
        &res_other, 1, sampik::Impl::mpi_type_v<ScalarType>, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE
      );
    }

    // Check same result between ranks
    if (rank == 1) {
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
}
