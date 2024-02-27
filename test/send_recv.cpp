/**
 * Copyright (C) 2024, CEA/DAM
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

#include "../src/sampik.hpp"

#include <Kokkos_Core.hpp>
#include <fmt/core.h>
#include <mpi.h>

#include <cassert>
#include <cstdint>

constexpr int64_t N = 1'000;
constexpr int64_t M = 16;

auto main(int32_t argc, char* argv[]) -> int32_t {
    int32_t lvl_reqst = MPI_THREAD_MULTIPLE;
    int32_t lvl_avail;
    MPI_Init_thread(&argc, &argv, lvl_reqst, &lvl_avail);
    Kokkos::initialize(argc, argv);
    {
        int32_t rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int32_t size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (size != 2) {
            fmt::print(stderr, "error: world size must be 2 but world is size: {}", size);
            return -1;
        }

        Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> v("v", N, M);
        double res_local = 0.0;
        double res_other;

        if (rank == 0) {
            // Initialize view to range from 0 to N
            Kokkos::parallel_for(
                "init",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, M}),
                KOKKOS_LAMBDA(int32_t i, int32_t j) { v(i, j) = 1.0; }
            );
            // Send initialized view
            Sampik::send(v, 1, 0, MPI_COMM_WORLD);

            // Perform a parallel reduction using Kokkos on the received view
            Kokkos::parallel_reduce(
                "reduce",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, M}),
                KOKKOS_LAMBDA(int32_t j, int32_t i, double& tmp) { tmp += v(i, j); },
                res_local
            );

            // Receive result from rank 1
            MPI_Recv(&res_other, 1, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, nullptr);
            // Send local sum to rank 1
            MPI_Send(&res_local, 1, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD);
        } else {
            // Receive initialized view from rank 0
            Sampik::recv(v, 0, 0, MPI_COMM_WORLD);

            // Perform a parallel reduction using Kokkos on the received view
            Kokkos::parallel_reduce(
                "reduce",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, M}),
                KOKKOS_LAMBDA(int32_t j, int32_t i, double& tmp) { tmp += v(i, j); },
                res_local
            );

            // Send local reduction result to rank 0
            MPI_Send(&res_local, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            // Receive result from rank 0
            MPI_Recv(&res_other, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, nullptr);
        }

        // Assert correct result between ranks
        assert(res_local == res_other);
        if (rank == 0) {
            fmt::print(
                "All ok!\n\tres from rank {}: {}\n\tres from rank {}: {}\n",
                rank,
                res_local,
                (rank + 1) % size,
                res_other
            );
        }
    }
    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
