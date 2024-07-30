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

#pragma once

#include <sampik/impl/concepts.hpp>
#include <sampik/traits.hpp>

#include <Kokkos_Core.hpp>
#include <mpi.h>

namespace sampik {

template <KokkosExecSpace ExecSpace>
class Context {
public:
  Context() : _space(Kokkos::DefaultExecutionSpace()), _comm(MPI_COMM_NULL) {
    MPI_Comm_dup(MPI_COMM_WORLD, &_comm);
  }

  Context(ExecSpace const& space) : _space(space), _comm(MPI_COMM_NULL) {
    MPI_Comm_dup(MPI_COMM_WORLD, &_comm);
  }

  Context(MPI_Comm comm) : _space(Kokkos::DefaultExecutionSpace()), _comm(MPI_COMM_NULL) {
    MPI_Comm_dup(comm, &_comm);
  }

  Context(ExecSpace const& space, MPI_Comm comm) : _space(space), _comm(MPI_COMM_NULL) {
    MPI_Comm_dup(comm, &_comm);
  }

  // NOTE: for now, we do not want to deal with copy/move ctors/assignments
  Context(Context const& other) = delete;
  Context(Context&& other) = default;
  auto operator=(Context const& other) -> Context& = delete;
  auto operator=(Context&& other) -> Context& = default;

  ~Context() {
    if (_comm != MPI_COMM_WORLD && _comm != MPI_COMM_SELF && _comm != MPI_COMM_NULL) {
      MPI_Comm_free(&_comm);
    }
  }

  [[nodiscard]] auto comm() const -> MPI_Comm const& { return _comm; }

  [[nodiscard]] auto space() const -> ExecSpace const& { return _space; }

  [[nodiscard]] auto rank() const -> int {
    int rank;
    MPI_Comm_rank(_comm, &rank);
    return rank;
  }

  [[nodiscard]] auto size() const -> int {
    int size;
    MPI_Comm_size(_comm, &size);
    return size;
  }

private:
  ExecSpace _space;
  MPI_Comm _comm;
};

}; // namespace sampik
