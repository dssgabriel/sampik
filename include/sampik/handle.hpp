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

#include <sampik/detail/communication_spaces.hpp>
#include <sampik/concepts.hpp>

#include <mpi.h>

namespace sampik {

using RankId = int;

template <CommunicationSpace CommSpace, KokkosExecSpace ExecSpace>
class CommHandle {};

template <KokkosExecSpace ExecSpace>
class CommHandle<MpiCommunicationSpace, ExecSpace> {
 public:
  using HandleType = MpiCommunicationSpace::CommunicatorType;

  CommHandle(HandleType comm, ExecSpace const&) : _comm(comm) {}

  ~CommHandle() {
    if (_comm != MPI_COMM_WORLD && _comm != MPI_COMM_SELF && _comm != MPI_COMM_NULL) {
      MPI_Comm_free(&_comm);
    }
  }

  static auto split(HandleType comm, int color, int key, ExecSpace const&) -> CommHandle {
    HandleType new_comm{};
    MPI_Comm_split(comm, color, key, &new_comm);
    return CommHandle<MpiCommunicationSpace, ExecSpace>(new_comm);
  }

  static auto duplicate(HandleType comm, ExecSpace const&) -> CommHandle {
    HandleType new_comm{};
    MPI_Comm_dup(comm, &new_comm);
    return CommHandle<MpiCommunicationSpace, ExecSpace>(new_comm);
  }

  auto rank() -> RankId {
    int rank;
    MPI_Comm_rank(_comm, &rank);
    return RankId(rank);
  }

  auto rank_count() -> int {
    int rank_count;
    MPI_Comm_size(_comm, &rank_count);
    return rank_count;
  }

  constexpr auto get_inner() -> HandleType {
    return _comm;
  }

 private:
  HandleType _comm;
};

template <KokkosExecSpace ExecSpace>
CommHandle(MPI_Comm, ExecSpace const&) -> CommHandle<MpiCommunicationSpace, ExecSpace>;

#if defined(SAMPIK_ENABLE_NCCL)
template <KokkosExecSpace ExecSpace>
class CommHandle<NcclCommunicationSpace> {
 public:
  using HandleType = NcclCommunicationSpace::CommunicatorType;

  CommHandle(HandleType comm, ExecSpace const&) : _comm(comm) {}

  auto rank() -> RankId {
    int rank;
    ncclCommUserRank(_comm, &rank);
    return RankId(rank);
  }

  auto rank_count() -> int {
    int rank_count;
    ncclCommCount(_comm, &rank_count);
    return rank_count;
  }

  constexpr auto get_inner() -> HandleType {
    return _comm;
  }

 private:
  HandleType _comm;
};

template <KokkosExecSpace ExecSpace>
CommHandle(ncclComm_t, ExecSpace) -> CommHandle<NcclCommunicationSpace, ExecSpace>;
#endif

}
