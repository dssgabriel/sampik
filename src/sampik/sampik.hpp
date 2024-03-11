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

#pragma once

#include <sampik/impl/types.hpp>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cassert>
#include <cstdint>
#include <type_traits>

namespace Sampik {

using Tag = int32_t;

/// Sampik wrapper over MPI communicators.
class Communicator {
public:
  Communicator() {
    // TODO: implement logic for duplicating MPI_COMM_WORLD?
    MPI_Comm_dup(MPI_COMM_WORLD, &m_comm);
  }
  ~Communicator() { MPI_Comm_free(&m_comm); }

  auto get_inner_as_ptr() -> MPI_Comm* { return &m_comm; }

private:
  MPI_Comm m_comm;
};

/// SAMPIK wrapper over MPI statuses.
class Status {
public:
  Status(MPI_Status sta) : m_sta(sta) {}
  ~Status() = default;

private:
  MPI_Status m_sta;
};

/// SAMPIK wrapper over MPI requests.
class Request {
public:
  Request(MPI_Request req) : m_req(req) {}
  ~Request() = default;

  auto get_inner_as_ptr() -> MPI_Request* { return &m_req; }

private:
  MPI_Request m_req;
};

/// Send a `Kokkos::View` through MPI.
///
/// This function is non-blocking.
/// Assumptions:
/// - View is rank-1 contiguous;
/// - View's `value_type` is an MPI-defined datatype;
template <Tag tag, class SV, class... SP>
auto send(Communicator comm, Kokkos::View<SV, SP...> const& view, int32_t dst) -> Request {
  using ViewType = Kokkos::View<SV, SP...>;
  using ScalarType = typename ViewType::value_type;

#if defined(SAMPIK_GPU_AWARE_MPI)
  if constexpr (ViewType::rank <= 1) {
    if (view.span_is_contiguous()) {
      // OK to call MPI directly because we guarantee that MPI is GPU aware.
      MPI_Request req;
      MPI_Isend(
        view.data(),
        view.span(),
        Impl::mpi_type_v<ScalarType>,
        dst,
        tag,
        *comm.get_inner_as_ptr(),
        &req
      );
      return Request(req);
    } else {
      assert(false && "Sampik::send only supports contiguous views");
      // TODO: make this portable
      __builtin_unreachable();
    }
  } else {
    static_assert(ViewType::Rank > 1, "Sampik::send only supports rank-1 views");
  }
#else
#error "MPI implementation must be GPU-aware"
#endif
}

/// Receive a `Kokkos::View` through MPI.
///
/// This function is non-blocking.
/// Assumptions:
/// - View is rank-1 contiguous
/// - View's `value_type` is an MPI-defined datatype
template <Tag tag, class SV, class... SP>
auto recv(Communicator comm, Kokkos::View<SV, SP...> const& view, int32_t src) -> Request {
  using ViewType = Kokkos::View<SV, SP...>;
  using ScalarType = typename ViewType::value_type;

#if defined(SAMPIK_GPU_AWARE_MPI)
  if constexpr (ViewType::rank <= 1) {
    if (view.span_is_contiguous()) {
      // OK to call MPI directly because we guarantee that MPI is GPU aware.
      MPI_Request req;
      MPI_Irecv(
        view.data(),
        view.span(),
        Impl::mpi_type_v<ScalarType>,
        src,
        tag,
        *comm.get_inner_as_ptr(),
        &req
      );
      return Request(req);
    } else {
      assert(false && "Sampik::recv only supports contiguous views");
      // TODO: make this portable
      __builtin_unreachable();
    }
  } else {
    static_assert(ViewType::Rank > 1, "Sampik::recv only supports rank-1 views");
  }
#else
#error "MPI implementation must be GPU-aware"
#endif
}

auto wait(Request req) -> Status {
  MPI_Status sta;
  MPI_Wait(req.get_inner_as_ptr(), &sta);
  return Status(sta);
}
} // namespace Sampik
