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

#include <sampik/detail/types.hpp>

#include <sampik/concepts.hpp>
#include <sampik/handle.hpp>
#include <sampik/request.hpp>
#include <sampik/traits.hpp>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cstdint>
#include <span>
#include <type_traits>

namespace sampik {

enum class ReduceOp {
  Max,
  Min,
  Sum,
  Prod,
  Land,
  Band,
  Lor,
  Bor,
  Lxor,
  Bxor,
  Minloc,
  Maxloc,
  Replace,
};

auto initialize(int const& argc, char* argv[]) -> void {}

auto finalize(void) -> void {}

template <KokkosView View, KokkosExecSpace ExecSpace, CommunicationSpace CommSpace>
auto broadcast(
  ExecSpace const& exec, CommHandle<CommSpace, ExecSpace> const& handle, View const view,
  RankId root
) -> Request<CommSpace> {
  return Request(typename CommSpace::RequestType());
}

template <
  KokkosView ViewIn, KokkosView ViewOut, KokkosExecSpace ExecSpace, CommunicationSpace CommSpace>
auto reduce(
  ExecSpace const& exec, CommHandle<CommSpace, ExecSpace> const& handle, ViewIn const input,
  ViewOut output, ReduceOp op, RankId root
) -> Request<CommSpace> {
  return Request(typename CommSpace::RequestType());
}

template <
  KokkosView ViewIn, KokkosView ViewOut, KokkosExecSpace ExecSpace, CommunicationSpace CommSpace>
auto all_reduce(
  ExecSpace const& exec, CommHandle<CommSpace, ExecSpace> const& handle, ViewIn const input,
  ViewOut output, ReduceOp op
) -> Request<CommSpace> {
  return Request(typename CommSpace::RequestType());
}

template <
  KokkosView ViewIn, KokkosView ViewOut, KokkosExecSpace ExecSpace, CommunicationSpace CommSpace>
auto all_gather(
  ExecSpace const& exec, CommHandle<CommSpace, ExecSpace> const& handle, ViewIn const input,
  ViewOut output
) -> Request<CommSpace> {
  return Request(typename CommSpace::RequestType());
}

template <
  KokkosView ViewIn, KokkosView ViewOut, KokkosExecSpace ExecSpace, CommunicationSpace CommSpace>
auto all_to_all(
  ExecSpace const& exec, CommHandle<CommSpace, ExecSpace> const& handle, ViewIn const input,
  ViewOut output
) -> Request<CommSpace> {
  return Request(typename CommSpace::RequestType());
}

template <CommunicationSpace CommSpace>
auto wait(Request<CommSpace>& req) -> void {}

template <CommunicationSpace CommSpace>
auto wait_all(std::span<Request<CommSpace>&> reqs) -> void {}

template <CommunicationSpace CommSpace>
auto test(Request<CommSpace>& req) -> bool {
  return true;
}

template <CommunicationSpace CommSpace>
auto test_all(std::span<Request<CommSpace>&> reqs) -> bool {
  return true;
}

/// Send a `Kokkos::View` through MPI.
/// Assumptions:
/// - View is on the `HostSpace` memory space
/// - View is contiguous
/// - View's `value_type` is an MPI-defined datatype
template <KokkosView View, KokkosExecSpace ExecSpace, CommunicationSpace CommSpace>
auto send(
  ExecSpace const& exec, CommHandle<CommSpace, ExecSpace> const& handle, View const view,
  RankId destination
) -> Request<CommSpace> {
  using ScalarType = typename View::value_type;

  if constexpr (!std::is_same_v<typename View::memory_space, Kokkos::HostSpace>) {
    static_assert(
      std::is_same_v<typename View::memory_space, Kokkos::HostSpace>,
      "`sampik::send` only supports views that are in `HostSpace`"
    );
  }

  typename CommSpace::RequestType req;
  if (view.span_is_contiguous()) {
    MPI_Isend(
      view.data(),
      view.span(),
      detail::mpi_type_v<ScalarType>,
      destination,
      0,
      handle.get_inner(),
      &req
    );
  } else { // TODO:
    assert(false && "`sampik::send` only supports contiguous views");
  }
  return Request(req);
}

/// Receive a `Kokkos::View` through MPI.
/// Assumptions:
/// - View is on the `HostSpace` memory space
/// - View is contiguous
/// - View's `value_type` is an MPI-defined datatype
template <KokkosView View, KokkosExecSpace ExecSpace, CommunicationSpace CommSpace>
auto recv(
  ExecSpace const& exec, CommHandle<CommSpace, ExecSpace> const& handle, View const view,
  RankId source
) -> Request<CommSpace> {
  using ScalarType = typename View::value_type;

  if constexpr (!std::is_same_v<typename View::memory_space, Kokkos::HostSpace>) {
    static_assert(
      std::is_same_v<typename View::memory_space, Kokkos::HostSpace>,
      "`sampik::recv` only support Kokkos Views that are in `HostSpace`"
    );
  }

  typename CommSpace::RequestType req;
  if (view.span_is_contiguous()) {
    MPI_Irecv(
      view.data(), view.span(), detail::mpi_type_v<ScalarType>, source, 0, handle.get_inner(), &req
    );
  } else { // TODO:
    assert(false && "`sampik::recv` only supports contiguous views");
  }
  return Request(req);
}

} // namespace sampik
