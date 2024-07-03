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

#include <Kokkos_Core_fwd.hpp>
#include <sampik/impl/concepts.hpp>
#include <sampik/impl/types.hpp>
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

  explicit Context(ExecSpace const& space) : _space(space), _comm(MPI_COMM_NULL) {
    MPI_Comm_dup(MPI_COMM_WORLD, &_comm);
  }

  explicit Context(MPI_Comm comm) : _space(Kokkos::DefaultExecutionSpace()), _comm(MPI_COMM_NULL) {
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
    int rank{};
    MPI_Comm_rank(_comm, &rank);
    return rank;
  }

  [[nodiscard]] auto size() const -> int {
    int size{};
    MPI_Comm_size(_comm, &size);
    return size;
  }

private:
  ExecSpace _space;
  MPI_Comm _comm;
};

class Request {
public:
  explicit Request(MPI_Request req) : _req(req) {}
  
  Request(Request const& other) = delete;
  Request(Request&& other) = default;
  auto operator=(Request const& other) -> Request& = delete;
  auto operator=(Request&& other) -> Request& = default;
  ~Request() { MPI_Wait(&_req, MPI_STATUS_IGNORE); }

  [[nodiscard]] auto req() const -> MPI_Request const& { return _req; }

  auto wait() -> void { MPI_Wait(&_req, MPI_STATUS_IGNORE); }

  [[nodiscard]] auto test() -> bool {
    int flag{};
    MPI_Test(&_req, &flag, MPI_STATUS_IGNORE);
    return 0 != flag;
  }

private:
  MPI_Request _req;
};

/// Send a `Kokkos::View` through MPI.
///
/// This function is non-blocking.
/// Assumptions:
/// - View is contiguous;
/// - View's `value_type` is an MPI-defined datatype;
template <KokkosExecSpace ExecSpace, KokkosView SendView>
auto send(Context<ExecSpace> const& ctx, SendView const& view, int target) -> Request {
  using SendScalar = typename SendView::non_const_value_type;

  if (!sampik::is_contiguous<SendView>) {
    MPI_Abort(ctx.comm(), -1);
  }

  MPI_Request req = MPI_REQUEST_NULL;
  MPI_Isend(
    sampik::data(view),
    sampik::span(view),
    Impl::mpi_type_v<SendScalar>,
    target,
    0,
    ctx.comm(),
    &req
  );

  return Request{req};
}

/// Receive a `Kokkos::View` through MPI.
///
/// This function is non-blocking.
/// Assumptions:
/// - View is contiguous
/// - View's `value_type` is an MPI-defined datatype
template <KokkosExecSpace ExecSpace, KokkosView RecvView>
auto recv(Context<ExecSpace> const& ctx, RecvView const& view, int target) -> Request {
  using RecvScalar = typename RecvView::non_const_value_type;

  if (!sampik::is_contiguous<RecvView>) {
    MPI_Abort(ctx.comm(), -1);
  }

  MPI_Request req = MPI_REQUEST_NULL;
  MPI_Irecv(
    sampik::data(view),
    sampik::span(view),
    Impl::mpi_type_v<RecvScalar>,
    target,
    0,
    ctx.comm(),
    &req
  );

  return Request{req};
}

} // namespace sampik
