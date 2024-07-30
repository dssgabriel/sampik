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
#include <sampik/impl/types.hpp>

#include <sampik/context.hpp>
#include <sampik/request.hpp>
#include <sampik/traits.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <mpi.h>

namespace sampik::Impl {

/// Receive a `Kokkos::View` through MPI.
///
/// This function is non-blocking.
/// Assumptions:
/// - View is contiguous
/// - View's `value_type` is an MPI-defined datatype
template <KokkosExecSpace ExecSpace, KokkosView RecvView>
auto recv(Context<ExecSpace> const& ctx, RecvView const view, int target) -> Request {
  using RecvScalar = typename RecvView::non_const_value_type;

  if (sampik::is_contiguous<RecvView>) {
    MPI_Request req;
    MPI_Irecv(
      sampik::data(view),
      sampik::span(view),
      Impl::mpi_type_v<RecvScalar>,
      target,
      0,
      ctx.comm(),
      &req
    );
    return Request(req);
  } else {
    MPI_Abort(ctx.comm(), -1);
  }
}

} // namespace sampik::Impl
