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

#include <sampik/comm_modes.hpp>
#include <sampik/context.hpp>
#include <sampik/request.hpp>
#include <sampik/traits.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <mpi.h>

namespace sampik::Impl {

template <typename Mode, KokkosExecSpace ExecSpace, KokkosView SendView>
auto send(Mode const&, Context<ExecSpace> const& ctx, SendView const view, int target) -> Request {
  using SendScalar = typename SendView::non_const_value_type;

  if (sampik::is_contiguous<SendView>) {
    MPI_Request req;
    if constexpr (std::is_same_v<Mode, StandardCommMode>) {
      MPI_Isend(
        sampik::data(view),
        sampik::span(view),
        Impl::mpi_type_v<SendScalar>,
        target,
        0,
        ctx.comm(),
        &req
      );
    } else if constexpr (std::is_same_v<Mode, SynchronousCommMode>) {
      MPI_Issend(
        sampik::data(view),
        sampik::span(view),
        Impl::mpi_type_v<SendScalar>,
        target,
        0,
        ctx.comm(),
        &req
      );
    } else if constexpr (std::is_same_v<Mode, ReadyCommMode>) {
      MPI_Irsend(
        sampik::data(view),
        sampik::span(view),
        Impl::mpi_type_v<SendScalar>,
        target,
        0,
        ctx.comm(),
        &req
      );
    }
    return Request(req);
  } else {
    MPI_Abort(ctx.comm(), -1);
  }
}

template <KokkosExecSpace ExecSpace, KokkosView SendView>
auto send(Context<ExecSpace> const& ctx, SendView const view, Rank target) -> Request {
  return send(DefaultCommMode(), ctx, view, target);
}

}; // namespace sampik::Impl
