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

#include <cstdint>
#include <type_traits>

namespace Sampik {
/// Send a `Kokkos::View` through MPI.
/// Assumptions:
/// - View is on the `HostSpace` memory space
/// - View is contiguous
/// - View's `value_type` is an MPI-defined datatype
template <class SV, class... SP>
auto send(Kokkos::View<SV, SP...> const& view, int32_t dst, int32_t tag, MPI_Comm comm) -> int32_t {
  using ViewType = Kokkos::View<SV, SP...>;
  using ScalarType = typename ViewType::value_type;

  if constexpr (!std::is_same_v<typename ViewType::memory_space, Kokkos::HostSpace>) {
    static_assert(false, "`Sampik::send` only supports views that are in `HostSpace`");
  }

  if (view.span_is_contiguous()) {
    return MPI_Send(view.data(), view.span(), Impl::mpi_type_v<ScalarType>, dst, tag, comm);
  } else { // TODO:
    assert(false && "`Sampik::send` only supports contiguous views");
    return -1; // unreachable
  }
}

/// Receive a `Kokkos::View` through MPI.
/// Assumptions:
/// - View is on the `HostSpace` memory space
/// - View is contiguous
/// - View's `value_type` is an MPI-defined datatype
template <typename V>
auto recv(V const& v, int32_t src, int32_t tag, MPI_Comm comm) -> int32_t {
  using ScalarType = typename V::value_type;

  if constexpr (!std::is_same_v<typename V::memory_space, Kokkos::HostSpace>) {
    static_assert(true, "`Sampik::recv` only support Kokkos Views that are in `HostSpace`");
  }

  if (v.span_is_contiguous()) {
    return MPI_Recv(
      v.data(), v.span(), Impl::mpi_type_v<ScalarType>, src, tag, comm, MPI_STATUS_IGNORE
    );
  } else { // TODO:
    assert(v.span_is_contiguous() && "`Sampik::recv` only supports contiguous views");
    return -1; // unreachable
  }
}
} // namespace Sampik
