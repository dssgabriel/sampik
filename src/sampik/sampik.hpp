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

#include <sampik/impl/types.hpp>

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cassert>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace sampik {

using NodeId = uint64_t;

class Channel {
  std::array<NodeId, N> endpoints;

public:
  Channel() {}
  ~Channel() {}
};

class Request {
public:
  Request() {}
  ~Request() = default;

  auto wait() -> void;
  auto test() -> void;
};

/// Send a `Kokkos::View` through MPI.
///
/// This function is non-blocking.
/// Assumptions:
/// - View is rank-1 contiguous;
/// - View's `value_type` is an MPI-defined datatype;
template <class SV, class... SP>
auto send(Channel chan, Kokkos::View<SV, SP...> const& view) -> Request {}

/// Receive a `Kokkos::View` through MPI.
///
/// This function is non-blocking.
/// Assumptions:
/// - View is rank-1 contiguous
/// - View's `value_type` is an MPI-defined datatype
template <class SV, class... SP>
auto recv(Channel chan, Kokkos::View<SV, SP...> const& view) -> Request {}

} // namespace sampik
