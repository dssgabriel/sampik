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

class Request {
public:
  Request(MPI_Request req) : _req(req) {}

  Request(Request const& other) = delete;
  Request(Request&& other) = default;
  auto operator=(Request const& other) -> Request& = delete;
  auto operator=(Request&& other) -> Request& = default;

  ~Request() { MPI_Wait(&_req, MPI_STATUS_IGNORE); }

  [[nodiscard]] constexpr auto req() const -> MPI_Request const& { return _req; }

  auto wait() -> void { MPI_Wait(&_req, MPI_STATUS_IGNORE); }

  [[nodiscard]] auto test() -> bool {
    int flag;
    MPI_Test(&_req, &flag, MPI_STATUS_IGNORE);
    return 0 != flag;
  }

private:
  MPI_Request _req;
};

} // namespace sampik
