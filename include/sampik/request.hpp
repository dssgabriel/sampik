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

template <CommunicationSpace CommSpace>
class Request {};

template <>
class Request<MpiCommunicationSpace> {
 public:
  using ReqType = MpiCommunicationSpace::RequestType;

  Request(ReqType req) : _req(req) {}

  constexpr auto get_inner() -> ReqType {
    return _req;
  }

 private:
  ReqType _req;
};

Request(MPI_Request) -> Request<MpiCommunicationSpace>;

} // namespace sampik
