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

#include <type_traits>

#include <mpi.h>
#if defined(SAMPIK_ENABLE_NCCL)
#include <nccl.h>
#endif

namespace sampik {

template <typename T>
struct is_communication_space : std::false_type{};

struct MpiCommunicationSpace {
  using CommunicatorType = MPI_Comm;
  using RequestType = MPI_Request;
};

using DefaultCommunicationSpace = MpiCommunicationSpace;
using FallbackCommunicationSpace = MpiCommunicationSpace;

template <>
struct is_communication_space<MpiCommunicationSpace> : std::true_type{};

#if defined(SAMPIK_ENABLE_NCCL)
struct NcclCommunicationSpace {
  using CommunicatorType = ncclComm_t;
  using RequestType = cudaStream_t;
};

using DefaultCommunicationSpace = NcclCommunicationSpace;
using FallbackCommunicationSpace = MpiCommunicationSpace;

template <>
struct is_communication_space<NcclCommunicationSpace> : std::true_type{};
#endif

template <typename T>
inline constexpr bool is_communication_space_v = is_communication_space<T>::value;

}
