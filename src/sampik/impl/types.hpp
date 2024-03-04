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

#include <mpi.h>

#include <cstdint>
#include <type_traits>

namespace Sampik::Impl {
template <typename T>
inline auto mpi_type() -> MPI_Datatype {
  static_assert(std::is_void_v<T>, "unimplemented MPI type");
  return MPI_DATATYPE_NULL;
}

template <>
inline auto mpi_type<char>() -> MPI_Datatype {
  return MPI_CHAR;
}

template <>
inline auto mpi_type<int8_t>() -> MPI_Datatype {
  return MPI_INT8_T;
}

template <>
inline auto mpi_type<int16_t>() -> MPI_Datatype {
  return MPI_INT16_T;
}

template <>
inline auto mpi_type<int32_t>() -> MPI_Datatype {
  return MPI_INT32_T;
}

template <>
inline auto mpi_type<int64_t>() -> MPI_Datatype {
  return MPI_INT64_T;
}

template <>
inline auto mpi_type<long long signed int>() -> MPI_Datatype {
  return MPI_LONG_LONG_INT;
}

template <>
inline auto mpi_type<uint8_t>() -> MPI_Datatype {
  return MPI_UINT8_T;
}

template <>
inline auto mpi_type<uint16_t>() -> MPI_Datatype {
  return MPI_UINT16_T;
}

template <>
inline auto mpi_type<uint32_t>() -> MPI_Datatype {
  return MPI_UINT32_T;
}

template <>
inline auto mpi_type<uint64_t>() -> MPI_Datatype {
  return MPI_UINT64_T;
}

template <>
inline auto mpi_type<long long unsigned int>() -> MPI_Datatype {
  return MPI_UNSIGNED_LONG_LONG;
}

template <>
inline auto mpi_type<float>() -> MPI_Datatype {
  return MPI_FLOAT;
}

template <>
inline auto mpi_type<double>() -> MPI_Datatype {
  return MPI_DOUBLE;
}

template <>
inline auto mpi_type<long double>() -> MPI_Datatype {
  return MPI_LONG_DOUBLE;
}

template <typename T>
inline MPI_Datatype mpi_type_v = mpi_type<T>();
} // namespace Sampik::Impl
