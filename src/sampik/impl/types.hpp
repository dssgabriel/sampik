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

namespace sampik::Impl {

template <typename Scalar>
MPI_Datatype mpi_type() {
  using T = std::decay_t<Scalar>;

  if constexpr (std::is_same_v<T, char>) {
    return MPI_CHAR;
  } else if constexpr (std::is_same_v<T, unsigned char>) {
    return MPI_UNSIGNED_CHAR;
  } else if constexpr (std::is_same_v<T, short>) {
    return MPI_SHORT;
  } else if constexpr (std::is_same_v<T, unsigned short>) {
    return MPI_UNSIGNED_SHORT;
  } else if constexpr (std::is_same_v<T, int>) {
    return MPI_INT;
  } else if constexpr (std::is_same_v<T, unsigned>) {
    return MPI_UNSIGNED;
  } else if constexpr (std::is_same_v<T, long>) {
    return MPI_LONG;
  } else if constexpr (std::is_same_v<T, unsigned long>) {
    return MPI_UNSIGNED_LONG;
  } else if constexpr (std::is_same_v<T, long long>) {
    return MPI_LONG_LONG;
  } else if constexpr (std::is_same_v<T, unsigned long long>) {
    return MPI_UNSIGNED_LONG_LONG;
  } else if constexpr (std::is_same_v<T, std::int8_t>) {
    return MPI_INT8_T;
  } else if constexpr (std::is_same_v<T, std::uint8_t>) {
    return MPI_UINT8_T;
  } else if constexpr (std::is_same_v<T, std::int16_t>) {
    return MPI_INT16_T;
  } else if constexpr (std::is_same_v<T, std::uint16_t>) {
    return MPI_UINT16_T;
  } else if constexpr (std::is_same_v<T, std::int32_t>) {
    return MPI_INT32_T;
  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    return MPI_UINT32_T;
  } else if constexpr (std::is_same_v<T, std::int64_t>) {
    return MPI_INT64_T;
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    return MPI_UINT64_T;
  } else if constexpr (std::is_same_v<T, std::size_t>) {
    if constexpr (sizeof(std::size_t) == 1) {
      return MPI_UINT8_T;
    }
    if constexpr (sizeof(std::size_t) == 2) {
      return MPI_UINT16_T;
    }
    if constexpr (sizeof(std::size_t) == 4) {
      return MPI_UINT32_T;
    }
    if constexpr (sizeof(std::size_t) == 8) {
      return MPI_UINT64_T;
    }
  } else if constexpr (std::is_same_v<T, std::ptrdiff_t>) {
    if constexpr (sizeof(std::ptrdiff_t) == 1) {
      return MPI_INT8_T;
    }
    if constexpr (sizeof(std::ptrdiff_t) == 2) {
      return MPI_INT16_T;
    }
    if constexpr (sizeof(std::ptrdiff_t) == 4) {
      return MPI_INT32_T;
    }
    if constexpr (sizeof(std::ptrdiff_t) == 8) {
      return MPI_INT64_T;
    }
  } else if constexpr (std::is_same_v<T, float>) {
    return MPI_FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    return MPI_DOUBLE;
  } else if constexpr (std::is_same_v<T, long double>) {
    return MPI_LONG_DOUBLE;
  } else if constexpr (std::is_same_v<T, Kokkos::complex<float>>) {
    return MPI_COMPLEX;
  } else if constexpr (std::is_same_v<T, Kokkos::complex<double>>) {
    return MPI_DOUBLE_COMPLEX;
  } else if constexpr (std::is_trivially_copyable_v<T>) {
    return MPI_BYTE;
  } else {
    static_assert(std::is_void_v<T>, "mpi_type not implemented");
    return MPI_CHAR; // unreachable
  }
}

template <typename Scalar>
inline MPI_Datatype mpi_type_v = mpi_type<Scalar>();

} // namespace sampik::Impl
