/**
 * Copyright (C) 2024, CEA/DAM
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

#include "impl/sampik_types.hpp"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <cstdint>
#include <type_traits>

namespace Sampik {

template <typename V>
auto send(
    V const& v,
    int32_t dst,
    int32_t tag,
    MPI_Comm comm
) -> std::enable_if_t<Kokkos::is_view<V>::value, int32_t> {
    using ScalarType = typename V::value_type;

    // Assume contiguous view
    if (v.span_is_contiguous()) {
        return MPI_Send(v.data(), v.span(), Impl::mpi_type_v<ScalarType>, dst, tag, comm);
    } else {
        assert(v.span_is_contiguous() && "`Sampik::send` only supports contiguous views");
        // TODO:
    }
    return -1;
}

template <typename V>
auto recv(
    V const& v,
    int32_t src,
    int32_t tag,
    MPI_Comm comm
) -> std::enable_if_t<Kokkos::is_view<V>::value, int32_t> {
    using ScalarType = typename V::value_type;

    // Assume contiguous view
    if (v.span_is_contiguous()) {
        return MPI_Recv(v.data(), v.span(), Impl::mpi_type_v<ScalarType>, src, tag, comm, nullptr);
    } else {
        assert(v.span_is_contiguous() && "`Sampik::recv` only supports contiguous views");
        // TODO:
    }
    return -1;
}

} // namespace Sampik
