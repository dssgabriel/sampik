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

namespace sampik {

template <typename T>
struct Traits {
  static_assert(std::is_void_v<T>, "sampik::Traits not specialized for type");
};

/*! \brief This can be specialized to do custom behavior for a particular view*/
template <KokkosView View>
struct Traits<View> {
  using non_const_packed_view_type = Kokkos::View<
    typename View::non_const_data_type, typename View::array_layout, typename View::memory_space>;
  using packed_view_type = Kokkos::View<
    typename View::data_type, typename View::array_layout, typename View::memory_space>;
};

template <KokkosView View>
constexpr auto data(View const v) -> View::pointer_type {
  return v.data();
}

template <KokkosView View>
constexpr auto span(View const v) -> size_t {
  return v.span();
}

// true iff product of extents is span
template <KokkosView View>
auto is_contiguous(View const v) -> bool {
  return v.span_is_contiguous();
}

template <KokkosView View>
constexpr auto rank() -> size_t {
  return View::rank;
}

template <KokkosView View>
constexpr auto extent(View const v, int const i) -> size_t {
  return v.extent(i);
}

template <KokkosView View>
auto stride(View const v, int const i) -> size_t {
  return v.stride(i);
}

template <KokkosView View>
constexpr auto is_reference_counted() -> bool {
  return true;
}

} // namespace sampik
