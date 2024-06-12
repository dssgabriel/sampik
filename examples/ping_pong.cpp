#include <Kokkos_Core.hpp>
#include <sampik/sampik.hpp>

auto main() -> int {
  sampik::initialize();
  Kokkos::initialize();
  {
    Kokkos::View<double*> v("v", 1000);
    auto id = sampik::get_id();

    if (id == 0) {
      Kokkos::parallel_for(v.extent(0), KOKKOS_LAMBDA(int i) { v(i) = double(i); });
      sampik::send(Channel{1}, v).wait();
      sampik::recv(Channel{1}, v).wait();
    } else {
      auto chan = Channel{0};
      sampik::recv(chan, v).wait();
      Kokkos::parallel_for(v.extent(0), KOKKOS_LAMBDA(int i) { v(i) += double(i); });
      sampik::send(chan, v).wait();
    }

    int errs;
    Kokkos::parallel_reduce(
      v.extent(0), KOKKOS_LAMBDA(int i, int& sum) { sum += v(i) != double(i) + double(i); }
    );
    assert(errs == 0);
  }

  Kokkos::finalize();
  sampik::finalize();
}
