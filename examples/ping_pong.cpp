#include <Kokkos_Core.hpp>
#include <sampik/sampik.hpp>

auto main() -> int {
  sampik::initialize();
  Kokkos::initialize();
  {
    sampik::Channel<sampik::DefaultChannel> chan;
    NodeId id = chan.id();
    static_assert(2 == chan.size());

    Kokkos::View<double*> v("v", 1000);
    if (id == 0) {
      Kokkos::parallel_for(v.extent(0), KOKKOS_LAMBDA(int i) { v(i) = double(i); });
      sampik::send(chan, v).wait();
      sampik::recv(chan, v).wait();
    } else {
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
