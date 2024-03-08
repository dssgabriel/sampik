# API code examples

This document presents some minimal code examples using SAMPIK to give an idea of what it is to use it for message passing interop with Kokkos.

## Point-to-point

Basic ping-pong between two processes:
```cpp
#include <Kokkos_Core.hpp>
#include <sampik/sampik.hpp>

auto main() -> int {
  Sampik::initialize();
  Kokkos::initialize();
  {

  int sampik_comm_size = Sampik::get_comm_size();
  assert(sampik_comm_size == 2);

  int my_rank = Sampik::get_self_rank();
  int other_rank = (my_rank + 1) % sampik_comm_size;

  Kokkos::View<...> view = ...;

  if (my_rank == 0) {
    Sampik::Request send_req = Sampik::send(view, other_rank);
    // do some calculations...
    Sampik::wait(send_req);
  } else {
    Sampik::Request recv_req = Sampik::recv(view, other_rank);
    Sampik::wait(recv_req); // try to receive immediately
    // do some calculations...
  }

  }
  Kokkos::finalize();
  Sampik::finalize();
}
```
