# Questions and ideas for Kokkos + MPI interoperability

This document tries to exhaustively list all questions and ideas that have come up in the Kokkos + MPI interoperability effort.


## Language and project layout

What should be the chosen C++ standard?
- [ ] C++17: rely on `if constexpr`, type traits, templated helper functions/classes. SFINAE should be the last resort.
- [ ] C++20: access to `concept` and `requires` clauses in place of SFINAE.

What should be the project structure?
- [ ] Kokkos-like:
  ```
  kokkos-core
  ├── perf_test
  ├── unit_test
  ├── src
  │   ├── impl
  │   └── ...
  ├── Kokkos_Core.hpp
  └── ...
  ```
  ```cpp
  #include <Kokkos_Core.hpp>
  #include <Kokkos_View.hpp>
  ```
- [ ] C++ [Canonical Project Structure](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1204r0.html)
  ```
  kokkos-comm
  ├── KokkosComms
  │   ├── impl
  │   │   └── ...
  │   ├── core.hpp
  │   └── ...
  ├── perf_tests
  └── unit_tests
  ```
  ```cpp
  #include <KokkosComms/core.hpp>
  ```

## API style

Should we explicitely expose this as MPI-only?
- [ ] `MPI` would/should appear explicitely -> `kokkos-mpi`, `KokkosMPI::<routine>`, etc...
- [ ] Implicit about how communications are implemented (could be MPI, NCCL, RCCL, etc...)-> `kokkos-comms`, `KokkosComms::<routine>`, etc...

How should we expose communication routines?
- [ ] Closely follow the MPI API:
  ```cpp
  int MPI_Ssend(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
  int MPI_Bsend(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
  int MPI_Recv(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Status* status);
  int MPI_Irecv(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request* req);

  template <class SV, class... SP>
  auto KokkosComms::ssend(Kokkos::View<SV, SP...> const& view, int dest, int tag, MPI_Comm comm) -> int;
  template <class SV, class... SP>
  auto KokkosComms::bsend(Kokkos::View<SV, SP...> const& view, int dest, int tag, MPI_Comm comm) -> int;
  template <class SV, class... SP>
  auto KokkosComms::recv(Kokkos::View<SV, SP...> const& view, int dest, int tag, MPI_Comm comm, MPI_Status& status) -> int;
  template <class SV, class... SP>
  auto KokkosComms::irecv(Kokkos::View<SV, SP...> const& view, int dest, int tag, MPI_Comm comm, MPI_Request& req) -> int;
  ```
- Don't expose MPI types, use template parameters to specify behavior and specify at compile-time:
  ```cpp
  enum class P2PMode {
    Default,
    Synchronized,
    Bufferized,
    Immediate,
    ...,
  };

  template <class SV, class... SP>
  auto KokkosComms::send<P2PMode>(Kokkos::View<SV, SP...> const& view, int dest, int tag, KokkosComms::Communicator comm) -> int;

  template <class SV, class... SP>
  auto KokkosComms::recv<P2PMode>(Kokkos::View<SV, SP...> const& view, int dest, int tag, KokkosComms::Communicator comm, KokkosComms::Status) -> int;
  ```
  How do we expose requests to the user for immediate/non-blocking communications in this case? Wrapper type that is returned?
- Don't expose MPI types, everything is assumed asynchronous, always return a wrapper type around an `MPI_Request`:
  ```cpp
  template <class SV, class... SP>
  auto KokkosComms::send(Kokkos::View<SV, SP...> const& view, int dest, int tag, KokkosComms::Communicator comm) -> KokkosComms::Request;

  template <class SV, class... SP>
  auto KokkosComms::recv(Kokkos::View<SV, SP...> const& view, int dest, int tag, KokkosComms::Communicator comm) -> KokkosComms::Request;

  auto KokkosComms::wait(KokkosComms::Request& req) -> KokkosComms::Status& sta;
  auto KokkosComms::test(KokkosComms::Request& req) -> KokkosComms::Status& sta; //Flag is still missing
  // Implement something like `MPI_Waitsome`, `MPI_Testany`, etc?
  // auto KokkosComms::test_some(std::vector<KokkosComms::Request>& reqs) -> int;
  ```

Do we allow non-contiguous views by default?
- [ ] Expose that this is not "normal" behavior (can have performance implications)
  ```cpp
  // Enforces "contiguousness" at compile-time (is it possible?)
  template <class SV, class... SP>
  auto KokkosComms::send(Kokkos::View<SV, SP...> const& view, int dest, int tag, KokkosComms::Communicator comm) -> KokkosComms::Request;

  // Make it explicit
  template <class SV, class... SP>
  auto KokkosComms::send_non_contiguous(Kokkos::View<SV, SP...> const& view, int dest, int tag, KokkosComms::Communicator comm) -> KokkosComms::Request;
  ```


## Collective communications

What would be the semantics of collectives like `MPI_Reduce`?
- Do we accept multi-dimensional views? What would it mean to reduce on such views?
