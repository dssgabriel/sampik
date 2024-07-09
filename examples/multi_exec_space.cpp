template <class ExecSpace, KokkosView View>
auto foo(ExecSpace const& space, View v, Rank dest, Comm comm) -> void {
  // Partition the execution space such that all but one resource is left to the first partition
  auto instances = Kokkos::partition_space(space, space.concurrency() - 1, 1);
  auto stream1_big = instances[0];
  auto stream2_small = instances[1];

  // Dispatch some work preparing our data (`v`) on the "big" exec space
  Kokkos::parallel_for("stream1_parallel_work", Kokkos::RangePolicy<ExecSpace>(stream1_big), Functor1(v, /* params... */));
  // Must fence the first exec space to ensure our data is ready
  stream1_big.fence();

  // Send the prepared data on the "small" exec space
  auto req = KokkosComm::send(stream2_small, v, dest, 0, comm);
  // Wait for the send to finish
  req.wait(); // calls `stream2_small.fence()`
}
