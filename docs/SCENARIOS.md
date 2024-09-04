# Legacy MPI + KokkosComm

- KokkosComm app needs to call compute library that uses MPI:
  - should `CommHandle` be statically or dynamically typed?
  - needs to provide:
  ```cpp
  auto get_inner_communicator(CommHandle) -> MPI_Comm;
  ```

- Hybrid communications
  - Example:
  ```cpp
  reqs[0] = send(exec_space, handle, view, target);
  // reqs[1] = MPI_Send(view.data(), ...);
  reqs[1] = Request(MPI_Send(other_object, ...)); // wrap MPI_Request by constructing KokkosComm request
  wait_all(reqs);
  ```
  - provide this too?
  ```cpp
  auto get_inner_request(Request) -> MPI_Request;
  ```
