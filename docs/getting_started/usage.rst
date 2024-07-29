Basic Usage
===========

Ping demo code
-------------------

.. code-block:: cpp
	:linenos:

	sampik::initialize(argc, argv);
	
	int N = 1'000'000;
	Kokkos::View<double*> v(N);
	auto exec = Kokkos::DefaultExecutionSpace();
	auto comm = sampik::DefaultCommunicationSpace();
	
	auto handle = sampik::Handle(comm, exec);
	assert(handle.size() == 2);
	
	if (handle.rank() == 0) {
	  Kokkos::parallel_for(Kokkos::RangePolicy(space, 0, N), KOKKOS_LAMBDA(int const i) {
	    v(i) = double(i);
	  });
	  auto req = sampik::send(space, handle, v, 1);
	  sampik::wait(space, req);
	} else if (handle.rank() == 1) {
	  auto req = sampik::recv(space, handle, v, 0);
	  sampik::wait(space, req);
	  double result;
	  Kokkos::parallel_reduce(Kokkos::RangePolicy(space, 0, N), KOKKOS_LAMBDA(int const i, double& errs) {
	    errs += v(i) != double(i);
	  }, result);
	  assert(errs == 0);
	}

	sampik::finalize();

Halo exchange
-------------

.. code-block:: cpp
	:linenos:

	auto halo_exchange() -> void {
	  using Scalar = double;
	  using Grid   = Kokkos::View<Scalar**, Kokkos::LayoutRight>;
	
	  // Problem size per rank
	  int nx     = 512;
	  int ny     = 512;
	
	  auto comm = sampik::DefaultCommunicationSpace();
	  auto exec = Kokkos::DefaultExecutionSpace();
	
	  auto handle = sampik::Handle(comm, exec);
	  auto rank   = handle.rank();
	  auto size   = handle.size();
	
	  const int rs = std::sqrt(size);
	  const int rx = rank % rs;
	  const int ry = rank / rs;
	
	  if (rank < rs * rs) {
	    // Grid of elements, and a radius-1 halo
	    Grid grid("g", nx + 2, ny + 2);
	
	    // 2D index of numbers in minus and plus direction (periodic)
	    const int xm1 = (rx + rs - 1) % rs;
	    const int ym1 = (ry + rs - 1) % rs;
	    const int xp1 = (rx + 1) % rs;
	    const int yp1 = (ry + 1) % rs;
	
	    // Convert 2D rank into 1D rank
	    auto get_1d_rank = [=](int const x, int const y) -> int { return y * rs + x; };
	
	    auto make_pair = [](sampik::RankId a, sampik::RankId b) -> Kokkos::pair<sampik::RankId, sampik::RankId> {
	      return Kokkos::pair{a, b};
	    }
	
	    // Create send/recv subviews
	    auto xp1_s = Kokkos::subview(grid, grid.extent(0) - 2, make_pair(1, ny + 1), Kokkos::ALL);
	    auto xp1_r = Kokkos::subview(grid, grid.extent(0) - 1, make_pair(1, ny + 1), Kokkos::ALL);
	    auto xm1_s = Kokkos::subview(grid, 1, make_pair(1, ny + 1), Kokkos::ALL);
	    auto xm1_r = Kokkos::subview(grid, 0, make_pair(1, ny + 1), Kokkos::ALL);
	    auto yp1_s = Kokkos::subview(grid, make_pair(1, nx + 1), grid.extent(1) - 2, Kokkos::ALL);
	    auto yp1_r = Kokkos::subview(grid, make_pair(1, nx + 1), grid.extent(1) - 1, Kokkos::ALL);
	    auto ym1_s = Kokkos::subview(grid, make_pair(1, nx + 1), 1, Kokkos::ALL);
	    auto ym1_r = Kokkos::subview(grid, make_pair(1, nx + 1), 0, Kokkos::ALL);
	
	    // Start sending the data
	    std::vector<sampik::Request> send_reqs;
	    send_reqs.push_back(sampik::send(exec, handle, xp1_s, get_rank(xp1, ry)));
	    send_reqs.push_back(sampik::send(exec, handle, xm1_s, get_rank(xm1, ry)));
	    send_reqs.push_back(sampik::send(exec, handle, yp1_s, get_rank(rx, yp1)));
	    send_reqs.push_back(sampik::send(exec, handle, ym1_s, get_rank(rx, ym1)));
	
	    // Compute kernel is enqueued on the same execution space as the send operations
	    Kokkos::parallel_for(Kokkos::Policy(exec, ...), KOKKOS_LAMBDA(...) {
	      // Do some useful work here
	    });
	
	    // Start receiving the data
	    // Will start only after the previous `parallel_for`, as we're enqueuing on the same execution space
	    std::vector<sampik::Request> recv_reqs;
	    recv_reqs.push_back(sampik::recv(exec, handle, xm1_r, get_rank(xm1, ry)));
	    recv_reqs.push_back(sampik::recv(exec, handle, xp1_r, get_rank(xp1, ry)));
	    recv_reqs.push_back(sampik::recv(exec, handle, ym1_r, get_rank(rx, ym1)));
	    recv_reqs.push_back(sampik::recv(exec, handle, yp1_r, get_rank(rx, yp1)));
	
	    // Wait for comms to finish
	    for (auto [sr, rr]: std::views::zip(send_reqs, recv_reqs)) {
	      sr.wait();
	      rr.wait();
	    }
	  }
	}

MPI interop
-----------

Create sampik handles from MPI communicators, and retrieve the inner communicator from sampik handles:

.. code-block:: cpp
	:linenos:

	MPI_Comm comm;
	MPI_Comm_dup(MPI_COMM_WORLD, &comm);

	// sampik handle from MPI communicator
	auto handle = sampik::Handle(comm, Kokkos::HostExecutionSpace());

	// MPI communicator from sampik handle
	MPI_Comm inner = handle.get_inner();

