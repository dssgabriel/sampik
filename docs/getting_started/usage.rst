Basic Usage
===========

Ping demo code
-------------------

.. code-block:: cpp
	:linenos:
	

	#include <Kokkos_Core.hpp>
	#include <sampik/sampik.hpp>

	auto main(int argc, char* argv[]) -> int {
	  sampik::initialize(argc, argv);

	  int N = 1'000'000;
	  Kokkos::View<double*> v(N);
	  auto exec = Kokkos::DefaultExecutionSpace();
	  auto comm = sampik::DefaultCommunicationSpace();

	  auto handle = sampik::Handle(comm, exec);

	  if (handle.rank() == 0) {
	    Kokkos::parallel_for(N, KOKKOS_LAMBDA(int const i) {
	      v(i) = double(i);
	    });

	    auto req = sampik::send(space, handle, v, 1);
	    sampik::wait(space, req);
	  } else if (handle.rank() == 1) {
	    auto req = sampik::recv(space, handle, v, 0);
	    sampik::wait(space, req);
	    
	    double result;
	    Kokkos::parallel_reduce(N, KOKKOS_LAMBDA(int const i, double& errs) {
	      errs += v(i) != double(i);
	    }, result);
	    assert(errs == 0);
	  }

	  sampik::finalize();
	  return 0;
	}
