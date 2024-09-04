MPI Interoperability
====================

Raw MPI objects
---------------

Sampik must provide interfaces that allow retrieving the underlying raw MPI objects that it wraps, in particular ``MPI_Comm`` and ``MPI_Request``:

.. code-block:: cpp

	sampik::get_inner_communicator(sampik::CommHandle const& handle) -> MPI_Comm;

	sampik::get_inner_request(sampik::Request const& req) -> MPI_Request;

Similarly, sampik wrappers can be constructed directly from raw MPI objects:

.. code-block:: cpp

	template <MpiCommunicationSpace>
	sampik::CommHandle(MPI_Comm const comm);

	template <MpiCommunicationSpace>
	sampik::Request(MPI_Request const req);

This ensures easy integration of sampik in existing Kokkos + MPI applications, where Kokkos is not used exclusively (e.g. there are calls to external libraries using MPI).

.. code-block:: cpp

	auto sampik_req = sampik::send(exec_space, handle, view, target);

	// External library using MPI is called using the sampik communicator
	MPI_Request mpi_req;
	external_function_call(sampik::get_inner_communicator(handle), &mpi_req, /* other params... */);

	// Wait on any of the requests, two possibilities:
	// A:
	MPI_Request requests[] = { sampik::get_inner_request(sampik_req), mpi_req };
	int matched_idx;
	MPI_Waitany(sizeof(requests), requests, &matched_idx, MPI_STATUS_IGNORE);
	// B:
	std::array requests = { sampik_req, sampik::Request(mpi_req) };
	auto matched_idx = sampik::wait_any(requests);
