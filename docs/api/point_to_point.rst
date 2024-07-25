Point to Point
==============

Non-blocking send & receive point-to-point operations.

Both take:

* an execution space where the communication happens;
* a handle wrapping the communication space's communicator associated to the view being exchanged;
* a Kokkos view holding the data to send/receive;
* the rank ID of the target of the communication.

The execution space and the handle's communication space must match (e.g. a CUDA exec space must be associated with a NCCL comm space).

.. code-block:: cpp

	template <KokkosExecutionSpace ExecSpace, CommunicationSpace CommSpace>
	auto send(
	  ExecSpace const& exec,
	  Handle<CommSpace> const& handle,
	  KokkosView v,
	  RankId target
	) -> std::expected<Request, Error>;

.. code-block:: cpp

	template <KokkosExecutionSpace ExecSpace, CommunicationSpace CommSpace>
	auto recv(
	  ExecSpace const& exec,
	  Handle<CommSpace> const& handle,
	  KokkosView v,
	  RankId target
	) -> std::expected<Request, Error>;
