Point to Point
==============

Non-blocking send & receive point-to-point operations.

Both take:

* an execution space: where will the communication happen;
* a handle: wrapping the communication space's communicator associated to the View exchange;
* a Kokkos View: the data to send/recv;
* a RankId: the target of the P2P communication.

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
