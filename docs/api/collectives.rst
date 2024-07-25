Collectives
===========

Non-blocking NCCL's basic collectives (apart from reduce-scatter). The execution space and communication space handle have the same behavior as the P2P functions.

.. code-block:: cpp

	template <KokkosExecutionSpace ExecSpace, CommunicationSpace CommSpace>
	auto broadcast(
	  ExecSpace const& exec,
	  Handle<CommSpace> const& handle,
	  KokkosView input,
	  KokkosView output,
	  RankId root
	) -> std::expected<Request, Error>;

.. code-block:: cpp

	template <KokkosExecutionSpace ExecSpace, CommunicationSpace CommSpace>
	auto reduce(
	  ExecSpace const& exec,
	  Handle<CommSpace> const& handle,
	  KokkosView input,
	  KokkosView output,
	  ReduceOp op,
	  RankId root
	) -> std::expected<Request, Error>;

.. code-block:: cpp

	template <KokkosExecutionSpace ExecSpace, CommunicationSpace CommSpace>
	auto all_reduce(
	  ExecSpace const& exec,
	  Handle<CommSpace> const& handle,
	  KokkosView input,
	  KokkosView output,
	  ReduceOp op
	) -> std::expected<Request, Error>;

.. code-block:: cpp

	template <KokkosExecutionSpace ExecSpace, CommunicationSpace CommSpace>
	auto all_gather(
	  ExecSpace const& exec,
	  Handle<CommSpace> const& handle,
	  KokkosView input,
	  KokkosView output
	) -> std::expected<Request, Error>;
