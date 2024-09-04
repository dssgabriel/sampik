Collectives
===========

Non-blocking NCCL's basic collectives (apart from reduce-scatter). The execution space and communication space handle have the same behavior as the P2P functions.

.. code-block:: cpp

	template <
	  KokkosView View,
	  KokkosExecutionSpace ExecSpace,
	  CommunicationSpace CommSpace
	>
	auto broadcast(
	  ExecSpace const& exec,
	  Handle<CommSpace> const& handle,
	  View view,
	  RankId root
	) -> std::expected<Request, Error>;

.. code-block:: cpp

	template <
	  KokkosViewIn ViewIn,
	  KokkosViewOut ViewOut,
	  KokkosExecutionSpace ExecSpace,
	  CommunicationSpace CommSpace
	>
	auto reduce(
	  ExecSpace const& exec,
	  Handle<CommSpace> const& handle,
	  ViewIn input,
	  ViewOut output,
	  ReduceOp op,
	  RankId root
	) -> std::expected<Request, Error>;

.. code-block:: cpp

	template <
	  KokkosViewIn ViewIn,
	  KokkosViewOut ViewOut,
	  KokkosExecutionSpace ExecSpace,
	  CommunicationSpace CommSpace
	>
	auto all_reduce(
	  ExecSpace const& exec,
	  Handle<CommSpace> const& handle,
	  ViewIn input,
	  ViewOut output,
	  ReduceOp op
	) -> std::expected<Request, Error>;

.. code-block:: cpp

	template <
	  KokkosViewIn ViewIn,
	  KokkosViewOut ViewOut,
	  KokkosExecutionSpace ExecSpace,
	  CommunicationSpace CommSpace
	>
	auto all_gather(
	  ExecSpace const& exec,
	  Handle<CommSpace> const& handle,
	  ViewIn input,
	  ViewOut output,
	) -> std::expected<Request, Error>;

.. code-block:: cpp

	template <
	  KokkosViewIn ViewIn,
	  KokkosViewOut ViewOut,
	  KokkosExecutionSpace ExecSpace,
	  CommunicationSpace CommSpace
	>
	auto all_to_all(
	  ExecSpace const& exec,
	  Handle<CommSpace> const& handle,
	  ViewIn input,
	  ViewOut output
	) -> std::expected<Request, Error>;
