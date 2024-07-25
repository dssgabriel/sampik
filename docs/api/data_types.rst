Data Types
==========

A rank ID strong type, representing a process's identifier inside of the communication space communicator:

.. code-block:: cpp

	using RankId = int;

A handle wrapper for the underlying communication backend "communicator":

.. code-block:: cpp

	template <CommunicationSpace CommSpace>
	class Handle {
	 public:
	  template <KokkosExecutionSpace ExecSpace>
	  Handle(ExecSpace const& exec_space);

	  auto rank(void) -> RankId;

	  auto size(void) -> int;

	  auto get_inner(void) -> CommSpace::CommunicatorType;

	 private:
	  ...
	};

A request wrapper for waiting/testing the progression of the communication:

.. code-block:: cpp

	template <CommunicationSpace CommSpace>
	class Request {
	 public:
	  auto get_inner(void) -> CommSpace::RequestType;

	 private:
	  ...
	};
