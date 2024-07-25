Concepts
========

``sampik::CommunicationSpace``
------------------------------

A concept that expresses which communication backend (e.g. MPI, NCCL, etc...) is being used by sampik to perform communication operations:

.. code-block:: cpp

	concept CommunicationSpace = ...;

``sampik::KokkosExecutionSpace``
--------------------------------

A concept to represent Kokkos's execution spaces, i.e. the place *where* code can actually be executed. From sampik's point of view, this translates to where the code that performs the communication is executed.

For instance, NCCL implements its communication routines as CUDA kernels. This means that when calling a sampik function using a NCCL Communication Space, the code performing the communication is executed on a CUDA device.

.. code-block:: cpp

	concept KokkosExecutionSpace = ...;
