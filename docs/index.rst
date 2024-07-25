SAMPIK: Simple API for Message Passing Interop with Kokkos 
==========================================================

.. important:: Experimental MPI interfaces (and more!) for the Kokkos C++ Performance Portability Programming ecosystem.

.. warning:: This is a work in progress and is not yet ready for general use.

Sampik is a simple, generic API enabling interoperability between Kokkos and Message Passing-based libraries such as MPI or NCCL/RCCL.

Get sampik from GitHub:

.. code-block:: console

	$ git clone https://github.com/dssgabriel/sampik.git

Full manual:

.. toctree::
	:maxdepth: 1
	:caption: Getting started

	getting_started/build
	getting_started/usage

.. toctree::
	:maxdepth: 1
	:caption: Design Model

	design_model/communication_spaces
	design_model/constraints

.. toctree::
	:maxdepth: 1
	:caption: API

	api/concepts
	api/data_types
	api/initialization
	api/point_to_point
	api/collectives
	api/utilities

.. toctree::
	:maxdepth: 1
	:caption: Contributing

	contributing/documentation
	contributing/developper_guide
