Building sampik
===============

System Prerequisites
--------------------

Sampik requires a C++20 conforming compiler.

.. csv-table:: System prerequesites for building sampik
	:file: ../tables/system_prereqs.csv
	:header-rows: 1

Get sampik from GitHub:

.. code-block:: console

	$ git clone https://github.com/dssgabriel/sampik.git
	$ cd sampik

Build
-----

A basic CMake configure and build:

.. code-block:: console

	$ cmake -S <PATH_TO_SAMPIK> \
		-B <PATH_TO_BUILD_DIR> \
		-DKokkosROOT=<PATH_TO_KOKKOS_INSTALL>
	$ cmake --build <PATH_TO_BUILD_DIR>

Install
-------

You can then install sampik using the provided ``install`` target:

.. code-block:: console

	$ cmake --build <PATH_TO_BUILD_DIR> --target install

Test
----

Tests must be enabled via CMake configuration options:

.. code-block:: console

	$ cmake -Dsampik_ENABLE_TESTS=ON # other options...
	$ cmake --build <PATH_TO_BUILD_DIR>
	$ ctest --test-dir <PATH_TO_BUILD_DIR>/test
