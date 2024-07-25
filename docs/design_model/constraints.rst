Constraints
===========

Parallel Regions
----------------

It is forbidden to call in of sampik's communication-related functions (point-to-point or collectives) from within a Kokkos parallel region. Doing so will likely result in a compilation error or a runtime crash.
