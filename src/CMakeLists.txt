add_library(sampik SHARED sampik.hpp impl/sampik_types.hpp)
target_link_libraries(sampik Kokkos::kokkos MPI::MPI_CXX)