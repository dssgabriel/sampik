add_executable(test_sendrecv send_recv.cpp)
target_link_libraries(test_sendrecv sampik Kokkos::kokkos MPI::MPI_CXX)
