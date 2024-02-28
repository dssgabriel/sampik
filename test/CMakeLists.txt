add_executable(send_recv send_recv.cpp)
target_include_directories(send_recv PUBLIC ../src)
target_link_libraries(send_recv sampik Kokkos::kokkos MPI::MPI_CXX)
