@PACKAGE_INIT@

find_package(MPI CONFIG REQUIRED)
find_package(Kokkos CONFIG REQUIRED)

include(${CMAKE_CURRENT_LIST_DIR}/sampikTargets.cmake)
