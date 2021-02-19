project(channeltest)

set(CMAKE_CXX_STANDARD 20)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Boost REQUIRED COMPONENTS system iostreams)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

add_executable(channeltest main.cpp)
target_precompile_headers(channeltest PUBLIC /usr/include/gnuplot-iostream.h)
target_compile_options(channeltest PUBLIC -fstack-protector-all)
target_link_libraries(channeltest fftw3 portaudio Boost::system Boost::iostreams Threads::Threads ZLIB::ZLIB)
