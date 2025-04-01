cmake_minimum_required(VERSION 3.25)
project(Autoencoder C)

set(CMAKE_C_STANDARD 11)

include_directories(
	AIfES/src
	Autoencoder/lib
)

file(GLOB_RECURSE AIFES_SRC
        AIfES/src/*.c
        AIfES/src/*.cpp)

file(GLOB GENERIC_SRCS
        Autoencoder/lib/*.c
)

add_library(AIfES STATIC ${AIFES_SRC})
add_library(GenericFuncs STATIC ${GENERIC_SRCS})

add_executable(Autoencoder mainDinamic.c)
target_link_libraries(Autoencoder AIfES m)