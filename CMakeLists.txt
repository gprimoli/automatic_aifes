cmake_minimum_required(VERSION 3.30)
project(Autoencoder C)

set(CMAKE_C_STANDARD 11)

include_directories(
        C:/Users/play_/CLionProjects/AIfES/src
        C:/Users/play_/CLionProjects/Autoencoder/lib
)

file(GLOB_RECURSE AIFES_SRC
        C:/Users/play_/CLionProjects/AIfES/src/*.c
        C:/Users/play_/CLionProjects/AIfES/src/*.cpp
)

file(GLOB GENERIC_SRCS
        C:/Users/play_/CLionProjects/Autoencoder/lib/*.c
)

add_library(AIfES STATIC ${AIFES_SRC})
add_library(GenericFuncs STATIC ${GENERIC_SRCS})

add_executable(Autoencoder mainDinamic.c)
target_link_libraries(Autoencoder AIfES GenericFuncs m)
