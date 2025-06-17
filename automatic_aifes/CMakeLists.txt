cmake_minimum_required(VERSION 3.25)
project(Autoencoder C)

set(CMAKE_C_STANDARD 11)

include_directories(
	AIfES/src
	lib
        inih-r60
)

file(GLOB_RECURSE AIFES_SRC
        AIfES/src/*.c
        AIfES/src/*.cpp)

file(GLOB_RECURSE INI_SRC inih-r60/*.c)

file(GLOB_RECURSE LIB lib/*.c)

add_library(AIfES STATIC ${AIFES_SRC})
add_library(INI STATIC ${INI_SRC})
add_library(Lib STATIC ${LIB})

add_executable(automatic_aifes mainDinamic.c)
target_link_libraries(automatic_aifes AIfES Lib INI m)