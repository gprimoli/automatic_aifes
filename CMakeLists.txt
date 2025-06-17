cmake_minimum_required(VERSION 3.30)
project(automatic_aifes C)

set(CMAKE_C_STANDARD 11)

include_directories(
        C:/Users/play_/CLionProjects/AIfES/src
        C:/Users/play_/CLionProjects/inih-r60
        C:/Users/play_/CLionProjects/automatic_aifes/lib
)

file(GLOB_RECURSE AIFES_SRC
        C:/Users/play_/CLionProjects/AIfES/src/*.c
        C:/Users/play_/CLionProjects/AIfES/src/*.cpp
)

file(GLOB_RECURSE INI_SRC C:/Users/play_/CLionProjects/inih-r60/*.c)

file(GLOB_RECURSE LIB C:/Users/play_/CLionProjects/automatic_aifes/lib/*.c)

add_library(AIfES STATIC ${AIFES_SRC})
add_library(INI STATIC ${INI_SRC})
add_library(Lib STATIC ${LIB})

add_executable(automatic_aifes mainDinamic.c)
target_link_libraries(automatic_aifes AIfES Lib INI m)