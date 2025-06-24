cmake_minimum_required(VERSION 3.25)
project(Autoencoder C)

set(CMAKE_C_STANDARD 11)
set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(AIFES_PATH "AIfES/src")
set(INI_PATH "inih-r60")
set(LIB_PATH "lib")
set(MAIN_FILE "mainDinamic.c")

include_directories(
        ${AIFES_PATH}
        ${INI_PATH}
        ${PROJECT_ROOT}
)

file(GLOB_RECURSE AIFES_SRC
        ${AIFES_PATH}/*.c
)



file(GLOB_RECURSE INI_SRC ${INI_PATH}/*.c)
file(GLOB_RECURSE LIB_SRC "${LIB_PATH}/*.c")
file(GLOB_RECURSE LIB_INCLUDE_DIRS LIST_DIRECTORIES true "${LIB_PATH}/*")
foreach (dir ${LIB_INCLUDE_DIRS})
    if (IS_DIRECTORY ${dir})
        include_directories(${dir})
    endif ()
endforeach ()



add_library(AIfES STATIC ${AIFES_SRC})
add_library(INI STATIC ${INI_SRC})
add_library(Lib STATIC ${LIB_SRC})

add_executable(automatic_aifes ${MAIN_FILE})

target_link_libraries(automatic_aifes
        "-Wl,--whole-archive" AIfES "-Wl,--no-whole-archive"
        Lib INI m
)