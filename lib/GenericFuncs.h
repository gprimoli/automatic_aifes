#ifndef GENERICFUNCS_H
#define GENERICFUNCS_H

#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdbool.h>

#define FREE(x) safe_free((void**)&(x))
#define FREE_ALL_1(x) FREE(x)
#define FREE_ALL_2(x, ...) FREE(x); FREE_ALL_1(__VA_ARGS__)
#define FREE_ALL_3(x, ...) FREE(x); FREE_ALL_2(__VA_ARGS__)
#define FREE_ALL_4(x, ...) FREE(x); FREE_ALL_3(__VA_ARGS__)
#define FREE_ALL_5(x, ...) FREE(x); FREE_ALL_4(__VA_ARGS__)
#define FREE_ALL_6(x, ...) FREE(x); FREE_ALL_5(__VA_ARGS__)
#define FREE_ALL_7(x, ...) FREE(x); FREE_ALL_6(__VA_ARGS__)
#define FREE_ALL_8(x, ...) FREE(x); FREE_ALL_7(__VA_ARGS__)

#define GET_MACRO(_1,_2,_3,_4,_5,_6,_7,_8,NAME,...) NAME
#define FREE_ALL_RESOURCES(...) \
GET_MACRO(__VA_ARGS__, FREE_ALL_8, FREE_ALL_7, FREE_ALL_6, FREE_ALL_5, FREE_ALL_4, FREE_ALL_3, FREE_ALL_2, FREE_ALL_1)(__VA_ARGS__)


#define ARRAY_LEN(arr) (sizeof(arr) / sizeof((arr)[0]))
#define BUFFER_SIZE 2048

bool save_float_array(char *fileName, float *fileContent, uint32_t size);

bool readCSV(char *fileName, void *arr, uint32_t size);

void safe_free(void **ptr);

void free_all_resources(int count, ...);

#define INIT_ARR(ptr, type, count) \
init_and_fill(NULL, (void **)&(ptr), sizeof(type), count, NULL)

#define INIT_AND_FILL_ARR(filename, ptr, type, count, reader_func) \
init_and_fill(filename, (void **)&(ptr), sizeof(type), count, reader_func)

bool init_and_fill(char *filename, void **arr, size_t element_size, uint32_t count,
                   bool (*reader)(char *filename, void *array, uint32_t count));

#endif //GENERICFUNCS_H
