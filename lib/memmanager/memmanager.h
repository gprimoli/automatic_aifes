#ifndef MEMMANAGER_H
#define MEMMANAGER_H

#include <stdlib.h>

void *mem_calloc(size_t count, size_t size);
void mem_free();
size_t mem_total();
void mem_dealloc(void *ptr);

void safe_exit_failure(char *msg);
void safe_exit_success(char *msg);

#define SAFE_EXIT_FAILURE(msg) safe_exit_failure(msg);
#define SAFE_EXIT_SUCCESS(msg) safe_exit_success(msg);

#endif //MEMMANAGER_H
