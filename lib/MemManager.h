#ifndef MEMMANAGER_H
#define MEMMANAGER_H

#include <stddef.h>

void *mem_calloc(size_t count, size_t size);
void mem_free();
size_t mem_total();

#define SAFE_EXIT_FAILURE mem_free(); exit(EXIT_FAILURE);
#define SAFE_EXIT_SUCCESS mem_free(); exit(EXIT_SUCCESS);

#endif //MEMMANAGER_H
