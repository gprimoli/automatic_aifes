#include "MemManager.h"
#include "Log.h"
#include <stdlib.h>

/*-------------------------------------------------------------*/
#include <stdarg.h>
#include <stdint.h>

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

void safe_free(void **ptr) ;
void free_all_resources(uint32_t count, ...) ;
void free_array(void **array, uint32_t count) ;
void mem_register(void *ptr, size_t size);
/*-------------------------------------------------------------*/

typedef struct MemNode {
    void *ptr;
    size_t size;
    struct MemNode *next;
} MemNode;

static MemNode *head = NULL;

void *mem_calloc(const size_t count, const size_t size) {
    void *ptr = calloc(count, size);
    mem_register(ptr, count * size);
    return ptr;
}

void mem_free() {
    MemNode *current = head;
    while (current) {
        MemNode *next = current->next;
        FREE_ALL_RESOURCES(current->ptr, current);
        current = next;
    }
    head = NULL;
}

size_t mem_total() {
    size_t total = 0;
    MemNode *current = head;
    while (current) {
        total += current->size;
        current = current->next;
    }
    return total;
}


/*-------------------------------------------------------------*/
void mem_register(void *ptr, const size_t size) {
    if (!ptr) return;
    MemNode *node = calloc(sizeof(MemNode), 1);
    if (!node) {
        LOG_ERROR("Failed to allocate memory for memory node");
        SAFE_EXIT_FAILURE;
    }
    node->ptr = ptr;
    node->size = size;
    node->next = head;
    head = node;
}

void safe_free(void **ptr) {
    if (ptr && *ptr) {
        free(*ptr);
        *ptr = NULL;
    }
}
/*-------------------------------------------------------------*/