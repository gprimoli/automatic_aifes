#include "memmanager.h"
#include "log.h"
#include <stdlib.h>

/*-------------------------------------------------------------*/
#include <stdarg.h>
#include <stdint.h>
#include <string.h>

#include "csv.h"
#include "aifescustom.h"

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

void safe_free(void **ptr);

void free_all_resources(uint32_t count, ...);

void free_array(void **array, uint32_t count);

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

    if (ptr == NULL) {
        SAFE_EXIT_FAILURE("Not enought memory!");
    }

    return ptr;
}

char *mem_strdup(const char *src) {
    if (!src) return NULL;
    size_t len = strlen(src);
    char *dest = mem_calloc(len + 1, sizeof(char));
    if (!dest) return NULL;
    memcpy(dest, src, len);
    return dest;
}

void mem_free() {
    MemNode *current = head;
    while (current) {
        MemNode *next = current->next;
        if (current->ptr) {
            FREE(current->ptr);
        }
        if (current) {
            FREE(current);
        }
        current = next;
        next = NULL;
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

void mem_dealloc(void *ptr) {
    if (!ptr || !head)
        return;

    MemNode *current = head;
    MemNode *previous = NULL;

    while (current) {
        if (current->ptr == ptr) {
            if (previous) {
                previous->next = current->next;
            } else {
                head = current->next;
            }
            if (current->ptr) {
                FREE_ALL_RESOURCES(current->ptr, current);
            }
            return;
        }
        previous = current;
        current = current->next;
    }
}

void safe_exit_failure(char *msg, ...) {
    mem_free();
    if (msg && strlen(msg) > 0) {
        char formatted[128];
        va_list args;
        va_start(args, msg);
        vsnprintf(formatted, sizeof(formatted), msg, args);
        va_end(args);

        LOG_ERROR("%s", formatted);
    }
    CLOSE_ALL_FILES(log_file);// TODO: Creare un manger di file simile a questo della memoria. Ogni volta che apro un file lo mette in una lista e poi posso chiuderli correttamente in caso di errori!
    exit(EXIT_FAILURE);
}

void safe_exit_success(char *msg, ...) {
    mem_free();
    if (msg && strlen(msg) > 0) {
        char formatted[128];
        va_list args;
        va_start(args, msg);
        vsnprintf(formatted, sizeof(formatted), msg, args);
        va_end(args);

        LOG_INFO("%s", formatted);
    }
    CLOSE_ALL_FILES(log_file);// TODO: Creare un manger di file simile a questo della memoria. Ogni volta che apro un file lo mette in una lista e poi posso chiuderli correttamente in caso di errori!
    exit(EXIT_SUCCESS);
}

/*-------------------------------------------------------------*/
void mem_register(void *ptr, const size_t size) {
    if (!ptr) return;
    MemNode *node = calloc(sizeof(MemNode), 1);
    if (!node) {
        SAFE_EXIT_FAILURE("Failed to allocate memory for memory node");
    } else {
        node->ptr = ptr;
        node->size = size;
        node->next = head;
        head = node;
    }
}

void safe_free(void **ptr) {
    if (ptr && *ptr) {
        free(*ptr);
        *ptr = NULL;
    }
}

/*-------------------------------------------------------------*/
