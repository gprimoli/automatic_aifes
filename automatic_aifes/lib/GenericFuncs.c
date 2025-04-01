#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include <errno.h>
#include "Log.h"
#include "GenericFuncs.h"



#define ARRAY_LEN(arr) (sizeof(arr) / sizeof((arr)[0]))
#define BUFFER_SIZE 2048

void safe_free(void **ptr) {
    if (ptr && *ptr) {
        free(*ptr);
        *ptr = NULL;
    }
}

void free_all_resources(uint32_t count, ...) {
    va_list args;
    va_start(args, count);
    for (int i = 0; i < count; ++i) {
        void **ptr = va_arg(args, void **);
        safe_free(ptr);
    }
    va_end(args);
}

bool save_float_array(char *fileName, float *fileContent, uint32_t size) {
    FILE *fp = fopen(fileName == NULL ? "weights" : fileName, "w");
    if (fp == NULL) {
        LOG_ERROR("Error opening %s: %s", fileName, strerror(errno));
        return false;
    }
    for (int i = 0; i < size; i++) {
        fprintf(fp, "%f,", fileContent[i]);
    }
    return true;
}

bool init_and_fill(char *filename, void **arr, size_t element_size, uint32_t count,
                   bool (*reader)(char *filename, void *array, uint32_t count)) {
    *arr = calloc(count, element_size);
    if (*arr == NULL) {
        LOG_ERROR("Error allocating memory for an array (%s)", filename != NULL ? filename : "NULL");
        return false;
    }

    if (filename && reader) {
        if (!reader(filename, *arr, count)) {
            LOG_ERROR("Failed to read data from file: %s", filename);
            safe_free(arr);
            return false;
        }
    }

    return true;
}


bool readCSV(char *fileName, void *arr, uint32_t size) {
    FILE *fp = fopen(fileName, "r");
    if (fp == NULL) {
        LOG_ERROR("Error opening %s: %s", fileName, strerror(errno));
        return false;
    }
    char line[BUFFER_SIZE];
    uint32_t arrIndex = 0;
    while (fgets(line, sizeof(line), fp) != NULL) {
        char *token = strtok(line, ",");
        while (token != NULL) {
            if (arrIndex >= size) {
                LOG_ERROR("Exceeded the limit of %u values while reading %s", size, fileName);
                fclose(fp);
                return false;
            }
            char *endptr;
            float value = strtof(token, &endptr);
            if (endptr == token) {
                LOG_ERROR("Not valid value: '%s'", token);
                fclose(fp);
                return false;
            }
            ((float *) arr)[arrIndex++] = value; //TODO: wrapper della funzione
            token = strtok(NULL, ",");
        }
    }

    fclose(fp);
    return true;
}

int argmax(float *arr, uint32_t size) {
    //TODO: wrapper della funzione
    int max_idx = 0;
    float max_val = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    return max_idx;
}

