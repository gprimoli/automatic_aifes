#ifndef GENERICFUNCS_H
#define GENERICFUNCS_H

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_LEN(arr) (sizeof(arr) / sizeof((arr)[0]))
#define BUFFER_SIZE 2048

int saveArray(char *fileName, float *fileContent, uint32_t size) {
    FILE *fp = fopen(fileName == NULL ? "weights" : fileName, "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening %s\n", fileName);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < size; i++) {
        fprintf(fp, "%f\n", fileContent[i]);
    }
    return EXIT_SUCCESS;
}

int readCSVA(char *fileName, float *arr, uint32_t colSize, uint32_t rowSize) {
    FILE *fp = fopen(fileName, "r");

    if (fp == NULL) {
        fprintf(stderr, "Error opening %s\n", fileName);
        return EXIT_FAILURE;
    }

    int el = 0;
    char line[BUFFER_SIZE];
    while (fgets(line, BUFFER_SIZE, fp) != NULL && el < rowSize * colSize) {
        char *token = strtok(line, ",");
        while (token != NULL) {
            if (el >= rowSize * colSize) break;
            arr[el++] = atof(token);
            token = strtok(NULL, ",");
        }
    }

    fclose(fp);
    return EXIT_SUCCESS;
}

int readCSVB(char *fileName, float *arr, uint32_t size) {
    FILE *fp = fopen(fileName, "r");
    if (fp == NULL) {
        fprintf(stderr, "Error opening %s\n", fileName);
        return EXIT_FAILURE;
    }

    char line[BUFFER_SIZE];
    int i = 0;
    while (fgets(line, BUFFER_SIZE, fp) != NULL) {
        char *token = strtok(line, ",");
        do {
            arr[i++] = atof(token);
            if (i > size) {
                fprintf(stderr, "Error reading %s\n", fileName);
                fclose(fp);
                return EXIT_FAILURE;
            }
        } while ((token = strtok(NULL, ",")) != NULL);
    }

    fclose(fp);
    return EXIT_SUCCESS;
}

void safeFree(void **ptr) {
    if (ptr && *ptr) {
        free(*ptr);
        *ptr = NULL;
    }
}

#endif //GENERICFUNCS_H
