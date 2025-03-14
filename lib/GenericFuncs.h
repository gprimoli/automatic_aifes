#ifndef GENERICFUNCS_H
#define GENERICFUNCS_H

#include <stdio.h>
#include <stdarg.h>

#define ARRAY_LEN(arr) (sizeof(arr) / sizeof((arr)[0]))

void handle_error(char *format, ...) {
    va_list args;
    va_start(args, format);

    vprintf(format, args);
    printf("\n");

    va_end(args);
    exit(500);
}

#endif //GENERICFUNCS_H
