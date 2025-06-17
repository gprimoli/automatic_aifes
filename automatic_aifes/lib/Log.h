#ifndef LOG_H
#define LOG_H

#include <stdio.h>

typedef enum {
    LOG_NONE = 0,
    LOG_ERROR_ONLY,
    LOG_WARNINGS,
    LOG_VERBOSE
} LogLevel;

extern LogLevel CURRENT_LOG_LEVEL;

#define LOG_ERROR(fmt, ...)   if (CURRENT_LOG_LEVEL >= LOG_ERROR_ONLY) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__); fflush(stderr);
#define LOG_WARN(fmt, ...)    if (CURRENT_LOG_LEVEL >= LOG_WARNINGS) fprintf(stderr, "[WARN] " fmt "\n", ##__VA_ARGS__); fflush(stderr);
#define LOG_INFO(fmt, ...)    if (CURRENT_LOG_LEVEL >= LOG_VERBOSE) fprintf(stdout, "[INFO] " fmt "\n", ##__VA_ARGS__); fflush(stdout);

#endif //LOG_H
