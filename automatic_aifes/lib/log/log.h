#ifndef LOG_H
#define LOG_H

#include <stdio.h>
#include <stdbool.h>

#include "log_internal.h"

typedef enum {
    LOG_NONE = 0,
    LOG_FILE_ONLY,
    LOG_ERROR_ONLY,
    LOG_WARNINGS,
    LOG_VERBOSE_NO_FILE,
    LOG_VERBOSE
} LogLevel;

extern LogLevel CURRENT_LOG_LEVEL;
extern FILE *log_file;

bool initLogFile(const char *path);

void closeLogFile(void);

#define LOG_ERROR(fmt, ...)  do { \
if (CURRENT_LOG_LEVEL >= LOG_ERROR_ONLY) { \
fprintf(stderr, "[%s] [ERROR] " fmt "\n", get_timestamp(), ##__VA_ARGS__); \
fflush(stderr); \
} \
if ((CURRENT_LOG_LEVEL == LOG_FILE_ONLY || CURRENT_LOG_LEVEL == LOG_VERBOSE) && log_file) { \
fprintf(log_file, "[%s] [ERROR] " fmt "\n", get_timestamp(), ##__VA_ARGS__); \
fflush(log_file); \
} \
} while (0)

// LOG_WARN
#define LOG_WARN(fmt, ...)  do { \
if (CURRENT_LOG_LEVEL >= LOG_WARNINGS) { \
fprintf(stderr, "[%s] [WARN] " fmt "\n", get_timestamp(), ##__VA_ARGS__); \
fflush(stderr); \
} \
if ((CURRENT_LOG_LEVEL == LOG_FILE_ONLY || CURRENT_LOG_LEVEL == LOG_VERBOSE) && log_file) { \
fprintf(log_file, "[%s] [WARN] " fmt "\n", get_timestamp(), ##__VA_ARGS__); \
fflush(log_file); \
} \
} while (0)

// LOG_INFO
#define LOG_INFO(fmt, ...)  do { \
if (CURRENT_LOG_LEVEL >= LOG_VERBOSE_NO_FILE) { \
fprintf(stdout, "[%s] [INFO] " fmt "\n", get_timestamp(), ##__VA_ARGS__); \
fflush(stdout); \
} \
if ((CURRENT_LOG_LEVEL == LOG_FILE_ONLY || CURRENT_LOG_LEVEL == LOG_VERBOSE) && log_file) { \
fprintf(log_file, "[%s] [INFO] " fmt "\n", get_timestamp(), ##__VA_ARGS__); \
fflush(log_file); \
} \
} while (0)

#endif //LOG_H
