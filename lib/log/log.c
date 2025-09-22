#include "log.h"

#include <stdlib.h>
#include <time.h>

LogLevel CURRENT_LOG_LEVEL = LOG_VERBOSE; // LOG_VERBOSE || LOG_VERBOSE_NO_FILE
FILE *log_file = NULL;

bool initLogFile(const char *path) {
    if (CURRENT_LOG_LEVEL == LOG_FILE_ONLY || CURRENT_LOG_LEVEL == LOG_VERBOSE) {
        time_t now = time(NULL);
        struct tm *t = localtime(&now);
        char filename[64];
        char fullpath[128];

        strftime(filename, sizeof(filename), "%Y-%m-%d_%H-%M-%S", t);
        snprintf(fullpath, sizeof(fullpath), "%s/%s", path, filename);
        snprintf(fullpath, sizeof(fullpath), "%s/%s_%u.log", path, filename, rand() % 100000);

        log_file = fopen(fullpath, "w");
        return log_file != NULL;
    }
    return true;
}

void closeLogFile(void) {
    if (log_file != NULL) {
        fclose(log_file);
        log_file = NULL;
    }
}
