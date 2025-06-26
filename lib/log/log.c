#include "log.h"
#include <time.h>
#include "memmanager.h"

LogLevel CURRENT_LOG_LEVEL = LOG_VERBOSE_NO_FILE;
FILE *log_file = NULL;

void initLogFile(const char *path) {
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char filename[64];
    char fullpath[128];

    strftime(filename, sizeof(filename), "%Y-%m-%d_%H-%M-%S.log", t);
    snprintf(fullpath, sizeof(fullpath), "%s/%s", path, filename);

    log_file = fopen(fullpath, "w");
    if (!log_file)
        SAFE_EXIT_FAILURE("Impossibile creare il file di log");
}

void closeLogFile(void) {
    if (log_file != NULL) {
        fclose(log_file);
        log_file = NULL;
    }
}

const char *get_timestamp(void) {
    static char buf[32];
    time_t now = time(NULL);
    struct tm *t = localtime(&now);

    if (t) {
        strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", t);
    } else {
        snprintf(buf, sizeof(buf), "timestamp non disponibile");
    }

    return buf;
}
