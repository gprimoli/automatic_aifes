#ifndef LOG_INTERNAL_H
#define LOG_INTERNAL_H
#include <time.h>

static const char *get_timestamp(void) {
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

#endif //LOG_INTERNAL_H
