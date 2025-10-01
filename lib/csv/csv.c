#include "csv.h"

#include "main.h"
#include <stdarg.h>
#include <stdlib.h>

#include "log.h"

bool open_csv(FILE **file, const char *base, const char *name, const char *mode) {
    char path[BUF_MIN];
    int n = snprintf(path, sizeof(path), "%s%c%s", base, DIR_SEPARATOR, name);
    if (n < 0 || (size_t) n >= sizeof(path)) return false;

    *file = fopen(path, mode);
    return *file != NULL;
}


bool open_dataset(FILE **files, const char *base, const char *name, const char *mode) {
    char tmp[BUF_MIN];
    int n;
    n = snprintf(tmp, sizeof(tmp), "x_%s", name);
    if (n < 0 || (size_t) n >= sizeof(tmp)) return false;
    bool a = open_csv(&files[0], base, tmp, mode);

    n = snprintf(tmp, sizeof(tmp), "y_%s", name);
    if (n < 0 || (size_t) n >= sizeof(tmp)) return false;
    bool b = open_csv(&files[1], base, tmp, mode);

    return a && b;
}

bool csv_read_one(float *out, FILE *f) {
    if (!f || !out) return false;

    char value_buf[64] = {0};
    int buf_pos = 0;
    int c;

    if (feof(f)) {
        rewind(f);
        clearerr(f);
    }

    while ((c = fgetc(f)) != EOF) {
        if (c == ',' || c == '\n' || c == '\r') {
            if (buf_pos > 0) {
                value_buf[buf_pos] = '\0';
                *out = strtof(value_buf, NULL);
                return true;
            }

            if (c == '\r') {
                int next = fgetc(f);
                if (next != '\n' && next != EOF) ungetc(next, f);
            }
        } else if (buf_pos < sizeof(value_buf) - 1) {
            value_buf[buf_pos++] = (char) c;
        }
    }

    if (buf_pos > 0) {
        value_buf[buf_pos] = '\0';
        *out = strtof(value_buf, NULL);
        return true;
    }

    return false;
}

bool csv_read(float arr[], uint32_t len, FILE *f) {
    if (!f || !arr || len == 0) return false;

    for (uint32_t i = 0; i < len; ++i) {
        if (!csv_read_one(&arr[i], f)) {
            return false;
        }
    }

    return true;
}

bool csv_write(const float arr[], uint32_t len, FILE *f) {
    if (!arr || len <= 0 || !f) return false;

    for (int i = 0; i < len; i++) {
        fprintf(f, "%.9g", arr[i]);

        if (i < len - 1) {
            fprintf(f, ",");
        } else {
            fprintf(f, "\n");
        }
    }

    return true;
}

void close_all_files(uint32_t count, ...) {
    va_list args;
    va_start(args, count);

    for (int i = 0; i < count; i++) {
        FILE *f = va_arg(args, FILE *);
        if (f) fclose(f);
    }

    va_end(args);
}

void reset_files(int count, ...) {
    va_list args;
    va_start(args, count);

    for (int i = 0; i < count; i++) {
        FILE *f = va_arg(args, FILE *);
        if (f) {
            rewind(f);
            clearerr(f);
        }
    }

    va_end(args);
}
