#include "csv.h"

#include "main.h"
#include <stdarg.h>
#include <stdlib.h>

bool open_csv(FILE **file, const char *base, const char *name, const char *mode) {
    char path[BUF_MIN];
    snprintf(path, sizeof(path), "%s%c%s", base, DIR_SEPARATOR, name);
    *file = fopen(path, mode);
    return *file != NULL;
}

bool csv_read(float arr[], uint32_t len, FILE *f) {
    if (!f || len <= 0) return false;

    char value_buf[64];
    int buf_pos = 0;
    int count = 0;
    int c;

    if (feof(f)) {
        rewind(f);
        clearerr(f);
    }

    while (count < len && (c = fgetc(f)) != EOF) {
        if (c == ',' || c == '\n' || c == '\r') {
            if (buf_pos > 0) {
                value_buf[buf_pos] = '\0';
                arr[count++] = strtof(value_buf, NULL);
                buf_pos = 0;
            }

            if (c == '\r') {
                int next = fgetc(f);
                if (next != '\n' && next != EOF) {
                    ungetc(next, f);
                }
            }
        } else if (buf_pos < sizeof(value_buf) - 1) {
            value_buf[buf_pos++] = (char) c;
        }
    }

    return count == len;
}

bool csv_write(const float arr[], uint32_t len, FILE *f) {
    if (!arr || len <= 0 || !f) return false;

    for (int i = 0; i < len; i++) {
        fprintf(f, "%.7f", arr[i]);

        if (i < len - 1) {
            fprintf(f, ",");
        } else {
            fprintf(f, "\n");
        }
    }

    fclose(f);
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