#include <aiconfiguration.h>
#include <aiconfiguration_intern.h>

#include "ini.h"

bool load_config(aiconfiguration_t *conf, const char *filename) {
    return ini_parse(filename, handler, conf);
}
