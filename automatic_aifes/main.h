#ifndef MAIN_H
#define MAIN_H

#ifdef _WIN32
#define DIR_SEPARATOR '\\'
#else
  #define DIR_SEPARATOR '/'
#endif

#define BUF_MIN 128

#include <unistd.h>
#include <sys/time.h>

unsigned int get_seed();

#endif //MAIN_H
