#ifndef MAIN_H
#define MAIN_H

#ifdef _WIN32
#define DIR_SEPARATOR '\\'
#else
  #define DIR_SEPARATOR '/'
#endif




#define INT2_MIN -2
#define INT2_MAX 1

#define BUF_MIN 128

unsigned int get_seed();

#endif //MAIN_H
