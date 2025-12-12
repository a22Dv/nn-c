/**
 * utils.h
 *
 * Project-wide utilities.
 */

#pragma once

#include <stdio.h>   // IWYU pragma: export
#include <stdlib.h>  // IWYU pragma: export

#define REQUIRE(cond, action)                                                                    \
  do {                                                                                           \
    if (!(cond)) {                                                                               \
      fprintf(stderr, "%s:%s():%d | REQUIRE FAILED: %s\n", __FILE__, __func__, __LINE__, #cond); \
      action;                                                                                    \
    }                                                                                            \
  } while (0)

#ifndef NDEBUG
  #define ASSERT(cond)                                                                         \
    do {                                                                                       \
      if (!(cond)) {                                                                           \
        fprintf(                                                                               \
            stderr, "%s:%s():%d | ASSERTION FAILED: %s\n", __FILE__, __func__, __LINE__, #cond \
        );                                                                                     \
        fprintf(stderr, "Continue? [Y/n]: ");                                                  \
        char c = getchar();                                                                    \
        if ((c & ~0x20) == 'Y') {                                                              \
          __debugbreak();                                                                      \
        } else {                                                                               \
          exit(-1);                                                                            \
        }                                                                                      \
      }                                                                                        \
    } while (0)
#else
  #define ASSERT(cond)
#endif

#if defined(__GNUC__) || defined(__clang__)
  #define FRCINL __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
  #define FRCINL __forceinline
#else
  #define FRCINL inline
#endif