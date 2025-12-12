/**
 * utils.h
 *
 * Project-wide utilities.
 */

#pragma once

#include <stdio.h>  // IWYU pragma: export

#ifdef NDEBUG
  #define DBG_PRINT(tokens)
#else
  #define DBG_PRINT(tokens) printf("%s\n", #tokens)
#endif

#define REQUIRE(cond, action) \
  do {                        \
    if (!(cond)) {            \
      DBG_PRINT(cond);        \
      action;                 \
    }                         \
  } while (0)