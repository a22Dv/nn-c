/**
 * utils.h
 *
 * Project-wide utilities.
 */

#pragma once

#include <stdio.h> // IWYU pragma: export

#define REQUIRE(cond, action) \
  do {                        \
    if (!(cond)) {            \
      printf("%s\n", #cond);  \
      action;                 \
    }                         \
  } while (0)