/**
 * utils.h
 *
 * Project-wide utilities.
 */

#pragma once

#define REQUIRE(cond, action) \
  do {                        \
    if (!(cond)) {            \
      action;                 \
    }                         \
  } while (0)