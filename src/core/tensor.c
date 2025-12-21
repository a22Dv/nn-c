/**
 * tensor.c
 *
 * BRIEF:
 * Implementation for tensor.h
 *
 * NOTE:
 * Assumes TNSR_MAX_RANK is 2.
 */

#include <float.h>
#include <stdbool.h>
#include <stdlib.h>

#include "core/tensor.h"
#include "utils/utils.h"

#define _TNSR_EIMPL(dst, a, op, b)                                                          \
  do {                                                                                      \
    ASSERT(a && b);                                                                         \
                                                                                            \
    tnsr_size_t bstrd[TNSR_MAX_RANK] = {};                                                  \
    REQUIRE(b_broadcast_to_a(a, b, bstrd), goto error);                                     \
                                                                                            \
    tnsr_t *rloc = dst;                                                                     \
    if (!rloc) {                                                                            \
      rloc = tnsr_create(TNSR_SHPE(a, 0), TNSR_SHPE(a, 1));                                 \
      REQUIRE(rloc, goto error);                                                            \
    }                                                                                       \
    ASSERT(TNSR_SHPE(rloc, 0) == TNSR_SHPE(a, 0) && TNSR_SHPE(rloc, 1) == TNSR_SHPE(a, 1)); \
                                                                                            \
    for (tnsr_size_t i = 0; i < TNSR_SHPE(rloc, 0); ++i) {                                  \
      for (tnsr_size_t j = 0; j < TNSR_SHPE(rloc, 1); ++j) {                                \
        TNSR_DATA(rloc, i, j) = TNSR_DATA(a, i, j) op b->data[i * bstrd[0] + j * bstrd[1]]; \
      }                                                                                     \
    }                                                                                       \
    return rloc;                                                                            \
                                                                                            \
  error:                                                                                    \
    return NULL;                                                                            \
  } while (0)

static bool b_broadcast_to_a(const tnsr_t *a, const tnsr_t *b, tnsr_size_t nstrd[TNSR_MAX_RANK]) {
  ASSERT(a && b);
  for (tnsr_size_t i = TNSR_MAX_RANK; i-- > 0;) {
    const bool shp_diff = TNSR_SHPE(a, i) != TNSR_SHPE(b, i);
    const bool bi_eq_one = TNSR_SHPE(b, i) == 1;

    if (shp_diff && bi_eq_one) {
      nstrd[i] = 0;
    } else if (shp_diff) {
      return false;
    } else {
      nstrd[i] = TNSR_STRD(b, i);
    }
  }
  return true;
}

tnsr_t *tnsr_create(tnsr_size_t m, tnsr_size_t n) {
  ASSERT(m > 0 && n > 0);

  tnsr_t *tensor = calloc(1, sizeof(tnsr_t) + sizeof(tnsr_type_t[m * n]));
  REQUIRE(tensor, goto error);

  TNSR_SHPE(tensor, 0) = m;
  TNSR_SHPE(tensor, 1) = n;
  TNSR_STRD(tensor, 0) = n;
  TNSR_STRD(tensor, 1) = 1;

  return tensor;

error:
  return NULL;
}

void tnsr_destroy(tnsr_t **t) {
  if (!t || !*t) {
    return;
  }
  free(*t);
  *t = NULL;
}

void tnsr_set(tnsr_t *t, tnsr_type_t x) {
  ASSERT(t);

#pragma omp parallel for
  for (tnsr_size_t i = 0; i < TNSR_SHPE(t, 0); ++i) {
#pragma omp simd
    for (tnsr_size_t j = 0; j < TNSR_SHPE(t, 1); ++j) {
      TNSR_DATA(t, i, j) = x;
    }
  }
}

void tnsr_reset(tnsr_t *t) {
  ASSERT(t);
  tnsr_set(t, 0);
}

tnsr_t *tnsr_contract(tnsr_t *restrict dst, const tnsr_t *restrict a, const tnsr_t *restrict b) {
  ASSERT(a && b);
  ASSERT(TNSR_SHPE(a, 1) == TNSR_SHPE(b, 0));

  tnsr_t *rloc = dst;

  if (!rloc) {
    rloc = tnsr_create(TNSR_SHPE(a, 0), TNSR_SHPE(b, 1));
    REQUIRE(rloc, goto error);
  }
  ASSERT( // Destination must have compatible shape.
    TNSR_SHPE(rloc, 0) == TNSR_SHPE(a, 0) && 
    TNSR_SHPE(rloc, 1) == TNSR_SHPE(b, 1)
  );

#pragma omp parallel for
  for (tnsr_size_t i = 0; i < TNSR_SHPE(rloc, 0); ++i) {
    for (tnsr_size_t k = 0; k < TNSR_SHPE(a, 1); ++k) {
      tnsr_type_t a_ik = TNSR_DATA(a, i, k);
#pragma omp simd
      for (tnsr_size_t j = 0; j < TNSR_SHPE(rloc, 1); ++j) {
        TNSR_DATA(rloc, i, j) += a_ik * TNSR_DATA(b, k, j);
      }
    }
  }

  return rloc;

error:
  return NULL;
}

tnsr_t *tnsr_eadd(tnsr_t *dst, const tnsr_t *a, const tnsr_t *restrict b) {
  _TNSR_EIMPL(dst, a, +, b);
}

tnsr_t *tnsr_esub(tnsr_t *dst, const tnsr_t *a, const tnsr_t *b) {
  _TNSR_EIMPL(dst, a, -, b);
}

tnsr_t *tnsr_emul(tnsr_t *dst, const tnsr_t *a, const tnsr_t *restrict b) {
  _TNSR_EIMPL(dst, a, *, b);
}

tnsr_t *tnsr_ediv(tnsr_t *dst, const tnsr_t *a, const tnsr_t *b) {
  _TNSR_EIMPL(dst, a, /, b);
}

tnsr_t *tnsr_emap(
    tnsr_t *dst, tnsr_t *a, tnsr_type_t (*f)(tnsr_type_t, void *), void *restrict ctx
) {
  ASSERT(a && f);
  tnsr_t *rloc = dst;
  if (!rloc) {
    rloc = tnsr_create(TNSR_SHPE(a, 0), TNSR_SHPE(a, 1));
    REQUIRE(rloc, goto error);
  }
  ASSERT(TNSR_SHPE(rloc, 0) == TNSR_SHPE(a, 0) && TNSR_SHPE(rloc, 1) == TNSR_SHPE(a, 1));

#pragma omp parallel for
  for (tnsr_size_t i = 0; i < TNSR_SHPE(rloc, 0); ++i) {
    for (tnsr_size_t j = 0; j < TNSR_SHPE(rloc, 1); ++j) {
      TNSR_DATA(rloc, i, j) = f(TNSR_DATA(a, i, j), ctx);
    }
  }
  return rloc;

error:
  return NULL;
}

tnsr_t *tnsr_transpose(tnsr_t *dst, tnsr_t *t) {
  ASSERT(t);
  if (dst == t) {
    tnsr_size_t t = TNSR_SHPE(dst, 0);
    TNSR_SHPE(dst, 0) = TNSR_SHPE(dst, 1);
    TNSR_SHPE(dst, 1) = t;

    t = TNSR_STRD(dst, 0);
    TNSR_STRD(dst, 0) = TNSR_STRD(dst, 1);
    TNSR_STRD(dst, 1) = t;
    return dst;
  }
  tnsr_t *tp = tnsr_create(TNSR_SHPE(t, 1), TNSR_SHPE(t, 0));
  REQUIRE(tp, goto error);

#pragma omp parallel for
  for (tnsr_size_t i = 0; i < TNSR_SHPE(tp, 0); ++i) {
#pragma omp simd
    for (tnsr_size_t j = 0; j < TNSR_SHPE(tp, 1); ++j) {
      TNSR_DATA(tp, i, j) = TNSR_DATA(t, j, i);
    }
  }
  return tp;
error:
  return NULL;
}

tnsr_t *tnsr_sum_over_axis(tnsr_t *restrict dst, tnsr_t *restrict t, tnsr_size_t axis) {
  ASSERT(t && axis >= 0 && axis < TNSR_MAX_RANK);

  const tnsr_size_t m = axis ? TNSR_SHPE(t, 0) : 1;
  const tnsr_size_t n = axis ? 1 : TNSR_SHPE(t, 1);
  tnsr_t *sumloc = dst;
  if (!sumloc) {
    sumloc = tnsr_create(m, n);
    REQUIRE(sumloc, goto error);
  }
  ASSERT(TNSR_SHPE(sumloc, 0) == m && TNSR_SHPE(sumloc, 1) == n);

  const tnsr_size_t p = axis ? TNSR_SHPE(t, 0) : TNSR_SHPE(t, 1);
  const tnsr_size_t q = axis ? TNSR_SHPE(t, 1) : TNSR_SHPE(t, 0);

#pragma omp parallel for
  for (tnsr_size_t i = 0; i < p; ++i) {
    tnsr_type_t si = 0;
    for (tnsr_size_t j = 0; j < q; ++j) {
      si += axis ? TNSR_DATA(t, i, j) : TNSR_DATA(t, j, i);
    }
    if (axis) {
      TNSR_DATA(sumloc, i, 0) = si;
    } else {
      TNSR_DATA(sumloc, 0, i) = si;
    }
  }

  return sumloc;
error:
  return NULL;
}

tnsr_t *tnsr_max_over_axis(tnsr_t *restrict dst, tnsr_t *restrict t, tnsr_size_t axis) {
  ASSERT(t && axis >= 0 && axis < TNSR_MAX_RANK);

  const tnsr_size_t m = axis ? TNSR_SHPE(t, 0) : 1;
  const tnsr_size_t n = axis ? 1 : TNSR_SHPE(t, 1);
  tnsr_t *maxloc = dst;
  if (!maxloc) {
    maxloc = tnsr_create(m, n);
    REQUIRE(maxloc, goto error);
  }
  ASSERT(TNSR_SHPE(maxloc, 0) == m && TNSR_SHPE(maxloc, 1) == n);

  const tnsr_size_t p = axis ? TNSR_SHPE(t, 0) : TNSR_SHPE(t, 1);
  const tnsr_size_t q = axis ? TNSR_SHPE(t, 1) : TNSR_SHPE(t, 0);

#pragma omp parallel for
  for (tnsr_size_t i = 0; i < p; ++i) {
    tnsr_type_t maxi = -FLT_MAX;
    for (tnsr_size_t j = 0; j < q; ++j) {
      const tnsr_type_t val = axis ? TNSR_DATA(t, i, j) : TNSR_DATA(t, j, i);
      maxi = max(maxi, val);
    }
    if (axis) {
      TNSR_DATA(maxloc, i, 0) = maxi;
    } else {
      TNSR_DATA(maxloc, 0, i) = maxi;
    }
  }
  return maxloc;
error:
  return NULL;
}

tnsr_t *tnsr_mean(tnsr_t *dst, tnsr_t *t) {
  ASSERT(t);

  tnsr_t *avg = dst;
  if (!avg) {
    avg = TNSR_SCALAR();
    REQUIRE(avg, goto error);
  }
  ASSERT(TNSR_SHPE(avg, 0) == 1 && TNSR_SHPE(avg, 1) == 1);

  tnsr_type_t sum = 0;
  for (tnsr_size_t i = 0; i < TNSR_SHPE(t, 0); ++i) {
    for (tnsr_size_t j = 0; j < TNSR_SHPE(t, 1); ++j) {
      sum += TNSR_DATA(t, i, j);
    }
  }
  tnsr_set(avg, sum / (TNSR_SHPE(t, 0) * TNSR_SHPE(t, 1)));
  return avg;

error:
  return NULL;
}

void tnsr_dbgprint(const tnsr_t *t) {
  ASSERT(t);
  for (tnsr_size_t i = 0; i < TNSR_SHPE(t, 0); ++i) {
    for (tnsr_size_t j = 0; j < TNSR_SHPE(t, 1); ++j) {
      printf("%+.3f ", TNSR_DATA(t, i, j));
    }
    printf("\n");
  }
}
