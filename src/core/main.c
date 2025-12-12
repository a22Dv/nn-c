#include "core/tensor.h"
#include "core/tensor_functions.h"

int main() {
  tnsr_t *tnsr = TNSR_ROWVEC(5);
  tnsr_emap(tnsr, tnsr, rand_uniform, NULL);

  tnsr_t *mul = TNSR_SCALAR();
  tnsr_set(mul, 5);
  tnsr_emul(tnsr, tnsr, mul);

  tnsr_t *tpose = tnsr_transpose(NULL, tnsr);
  tnsr_t *cntrct = tnsr_contract(NULL, tnsr, tpose);

  tnsr_dbgprint(tnsr);
  tnsr_dbgprint(tpose);
  tnsr_dbgprint(cntrct);

  tnsr_destroy(&tnsr);
  tnsr_destroy(&tpose);
  tnsr_destroy(&cntrct); 
}