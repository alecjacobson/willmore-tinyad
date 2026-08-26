#pragma once

// IPC-style log barrier (Li et al. 2020, "Incremental Potential Contact"),
// applied to triangle area rather than to a contact distance.
//
//     b(a) = -(a - â)² · ln(a/â)   for 0 < a < â
//            0                     for a ≥ â
//
// It is zero and C² at the activation threshold â, positive below it, and grows
// without bound as a → 0, so a finite-energy configuration cannot contain a
// collapsed triangle. Deliberately free of any TinyAD dependency: the scalar is
// a template parameter, so TinyAD::Scalar differentiates it exactly.
//
// Note the units: (a - â)² is an area squared, so the stiffness that multiplies
// this term carries 1/area² if the total is to stay commensurate with a
// dimensionless energy such as ∫H²dA.

#include <cmath>
#include <limits>

template <typename Scalar>
inline Scalar area_barrier(const Scalar & a, const double a_hat)
{
  // Pull in std::log for the passive (double) case; for an AD scalar the
  // overload is found by argument-dependent lookup.
  using std::log;

  if(a >= a_hat)
  {
    return Scalar(0.0);
  }
  if(a <= 0.0)
  {
    // Past the barrier: an inverted or fully collapsed triangle. Infinity keeps
    // Armijo from ever accepting the step, and unlike NaN it survives TinyAD's
    // line-search assertion that the energy equals itself.
    return Scalar(std::numeric_limits<double>::infinity());
  }
  const Scalar d = a - a_hat;
  return -d*d*log(a/a_hat);
}
