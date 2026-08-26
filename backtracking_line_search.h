#pragma once
#include <Eigen/Core>
#include <cassert>

// Backtracking line search satisfying the Armijo sufficient-decrease condition
//
//     f(x0 + t·dx) ≤ f(x0) + α·t·∇f(x0)ᵀdx
//
// starting from the incoming step size t and shrinking by β until it holds.
//
// Inputs:
//   f        callable, f(x) -> scalar
//   x0       #x starting point
//   dfdx0    #x gradient of f at x0
//   dx       #x search direction; must be a descent direction, ∇f(x0)ᵀdx < 0
//   alpha    sufficient-decrease constant in (0,½), e.g. 1e-4
//   beta     step shrink factor in (0,1), e.g. 0.8
//   max_iter maximum number of shrinks before giving up
//   t        initial step size, e.g. 1
// Outputs:
//   t        accepted step size, or 0 if none was found
//   x        x0 + t·dx, or x0 if no step was accepted
//   fx       f(x)
//
// Note the functor and the returned value cannot both be called `f`, so the
// value is `fx` here.
template <
  typename Func,
  typename Derivedx0,
  typename Deriveddfdx0,
  typename Deriveddx,
  typename Derivedx>
void backtracking_line_search(
  Func & f,
  const Eigen::MatrixBase<Derivedx0> & x0,
  const Eigen::MatrixBase<Deriveddfdx0> & dfdx0,
  const Eigen::MatrixBase<Deriveddx> & dx,
  const typename Derivedx::Scalar alpha,
  const typename Derivedx::Scalar beta,
  const int max_iter,
  typename Derivedx::Scalar & t,
  Eigen::PlainObjectBase<Derivedx> & x,
  typename Derivedx::Scalar & fx)
{
  using Scalar = typename Derivedx::Scalar;
  assert(x0.size() == dfdx0.size());
  assert(x0.size() == dx.size());
  assert(alpha > 0 && alpha < 0.5);
  assert(beta > 0 && beta < 1);
  assert(t > 0);

  const Scalar f0 = f(x0);
  // Directional derivative along dx. Non-negative means dx does not descend, so
  // no step size can satisfy Armijo and shrinking would just burn evaluations.
  const Scalar slope = dfdx0.dot(dx);
  if(!(slope < 0))
  {
    t = 0;
    x = x0;
    fx = f0;
    return;
  }

  for(int iter = 0;iter < max_iter;iter++)
  {
    x = x0 + t*dx;
    fx = f(x);
    // Reject NaN/Inf the same as insufficient decrease: the comparison below is
    // false for NaN, so this only needs to keep it from being accepted.
    if(fx <= f0 + alpha*t*slope)
    {
      return;
    }
    t *= beta;
  }

  // Nothing satisfied the condition; report no progress rather than a bad step.
  t = 0;
  x = x0;
  fx = f0;
}
