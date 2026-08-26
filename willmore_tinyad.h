#pragma once

// TinyAD glue for the one-ring Willmore energy: everything that has to know
// about TinyAD's element types, kept out of willmore.h so that header stays a
// plain Eigen/libigl dependency.

#include "willmore.h"

#include <TinyAD/ScalarFunction.hh>
#include <Eigen/Core>

#include <cassert>
#include <cstdio>
#include <type_traits>
#include <vector>

// TinyAD first probes each element with a RecorderElement, purely to count how
// many variables it touches. That type carries no n_element (it cannot — the
// valence is what it is measuring), so the one-ring size is known at compile
// time only in the real Element passes.
//
// Even then a fixed-size fan is not always wanted: TinyAD::Scalar carries a
// value, a gradient and a Hessian, so it grows with the square of the element
// valence. A valence-10 one-ring is ~7.5KB per scalar and ~200KB for the whole
// 9x3 fan, well past Eigen's stack-allocation limit (which is a hard error, and
// one that only shows up once plain_array's default constructor is instantiated
// — Release builds can miss it). Fall back to Dynamic, i.e. heap storage, both
// during the probe and whenever the fixed-size fan would not fit.
template <typename E, typename T, typename = void>
struct fan_rows : std::integral_constant<int,Eigen::Dynamic> {};
template <typename E, typename T>
struct fan_rows<E,T,std::void_t<decltype(E::n_element)>>
  : std::integral_constant<int,
      ((std::size_t)(E::n_element/3 - 1)*3*sizeof(T) <= EIGEN_STACK_ALLOCATION_LIMIT)
        ? E::n_element/3 - 1 : Eigen::Dynamic> {};

// Reports the valence range present in A, and whether it is covered by the
// half-open interval of element valences [MinSize,EndSize) that was compiled.
// TinyAD would throw for an element valence at or above EndSize, but without
// naming the vertex; one below MinSize would be quietly padded up to the next
// compiled size, which the fixed-size fan cannot tolerate.
template <int MinSize, int EndSize>
inline bool willmore_valence_range_ok(const std::vector<std::vector<int>> & A)
{
  assert(!A.empty());
  int lo = (int)A[0].size(), hi = (int)A[0].size(), lo_v = 0, hi_v = 0;
  for(int i = 0;i<(int)A.size();i++)
  {
    if((int)A[i].size() < lo){ lo = A[i].size(); lo_v = i; }
    if((int)A[i].size() > hi){ hi = A[i].size(); hi_v = i; }
  }
  printf("valences in [%d,%d] -> element sizes [%d,%d]; compiled for [%d,%d]\n",
      lo,hi,lo+1,hi+1,MinSize,EndSize-1);
  if(hi+1 >= EndSize)
  {
    fprintf(stderr,"Error: vertex %d has valence %d (element size %d) but "
        "the largest compiled size is %d. Raise EndSize.\n",hi_v,hi,hi+1,EndSize-1);
    return false;
  }
  if(lo+1 < MinSize)
  {
    fprintf(stderr,"Error: vertex %d has valence %d (element size %d), below "
        "MinSize %d. It would be padded up to the next compiled size and the "
        "fan would read uninitialized rows. Lower MinSize.\n",lo_v,lo,lo+1,MinSize);
    return false;
  }
  return true;
}
