#pragma once

// The Willmore energy itself. Deliberately free of any TinyAD dependency: the
// scalar type is a template parameter, so TinyAD::Scalar works here without this
// header knowing about it. The TinyAD glue lives in willmore_tinyad.h.

#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/harmonic.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Sparse>

#include <cassert>

template <typename Scalar, int N>
inline Scalar willmore_contribution(
    const Eigen::Matrix<Scalar,N,3,Eigen::RowMajor> & fanV)
{
  static_assert(N == Eigen::Dynamic || N >= 3, "N must be positive or dynamic");
  // Build effective_N using compile-time N if possible, otherwise use the
  // runtime fanV.rows().
  const int effective_N = N > 0 ? N : fanV.rows();

  assert(fanV.rows() >= 3);
  using RowVector3S = Eigen::Matrix<Scalar,1,3>;
  RowVector3S H(0,0,0);
  //  origin
  //    |  \
  //    |α β\
  //    i----i+1
  Scalar area = 0;
  // i1 trails i by one and wraps, so the fan closes without a modulo — `%` would
  // be a runtime division here even when N is known, since the divisor would come
  // from fanV.rows() rather than from effective_N.
  for(int i = 0, i1 = 1;i < effective_N;i++, i1 = (i1+1 == effective_N ? 0 : i1+1))
  {
    RowVector3S e0 = fanV.row(i);
    RowVector3S e1 = fanV.row(i1);
    // squared norm
    const Scalar l0 =       e0.squaredNorm();
    const Scalar l1 =       e1.squaredNorm();
    const Scalar l01 = (e1-e0).squaredNorm();
    const Scalar double_area = (e0.cross(e1)).norm();
    // The cotangent at a corner is (sum of the two squared edge lengths meeting
    // there - the opposite one) / (4·area), and double_area is only 2·area.
    const Scalar cot_alpha = (l0 + l01 - l1) / (2.0*double_area);
    const Scalar cot_beta  = (l1 + l01 - l0) / (2.0*double_area);
    H += cot_alpha*e1 + cot_beta*e0;
    area += double_area;
  }
  // H is now Σⱼ(cot α + cot β)(Vⱼ-Vᵢ) = 2·(L V)ᵢ. Barycentric mass gives vertex
  // i a third of its incident area, and `area` summed 2·area per triangle, so
  // Mᵢᵢ = area/6. This vertex's share of Σᵢ|(L V)ᵢ|²/Mᵢᵢ is then |H|²/4/Mᵢᵢ.
  const Scalar mass = area/6.0;
  return H.squaredNorm() / 4.0 / mass;
}

inline double willmore_energy(
    const Eigen::Matrix<double,Eigen::Dynamic,3,Eigen::RowMajor> & V,
    const Eigen::Matrix<int,Eigen::Dynamic,3,Eigen::RowMajor> & F)
{
  Eigen::SparseMatrix<double> L,M,B;
  igl::cotmatrix(V,F,L);
  igl::massmatrix(V,F,igl::MASSMATRIX_TYPE_BARYCENTRIC,M);
  igl::harmonic(L,M,2,B);
  return (V.array() * (B*V).eval().array()).sum();
}
