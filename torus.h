#pragma once

#include <Eigen/Core>
#include <cmath>
#include <tuple>

// TORUS Construct a triangle mesh of a unit torus.
//
// [V,F] = torus(n,m,r)
// [V,F] = torus(n,m,r,'ParameterName',ParameterValue, ...)
//
// Inputs:
//   n  number of vertices around inner ring {40}
//   m  number of vertices around outer ring {round(0.4*n)}
//   r  radius of the inner ring {0.4}
//   Optional:
//     'R'  followed by outer ring radius {1}
// Outputs:
//   V  #V by 3 list of mesh vertex positions
//   F  #F by 3 list of triangle mesh indices
//
// Example:
//   % Roughly even shaped triangles
//   n = 40;
//   r = 0.4;
//   [V,F] = torus(n,round(r*n),r);
inline std::tuple<
  Eigen::Matrix<double,Eigen::Dynamic,3,Eigen::RowMajor>,
  Eigen::Matrix<int,Eigen::Dynamic,3,Eigen::RowMajor>
>
  torus(const int n = 40, const int m = -1, const double r = 0.4, const double R = 1.0)
{
  // n samples the angle θ swept about the z-axis (the ring of radius R) and m
  // samples the angle φ swept about the tube (the ring of radius r); that's the
  // pairing the example above assumes, since m = round(r*n) makes the sample
  // spacing 2πR/n and 2πr/m match.
  const int mm = m < 0 ? (int)std::round(r * n) : m;
  constexpr double pi = 3.14159265358979323846;

  Eigen::Matrix<double,Eigen::Dynamic,3,Eigen::RowMajor> V(n*mm,3);
  Eigen::Matrix<int,Eigen::Dynamic,3,Eigen::RowMajor> F(2*n*mm,3);
  for(int i = 0;i < n;i++)
  {
    const double th = 2.0*pi*i/n;
    const double cos_th = std::cos(th);
    const double sin_th = std::sin(th);
    // wrap around: the last ring stitches back onto the first
    const int i1 = (i+1)%n;
    for(int j = 0;j < mm;j++)
    {
      const double ph = 2.0*pi*j/mm;
      const int j1 = (j+1)%mm;

      V.row(i*mm+j) <<
        (R + r*std::cos(ph))*cos_th,
        (R + r*std::cos(ph))*sin_th,
        r*std::sin(ph);

      // Split the quad (i,j),(i+1,j),(i+1,j+1),(i,j+1) along its diagonal.
      // Winding ∂θ then ∂φ makes ∂θ×∂φ, which points away from the tube's
      // core, so both triangles come out facing outward.
      F.row(2*(i*mm+j)+0) << i*mm+j, i1*mm+j, i1*mm+j1;
      F.row(2*(i*mm+j)+1) << i*mm+j, i1*mm+j1, i*mm+j1;
    }
  }
  return {V,F};
}
