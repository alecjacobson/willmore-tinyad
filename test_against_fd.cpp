// Checks TinyAD's analytic gradient and Hessian of the one-ring Willmore energy
// against finite differences of the global cotangent-Laplacian formulation, so
// the two independent implementations in willmore.h have to agree.
//
//   ./test_against_fd [n] [h]
//     n  torus resolution around the main ring {5}
//     h  step for the Hessian's mixed central differences {1e-4}
//
// Exits nonzero if either derivative is out of tolerance.
#include "torus.h"
#include "willmore.h"
#include "willmore_tinyad.h"

#include <igl/adjacency_list.h>
#include <Eigen/Core>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using MatrixX3dR = Eigen::Matrix<double,Eigen::Dynamic,3,Eigen::RowMajor>;
using MatrixX3iR = Eigen::Matrix<int,Eigen::Dynamic,3,Eigen::RowMajor>;

// Central differences of willmore_energy. Error is O(eps²) plus roundoff O(ε/eps),
// so eps near ε^(1/3) is the sweet spot for a first derivative.
static MatrixX3dR fd_gradient(
    const MatrixX3dR & V,
    const MatrixX3iR & F,
    const double eps)
{
  MatrixX3dR g(V.rows(),3);
  MatrixX3dR V0 = V;
  for(int i = 0;i<V.rows();i++)
  {
    for(int j = 0;j<3;j++)
    {
      V0(i,j) = V(i,j) + eps;
      const double f1 = willmore_energy(V0,F);
      V0(i,j) = V(i,j) - eps;
      const double f2 = willmore_energy(V0,F);
      V0(i,j) = V(i,j);
      g(i,j) = (f1-f2)/(2*eps);
    }
  }
  return g;
}

// Mixed central differences:
//   H(ij,kl) ≈ [f(x+h·eᵢⱼ+h·e_kl) - f(x+h·eᵢⱼ-h·e_kl)
//             - f(x-h·eᵢⱼ+h·e_kl) + f(x-h·eᵢⱼ-h·e_kl)] / (4h²)
// Dividing by h² lets roundoff in f enter as ~ε/h², so the step wants to be near
// ε^(1/4) rather than the ε^(1/3) that suits a first difference.
static Eigen::MatrixXd fd_hessian(
    const MatrixX3dR & V,
    const MatrixX3iR & F,
    const double h)
{
  Eigen::MatrixXd H(V.size(),V.size());
  MatrixX3dR V0 = V;
  for(int i = 0;i<V.rows();i++)
  {
    for(int j = 0;j<3;j++)
    {
      for(int k = 0;k<V.rows();k++)
      {
        for(int l = 0;l<3;l++)
        {
          // Walk the four corners. When (i,j)==(k,l) the two ±h steps land on
          // the same entry and this collapses to the usual second difference
          // f(x+2h)-2f(x)+f(x-2h) over (2h)², which is what we want there.
          V0(i,j) += h; V0(k,l) += h;
          const double fpp = willmore_energy(V0,F);
          V0(k,l) -= 2*h;
          const double fpm = willmore_energy(V0,F);
          V0(i,j) -= 2*h;
          const double fmm = willmore_energy(V0,F);
          V0(k,l) += 2*h;
          const double fmp = willmore_energy(V0,F);
          V0(i,j) = V(i,j); V0(k,l) = V(k,l);
          H(i*3+j,k*3+l) = (fpp - fpm - fmp + fmm)/(4*h*h);
        }
      }
    }
  }
  return H;
}

int main(int argc, char* argv[])
{
  const int n = argc > 1 ? std::atoi(argv[1]) : 5;
  const double h = argc > 2 ? std::atof(argv[2]) : 1e-4;

  // A Clifford torus, whose smooth Willmore energy is 2*pi^2. The L M^-1 L
  // normalisation used here reports 4x that, so the discrete value approaches
  // 8*pi^2 under refinement rather than matching it at this resolution.
  const double ideal_r = 1.0/std::sqrt(2.0);
  const auto [V,F] = torus(n,(int)std::round(ideal_r*n),ideal_r);
  printf("torus n=%d: %ld vertices, %ld faces (h=%g)\n",n,(long)V.rows(),(long)F.rows(),h);

  std::vector<std::vector<int>> A;
  igl::adjacency_list(F,A,true);

  int failures = 0;
  const auto check = [&](const bool ok, const char * what, const double err, const double tol)
  {
    printf("%s %-44s rel err %9.3e  (tol %.0e)\n",ok?"PASS":"FAIL",what,err,tol);
    if(!ok){ failures++; }
  };

  // 1. The two energy implementations must agree exactly, not just to tolerance.
  const double f_global = willmore_energy(V,F);
  double f_fan = 0;
  for(int i = 0;i<V.rows();i++)
  {
    MatrixX3dR fanV(A[i].size(),3);
    for(int j = 0;j<(int)A[i].size();j++)
    {
      fanV.row(j) = V.row(A[i][j]) - V.row(i);
    }
    f_fan += willmore_contribution(fanV);
  }
  printf("energy: global %.12g  fan %.12g\n",f_global,f_fan);
  check(std::abs(f_global-f_fan) <= 1e-9*std::abs(f_global),
      "fan energy == cotmatrix energy",
      std::abs(f_global-f_fan)/std::abs(f_global), 1e-9);

  // 2. Build the TinyAD objective over the same one-rings.
  constexpr int MIN_SIZE = 4;  // vertex valence 3
  constexpr int END_SIZE = 11; // one past vertex valence 9
  auto func = TinyAD::scalar_function<3>(TinyAD::range(V.rows()));
  if(!willmore_valence_range_ok<MIN_SIZE,END_SIZE>(A)){ return EXIT_FAILURE; }
  func.add_elements_dynamic<TinyAD::element_valence_range_t<MIN_SIZE,END_SIZE>>(
    TinyAD::range((int)A.size()),
    [&A] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
      // Evaluate element using either double or TinyAD::Double
      using T = TINYAD_SCALAR_TYPE(element);
      // n_element counts scalars, so /3 is the element valence and -1 drops the
      // centre vertex. Exact because the valence range is contiguous, so no
      // element is ever padded up to a larger compiled size.
      constexpr int K = fan_rows<std::decay_t<decltype(element)>,T>::value;
      const int v = (int)element.handle;
      const int n_fan = K == Eigen::Dynamic ? (int)A[v].size() : K;
      assert((int)A[v].size() == n_fan);
      const Eigen::Vector3<T> v0 = element.variables(v);
      Eigen::Matrix<T,K,3,Eigen::RowMajor> fanV(n_fan,3);
      for(int j = 0;j<n_fan;j++)
      {
        fanV.row(j) = (element.variables(A[v][j]) - v0).transpose();
      }
      return willmore_contribution(fanV);
    });

  const Eigen::VectorXd x = func.x_from_data(
      [&](const int v){ return V.row(v).transpose(); });
  const auto [f_ad,g_ad,H_ad] = func.eval_with_derivatives(x);

  check(std::abs(f_global-f_ad) <= 1e-9*std::abs(f_global),
      "TinyAD energy == cotmatrix energy",
      std::abs(f_global-f_ad)/std::abs(f_global), 1e-9);

  // 3. Gradient. x packs 3 consecutive entries per vertex, so it reshapes to #V by 3.
  const MatrixX3dR g_fd = fd_gradient(V,F,1e-6);
  const auto g_ad_m = Eigen::Map<const MatrixX3dR>(g_ad.data(),V.rows(),3);
  const double g_err = (g_ad_m - g_fd).cwiseAbs().maxCoeff()/g_fd.cwiseAbs().maxCoeff();
  check(g_err <= 1e-6, "TinyAD gradient == finite differences", g_err, 1e-6);

  // 4. Hessian, same packing: entry (v*3+c) of x is V(v,c).
  const Eigen::MatrixXd H_fd = fd_hessian(V,F,h);
  const Eigen::MatrixXd H_ad_dense = Eigen::MatrixXd(H_ad);
  const double H_err = (H_ad_dense - H_fd).cwiseAbs().maxCoeff()/H_ad_dense.cwiseAbs().maxCoeff();
  check(H_err <= 1e-5, "TinyAD Hessian == finite differences", H_err, 1e-5);

  // 5. A Hessian must be symmetric regardless of what it is compared against.
  const double sym_err =
    (H_ad_dense - H_ad_dense.transpose()).cwiseAbs().maxCoeff()/H_ad_dense.cwiseAbs().maxCoeff();
  check(sym_err <= 1e-12, "TinyAD Hessian is symmetric", sym_err, 1e-12);

  printf("\n%s (%d failure%s)\n",failures?"FAILED":"ALL PASSED",failures,failures==1?"":"s");
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
