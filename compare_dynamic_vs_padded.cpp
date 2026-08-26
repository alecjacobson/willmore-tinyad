// Compares two ways of registering the one-ring Willmore objective with TinyAD:
//
//   dynamic  add_elements_dynamic<element_valence_range_t<BEGIN,END>>
//            One compiled code path per element valence in [BEGIN,END). Every
//            element hits its exact size, so the fan can be fixed-size.
//
//   padded   add_elements<END-1>
//            A single compiled code path at the largest element valence. Smaller
//            one-rings simply touch fewer variable handles, leaving the trailing
//            slots of TinyAD's index map unused, so every element pays for an
//            (3*(END-1))-square Hessian regardless of its actual valence. The fan
//            is the Dynamic willmore_contribution.
//
// Equivalence is checked before any timing: if the two disagree the timings are
// meaningless, so the program reports the mismatch and exits nonzero.
//
//   ./compare_dynamic_vs_padded [n] [reps]
//     n     torus resolution around the main ring {20}
//     reps  eval_with_derivatives calls per variant {50}
#include "torus.h"
#include "willmore.h"
#include "willmore_tinyad.h"

#include <igl/adjacency_list.h>
#include <Eigen/Core>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

// An element is a vertex plus its one-ring, so a vertex of valence n gives an
// element valence of n+1. Half-open: END_VALENCE_SIZE is one past the largest.
constexpr int BEGIN_VALENCE_SIZE = 4;  // vertex valence 3
constexpr int END_VALENCE_SIZE = 11;   // one past vertex valence 9

using MatrixX3dR = Eigen::Matrix<double,Eigen::Dynamic,3,Eigen::RowMajor>;

int main(int argc, char* argv[])
{
  const int n = argc > 1 ? std::atoi(argv[1]) : 20;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 50;

  const double ideal_r = 1.0/std::sqrt(2.0);
  const auto [V,F] = torus(n,(int)std::round(ideal_r*n),ideal_r);
  std::vector<std::vector<int>> A;
  igl::adjacency_list(F,A,true);
  printf("torus n=%d: %ld vertices, %ld faces, %d reps\n",
      n,(long)V.rows(),(long)F.rows(),reps);
  if(!willmore_valence_range_ok<BEGIN_VALENCE_SIZE,END_VALENCE_SIZE>(A))
  {
    return EXIT_FAILURE;
  }

  // --- variant "dynamic": exact valence per element -------------------------
  auto func_dyn = TinyAD::scalar_function<3>(TinyAD::range(V.rows()));
  func_dyn.add_elements_dynamic<
      TinyAD::element_valence_range_t<BEGIN_VALENCE_SIZE,END_VALENCE_SIZE>>(
    TinyAD::range(V.rows()),
    [&A] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
      using T = TINYAD_SCALAR_TYPE(element);
      // Exact valence per element, so the fan can be fixed-size when it fits.
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

  // --- variant "padded": one static size for every element ------------------
  auto func_pad = TinyAD::scalar_function<3>(TinyAD::range(V.rows()));
  func_pad.add_elements<END_VALENCE_SIZE-1>(TinyAD::range(V.rows()),
    [&A] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
      using T = TINYAD_SCALAR_TYPE(element);
      const int v = (int)element.handle;
      const Eigen::Vector3<T> v0 = element.variables(v);
      // Always the Dynamic (heap) fan: the compiled element valence is
      // END_VALENCE_SIZE-1, which says nothing about this vertex's one-ring.
      Eigen::Matrix<T,Eigen::Dynamic,3,Eigen::RowMajor> fanV(A[v].size(),3);
      for(int j = 0;j<(int)A[v].size();j++)
      {
        fanV.row(j) = (element.variables(A[v][j]) - v0).transpose();
      }
      return willmore_contribution(fanV);
    });

  const Eigen::VectorXd x = func_dyn.x_from_data(
      [&](const int v){ return V.row(v).transpose(); });

  // --- equivalence, before timing anything ----------------------------------
  const auto [f_dyn,g_dyn,H_dyn] = func_dyn.eval_with_derivatives(x);
  const auto [f_pad,g_pad,H_pad] = func_pad.eval_with_derivatives(x);

  const double f_err = std::abs(f_dyn-f_pad)/std::abs(f_dyn);
  const double g_err = (g_dyn-g_pad).cwiseAbs().maxCoeff()/g_dyn.cwiseAbs().maxCoeff();
  const Eigen::MatrixXd H_dyn_d = Eigen::MatrixXd(H_dyn);
  const Eigen::MatrixXd H_pad_d = Eigen::MatrixXd(H_pad);
  const double H_err = (H_dyn_d-H_pad_d).cwiseAbs().maxCoeff()/H_dyn_d.cwiseAbs().maxCoeff();

  printf("\nequivalence (dynamic vs padded)\n");
  printf("  energy    %.14g vs %.14g   rel err %.3e\n",f_dyn,f_pad,f_err);
  printf("  gradient  rel err %.3e\n",g_err);
  printf("  hessian   rel err %.3e  (nnz %d vs %d)\n",
      H_err,(int)H_dyn.nonZeros(),(int)H_pad.nonZeros());
  if(!(f_err <= 1e-12 && g_err <= 1e-12 && H_err <= 1e-12))
  {
    fprintf(stderr,"\nFAILED: variants disagree, timings would be meaningless\n");
    return EXIT_FAILURE;
  }
  printf("  -> identical to 1e-12, timing is meaningful\n");

  // --- timing ---------------------------------------------------------------
  const auto time_variant = [&](auto & func)
  {
    // One untimed call so any lazily built internal state is warm for both.
    { const auto [f0,g0,H0] = func.eval_with_derivatives(x); (void)f0; }
    double acc = 0;
    const auto t0 = std::chrono::high_resolution_clock::now();
    for(int r = 0;r<reps;r++)
    {
      const auto [f,g,H] = func.eval_with_derivatives(x);
      acc += f;
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    (void)acc;
    return std::chrono::duration<double,std::milli>(t1-t0).count()/reps;
  };

  // Interleave so drift in machine state hits both variants alike.
  double ms_dyn = 0, ms_pad = 0;
  const int rounds = 3;
  for(int i = 0;i<rounds;i++)
  {
    ms_dyn += time_variant(func_dyn);
    ms_pad += time_variant(func_pad);
  }
  ms_dyn /= rounds;
  ms_pad /= rounds;

  printf("\neval_with_derivatives, mean of %d rounds x %d reps\n",rounds,reps);
  printf("  dynamic (exact valence)      %8.4f ms/call\n",ms_dyn);
  printf("  padded  (add_elements<%2d>)   %8.4f ms/call\n",END_VALENCE_SIZE-1,ms_pad);
  printf("  padded / dynamic             %8.2fx\n",ms_pad/ms_dyn);

  return EXIT_SUCCESS;
}
