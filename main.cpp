#include "torus.h"
#include "willmore.h"
#include "backtracking_line_search.h"
#include "area_barrier.h"
#include "backtracking_line_search.h"

#include <igl/opengl/glfw/Viewer.h>
#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>
#include <igl/is_border_vertex.h>
#include <igl/adjacency_list.h>
#include <igl/matlab_format.h>
#include <TinyAD/ScalarFunction.hh>
#include <TinyAD/Utils/LinearSolver.hh>
#include <TinyAD/Utils/LineSearch.hh>
#include <TinyAD/Utils/NewtonDecrement.hh>
#include <TinyAD/Utils/NewtonDirection.hh>
#include <TinyAD/Utils/Helpers.hh>
#include <TinyAD/Utils/NewtonDirection.hh>
#include <TinyAD/Utils/LineSearch.hh>
#include <Eigen/Core>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>
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

template <typename E, typename = void>
struct has_n_element : std::false_type {};
template <typename E>
struct has_n_element<E, std::void_t<decltype(E::n_element)>> : std::true_type {};

struct Options
{
  std::string input;                    // -i, empty means generate a torus
  int torus_resolution = 5;             // -n
  int solver_iterations = 0;            // -s, 0 means evaluate only
  enum Method { GRADIENT_DESCENT, PROJECTED_NEWTON };
  Method method = PROJECTED_NEWTON;     // -m
  enum LineSearch { TINYAD, NATIVE };
  LineSearch line_search = TINYAD;      // -l
  double alpha = 1e-4;                  // -a, Armijo sufficient-decrease constant
  double beta = 0.8;                    // -b, step shrink factor
  int max_shrinks = 64;                 // -k, shrinks before the search gives up
  double w_identity = 1e-9;             // -w, Levenberg regulariser on H_proj
  double barrier_stiffness = 0.0;       // -B, 0 disables the area barrier
  double barrier_threshold = -1.0;      // -d, <0 means a fraction of the initial min area
  double quality_threshold = -1.0;      // -q, >0 switches to a scale-invariant barrier
  bool fix_area = false;                // -F, rescale to the initial total area each step
};

static void usage(const char * exe)
{
  fprintf(stderr,
    "Usage: %s [options]\n"
    "  -i <path>     input triangle mesh; default is a generated torus\n"
    "  -n <int>      torus resolution, used when -i is not given (default 5)\n"
    "  -s <int>      solver iterations (default 0, i.e. evaluate only)\n"
    "  -m <method>   gradient-descent | projected-newton (default projected-newton)\n"
    "  -l <search>   tinyad | native line search (default tinyad)\n"
    "  -a <float>    Armijo sufficient-decrease constant, in (0,0.5) (default 1e-4)\n"
    "  -b <float>    line search step shrink factor, in (0,1) (default 0.8)\n"
    "  -k <int>      max shrinks before the line search gives up (default 64)\n"
    "  -w <float>    identity regulariser added to H_proj, >= 0 (default 1e-9)\n"
    "  -B <float>    IPC-style area barrier stiffness, 0 disables it (default 0)\n"
    "  -d <float>    barrier activation area (default 0.5x the initial min area)\n"
    "  -q <float>    barrier on scale-invariant triangle quality instead of area,\n"
    "                activating below this fraction of equilateral, in (0,1)\n"
    "  -F            rescale to the initial total area after each step; free,\n"
    "                since the Willmore energy is exactly scale invariant\n",
    exe);
}

static bool parse_args(const int argc, char * argv[], Options & o)
{
  for(int i = 1;i<argc;i++)
  {
    // Consumes the next argument as this flag's value, or reports it missing.
    const auto value = [&](const char * flag) -> const char *
    {
      if(i+1 >= argc){ fprintf(stderr,"Error: %s needs a value\n",flag); return nullptr; }
      return argv[++i];
    };
    if(!std::strcmp(argv[i],"-i"))
    {
      const char * v = value("-i"); if(!v){ return false; }
      o.input = v;
    }
    else if(!std::strcmp(argv[i],"-n"))
    {
      const char * v = value("-n"); if(!v){ return false; }
      o.torus_resolution = std::atoi(v);
      if(o.torus_resolution < 3)
      {
        fprintf(stderr,"Error: -n must be at least 3 (got '%s')\n",v);
        return false;
      }
    }
    else if(!std::strcmp(argv[i],"-s"))
    {
      const char * v = value("-s"); if(!v){ return false; }
      o.solver_iterations = std::atoi(v);
      if(o.solver_iterations < 0)
      {
        fprintf(stderr,"Error: -s must not be negative (got '%s')\n",v);
        return false;
      }
    }
    else if(!std::strcmp(argv[i],"-m"))
    {
      const char * v = value("-m"); if(!v){ return false; }
      if(!std::strcmp(v,"gradient-descent")){ o.method = Options::GRADIENT_DESCENT; }
      else if(!std::strcmp(v,"projected-newton")){ o.method = Options::PROJECTED_NEWTON; }
      else
      {
        fprintf(stderr,"Error: -m must be gradient-descent or projected-newton, got '%s'\n",v);
        return false;
      }
    }
    else if(!std::strcmp(argv[i],"-l"))
    {
      const char * v = value("-l"); if(!v){ return false; }
      if(!std::strcmp(v,"tinyad")){ o.line_search = Options::TINYAD; }
      else if(!std::strcmp(v,"native")){ o.line_search = Options::NATIVE; }
      else
      {
        fprintf(stderr,"Error: -l must be tinyad or native, got '%s'\n",v);
        return false;
      }
    }
    else if(!std::strcmp(argv[i],"-a"))
    {
      const char * v = value("-a"); if(!v){ return false; }
      o.alpha = std::atof(v);
      if(!(o.alpha > 0 && o.alpha < 0.5))
      {
        fprintf(stderr,"Error: -a must be in (0,0.5), got '%s'\n",v);
        return false;
      }
    }
    else if(!std::strcmp(argv[i],"-b"))
    {
      const char * v = value("-b"); if(!v){ return false; }
      o.beta = std::atof(v);
      if(!(o.beta > 0 && o.beta < 1))
      {
        fprintf(stderr,"Error: -b must be in (0,1), got '%s'\n",v);
        return false;
      }
    }
    else if(!std::strcmp(argv[i],"-k"))
    {
      const char * v = value("-k"); if(!v){ return false; }
      o.max_shrinks = std::atoi(v);
      if(o.max_shrinks < 1)
      {
        fprintf(stderr,"Error: -k must be at least 1, got '%s'\n",v);
        return false;
      }
    }
    else if(!std::strcmp(argv[i],"-w"))
    {
      const char * v = value("-w"); if(!v){ return false; }
      o.w_identity = std::atof(v);
      if(!(o.w_identity >= 0))
      {
        fprintf(stderr,"Error: -w must be non-negative, got '%s'\n",v);
        return false;
      }
    }
    else if(!std::strcmp(argv[i],"-B"))
    {
      const char * v = value("-B"); if(!v){ return false; }
      o.barrier_stiffness = std::atof(v);
      if(!(o.barrier_stiffness >= 0))
      {
        fprintf(stderr,"Error: -B must be non-negative, got '%s'\n",v);
        return false;
      }
    }
    else if(!std::strcmp(argv[i],"-d"))
    {
      const char * v = value("-d"); if(!v){ return false; }
      o.barrier_threshold = std::atof(v);
      if(!(o.barrier_threshold > 0))
      {
        fprintf(stderr,"Error: -d must be positive, got '%s'\n",v);
        return false;
      }
    }
    else if(!std::strcmp(argv[i],"-q"))
    {
      const char * v = value("-q"); if(!v){ return false; }
      o.quality_threshold = std::atof(v);
      if(!(o.quality_threshold > 0 && o.quality_threshold < 1))
      {
        fprintf(stderr,"Error: -q must be in (0,1), got '%s'\n",v);
        return false;
      }
    }
    else if(!std::strcmp(argv[i],"-F"))
    {
      o.fix_area = true;
    }
    else if(!std::strcmp(argv[i],"-h") || !std::strcmp(argv[i],"--help"))
    {
      usage(argv[0]);
      return false;
    }
    else
    {
      fprintf(stderr,"Error: unknown argument '%s'\n",argv[i]);
      usage(argv[0]);
      return false;
    }
  }
  return true;
}

int main(int argc, char* argv[])
{
  Options opts;
  if(!parse_args(argc,argv,opts)){ return EXIT_FAILURE; }

  Eigen::Matrix<double,Eigen::Dynamic,3,Eigen::RowMajor> V;
  Eigen::Matrix<int,Eigen::Dynamic,3,Eigen::RowMajor> F;

  // If filename not given, fall back to a Clifford torus.
  if(!opts.input.empty())
  {
    if(!igl::read_triangle_mesh(opts.input,V,F))
    {
      fprintf(stderr,"Error: could not read mesh from '%s'\n",opts.input.c_str());
      return EXIT_FAILURE;
    }
    printf("read %s: %ld vertices, %ld faces\n",
        opts.input.c_str(),(long)V.rows(),(long)F.rows());
  }
  else
  {
    const int n = opts.torus_resolution;
    const double ideal_r = 1.0/std::sqrt(2.0);
    const int m = (int)std::round(ideal_r*n);
    std::tie(V,F) = torus(n,m,ideal_r);
    printf("torus n=%d: %ld vertices, %ld faces\n",n,(long)V.rows(),(long)F.rows());
  }

  // The fan wraps from the last neighbour back to the first, so it only means
  // anything on a closed surface: at a boundary vertex the one-ring is an open
  // path and the wrap invents a triangle that is not in the mesh.
  {
    const std::vector<bool> is_border = igl::is_border_vertex(F);
    const int num_border = (int)std::count(is_border.begin(),is_border.end(),true);
    if(num_border > 0)
    {
      fprintf(stderr,"Error: mesh has %d boundary vertices; the one-ring Willmore "
          "term assumes a closed surface.\n",num_border);
      return EXIT_FAILURE;
    }
  }

  igl::write_triangle_mesh("input.obj",V,F);

  printf("Willmore energy = %g\n",willmore_energy(V,F));
  std::vector<std::vector<int>> A;
  igl::adjacency_list(F,A,true);
  // Cumulative counts of the number of vertices in each fan, so C(p+1)-C(p) is
  // the
  Eigen::VectorXi C(V.rows()+1);
  C(0) = 0;
  for(int i = 0;i<V.rows();i++) { C(i+1) = C(i) + (int)A[i].size(); }
  Eigen::VectorXi I(C(V.rows()));
  for(int i = 0;i<V.rows();i++)
  {
    for(int j = 0;j<A[i].size();j++)
    {
      I(C(i)+j) = A[i][j];
    }
  }

  auto func = TinyAD::scalar_function<3>(TinyAD::range(V.rows()));
  constexpr int BEGIN_VALID_SIZE = 4;  // vertex valence 3 + 1
  constexpr int END_VALID_SIZE = 12;   // one past max vertex valence 9 + 1
  {
    Eigen::VectorXi element_valences = (C.tail(C.size()-1) - C.head(C.size()-1)).array() + 1;
    const int min_element_valence = element_valences.minCoeff();
    const int max_element_valence = element_valences.maxCoeff();
    if(min_element_valence < BEGIN_VALID_SIZE || max_element_valence >= END_VALID_SIZE)
    {
      fprintf(stderr,"Error: element valences [%d,%d] outside compiled range [%d,%d)\n",
          min_element_valence,max_element_valence,BEGIN_VALID_SIZE,END_VALID_SIZE);
      return 1;
    }
  }
  

  func.template add_elements_dynamic<
    TinyAD::element_valence_range_t<BEGIN_VALID_SIZE,END_VALID_SIZE>
    >(
    TinyAD::range((int)V.rows()),
    [&I,&C] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
      // Evaluate element using either double or TinyAD::Double
      using T = TINYAD_SCALAR_TYPE(element);
      // n_element counts scalars, so /3 is the element valence and -1 drops the
      // centre vertex. Exact because the valence range is contiguous, so no
      // element is ever padded up to a larger compiled size.
      constexpr int K = fan_rows<std::decay_t<decltype(element)>,T>::value;
      const int v = (int)element.handle;
      const int element_valence = C(v+1) - C(v);

      //if constexpr(has_n_element<std::decay_t<decltype(element)>>::value)
      //{
      //  printf("element %d: n_element=%d, K=%d, element_valence: %d\n",
      //      (int)element.handle,(int)element.n_element,K, element_valence);
      //}
      const int n_fan = K == Eigen::Dynamic ? (int)element_valence : K;
      assert(element_valence == n_fan);
      const Eigen::Vector3<T> v0 = element.variables(v);
      Eigen::Matrix<T,K,3,Eigen::RowMajor> fanV(n_fan,3);
      for(int j = 0;j<n_fan;j++)
      {
        const int vj = I(C(v)+j);
        fanV.row(j) = (element.variables(vj) - v0).transpose();
      }
      return willmore_contribution(fanV);
    });

  // Smallest triangle area of the input, both to seed the barrier threshold and
  // to report against later.
  const auto min_triangle_area = [&](const Eigen::MatrixXd & Vq)
  {
    double m = std::numeric_limits<double>::max();
    for(int fi = 0;fi<F.rows();fi++)
    {
      const Eigen::RowVector3d a = Vq.row(F(fi,0));
      const Eigen::RowVector3d b = Vq.row(F(fi,1));
      const Eigen::RowVector3d c = Vq.row(F(fi,2));
      m = std::min(m,0.5*(b-a).cross(c-a).norm());
    }
    return m;
  };
  const double initial_min_area = min_triangle_area(V);
  double initial_total_area = 0;
  for(int fi = 0;fi<F.rows();fi++)
  {
    const Eigen::RowVector3d a = V.row(F(fi,0));
    const Eigen::RowVector3d b = V.row(F(fi,1));
    const Eigen::RowVector3d c = V.row(F(fi,2));
    initial_total_area += 0.5*(b-a).cross(c-a).norm();
  }

  // A second element set, one per face. TinyAD sums the terms, so the barrier
  // simply adds to the per-vertex Willmore elements already registered.
  const double a_hat = opts.barrier_threshold > 0 ?
    opts.barrier_threshold : 0.5*initial_min_area;
  if(opts.barrier_stiffness > 0)
  {
    printf("area barrier: stiffness %g, activates below %g (initial min area %g)\n",
        opts.barrier_stiffness,a_hat,initial_min_area);
    const double kappa = opts.barrier_stiffness;
    const bool use_quality = opts.quality_threshold > 0;
    const double q_hat = opts.quality_threshold;
    func.add_elements<3>(TinyAD::range(F.rows()),
      [&F,kappa,a_hat,use_quality,q_hat] (auto& element) -> TINYAD_SCALAR_TYPE(element)
      {
        using T = TINYAD_SCALAR_TYPE(element);
        const int fi = (int)element.handle;
        const Eigen::Vector3<T> a = element.variables(F(fi,0));
        const Eigen::Vector3<T> b = element.variables(F(fi,1));
        const Eigen::Vector3<T> c = element.variables(F(fi,2));
        const T area = 0.5*(b-a).cross(c-a).norm();
        if(use_quality)
        {
          // A/(l0²+l1²+l2²) is dimensionless, so unlike an absolute area it
          // cannot be satisfied by simply growing the mesh. Normalised so that
          // an equilateral triangle scores 1.
          const T sum_sq = (b-a).squaredNorm()+(c-b).squaredNorm()+(a-c).squaredNorm();
          const T q = area/sum_sq/(std::sqrt(3.0)/12.0);
          return kappa*area_barrier(q,q_hat);
        }
        return kappa*area_barrier(area,a_hat);
      });
  }

  Eigen::VectorXd x = func.x_from_data(
      [&](const int v){ return V.row(v).transpose(); });
  {
    const auto [f0,g0] = func.eval_with_gradient(x);
    printf("Willmore energy (TinyAD) = %.10g   |g| = %.3e\n",f0,g0.norm());
  }

  // Willmore energy is invariant to rigid motion and to scale, so H_proj is
  // singular along those directions and a bare LDLT solve can fail. A small
  // multiple of the identity keeps the factorisation well behaved without
  // meaningfully bending the Newton direction.
  // Zero-pad the iteration index so the meshes sort in the order they were run.
  const int index_width =
    (int)std::to_string(std::max(1,opts.solver_iterations-1)).size();
  TinyAD::LinearSolver<double> solver;
  for(int it = 0;it<opts.solver_iterations;it++)
  {
    double f = 0;
    Eigen::VectorXd g, d;
    if(opts.method == Options::PROJECTED_NEWTON)
    {
      const auto [fi,gi,H_proj] = func.eval_with_hessian_proj(x);
      f = fi; g = gi;
      d = TinyAD::newton_direction(g,H_proj,solver,opts.w_identity);
    }
    else
    {
      const auto [fi,gi] = func.eval_with_gradient(x);
      f = fi; g = gi;
      d = -g;
    }
    const double decrement = TinyAD::newton_decrement(d,g);
    printf("iter %*d  f = %.10g  |g| = %.3e  decrement = %.3e",
        index_width,it,f,g.norm(),decrement);
    if(decrement < 1e-12)
    {
      printf("\nconverged after %d iteration%s\n",it,it==1?"":"s");
      break;
    }
    Eigen::VectorXd x0 = x;
    double t = 0;
    if(opts.line_search == Options::TINYAD)
    {
      const Eigen::VectorXd x_new =
        TinyAD::line_search(x,d,f,g,func,1.0,opts.beta,opts.max_shrinks,opts.alpha);
      // TinyAD returns x0 unchanged when it cannot satisfy Armijo; recover the
      // accepted step size by projecting the move back onto the direction.
      t = d.squaredNorm() > 0 ? (x_new-x).dot(d)/d.squaredNorm() : 0.0;
      x = x_new;
    }
    else
    {
      auto energy = [&](const Eigen::VectorXd & z){ return func.eval(z); };
      t = 1.0;
      double fx = f;
      Eigen::VectorXd x_new;
      backtracking_line_search(
          energy,x,g,d,opts.alpha,opts.beta,opts.max_shrinks,t,x_new,fx);
      x = x_new;
    }
    if(t <= 0)
    {
      printf("\nline search found no admissible step after %d shrinks; stopping\n",
          opts.max_shrinks);
      break;
    }

    // Smallest triangle area, to tell "converged" apart from "the mesh is
    // collapsing": willmore_contribution divides by double_area and by the
    // barycentric mass, so vanishing area sends f and its derivatives to
    // infinity and the line search then has nothing admissible to find.
    func.x_to_data(x,[&](const Eigen::Index v, const Eigen::Vector3d & p)
        { V.row(v) = p.transpose(); });
    double min_area = std::numeric_limits<double>::max();
    for(int fi = 0;fi<F.rows();fi++)
    {
      const Eigen::RowVector3d a = V.row(F(fi,0));
      const Eigen::RowVector3d b = V.row(F(fi,1));
      const Eigen::RowVector3d c = V.row(F(fi,2));
      min_area = std::min(min_area,0.5*(b-a).cross(c-a).norm());
    }
    // Gauge-fix the scale. The Willmore term is exactly invariant under V -> sV,
    // so this leaves it untouched and only removes the null direction the
    // optimiser was exploiting to inflate past an absolute area threshold.
    if(opts.fix_area)
    {
      double area_now = 0;
      for(int fi = 0;fi<F.rows();fi++)
      {
        const Eigen::RowVector3d a = V.row(F(fi,0));
        const Eigen::RowVector3d b = V.row(F(fi,1));
        const Eigen::RowVector3d c = V.row(F(fi,2));
        area_now += 0.5*(b-a).cross(c-a).norm();
      }
      if(area_now > 0)
      {
        const Eigen::RowVector3d centroid = V.colwise().mean();
        V = ((V.rowwise()-centroid)*std::sqrt(initial_total_area/area_now)).rowwise()
            + centroid;
        x = func.x_from_data([&](const int v){ return V.row(v).transpose(); });
      }
    }

    // Smallest interior angle, to catch the degeneracy that area alone misses:
    // a sliver can keep its area while collapsing an angle.
    double min_angle = 180.0;
    double total_area = 0;
    for(int fi = 0;fi<F.rows();fi++)
    {
      const Eigen::RowVector3d p0 = V.row(F(fi,0));
      const Eigen::RowVector3d p1 = V.row(F(fi,1));
      const Eigen::RowVector3d p2 = V.row(F(fi,2));
      total_area += 0.5*(p1-p0).cross(p2-p0).norm();
      const Eigen::RowVector3d e[3] = {p1-p0,p2-p1,p0-p2};
      for(int c = 0;c<3;c++)
      {
        const Eigen::RowVector3d u = -e[(c+2)%3], v = e[c];
        const double cs = u.normalized().dot(v.normalized());
        min_angle = std::min(min_angle,
            std::acos(std::max(-1.0,std::min(1.0,cs)))*180.0/M_PI);
      }
    }
    printf("  t = %.3e  min_area = %.3e  min_angle = %.3f  area = %.4f  willmore = %.10g\n",
        t,min_area,min_angle,total_area,willmore_energy(V,F));

    func.x_to_data(x,[&](const Eigen::Index v, const Eigen::Vector3d & p)
        { V.row(v) = p.transpose(); });
    //char filename[64];
    //snprintf(filename,sizeof(filename),"output_iter_%0*d.obj",index_width,it);
    //igl::write_triangle_mesh(filename,V,F);
  }

  if(opts.solver_iterations > 0)
  {
    const auto [f_end,g_end] = func.eval_with_gradient(x);
    printf("final energy = %.10g   |g| = %.3e\n",f_end,g_end.norm());
    igl::write_triangle_mesh("output.obj",V,F);
  }

  return 0;
}
