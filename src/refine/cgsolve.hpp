// Author: "Keitaro Yamashita, Garib N. Murshudov"
// MRC Laboratory of Molecular Biology

#ifndef SERVALCAT_REFINE_CGSOLVE_HPP_
#define SERVALCAT_REFINE_CGSOLVE_HPP_

#include "ll.hpp"
#include "geom.hpp"
#include <ostream>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

namespace servalcat {

struct CgSolve {
  const GeomTarget *geom;
  const LL *ll;
  double gamma = 0.;
  double toler = 1.e-4;
  int ncycle = 2000;
  int max_gamma_cyc = 500;
  CgSolve(const GeomTarget *geom, const LL *ll)
    : geom(geom), ll(ll) {}

  template<typename Preconditioner = Eigen::DiagonalPreconditioner<double>>
  Eigen::VectorXd solve(double weight, std::ostream* logger) {
    Eigen::VectorXd vn = Eigen::VectorXd::Map(geom->vn.data(), geom->vn.size());
    Eigen::SparseMatrix<double> am = geom->make_spmat();
    if (logger)
      *logger << "diag(geom) min= " << am.diagonal().minCoeff()
              << " max= " <<  am.diagonal().maxCoeff() << "\n";
    if (ll != nullptr) {
      auto ll_mat = ll->make_spmat();
      if (logger)
        *logger << "diag(data) min= " << ll_mat.diagonal().minCoeff()
                << " max= " <<  ll_mat.diagonal().maxCoeff() << "\n";
      vn += Eigen::VectorXd::Map(ll->vn.data(), ll->vn.size()) * weight;
      am += ll_mat * weight;
    }
    if (logger)
      *logger << "diag(all) min= " << am.diagonal().minCoeff()
              << " max= " <<  am.diagonal().maxCoeff() << "\n";
    const int n = am.cols();
    Preconditioner precond;
    precond.compute(am);

    double gamma_save = 0;
    bool gamma_flag = false;
    bool conver_flag = false;
    Eigen::VectorXd dv(n), dv_save(n);
    dv.setZero();
    dv_save.setZero();

    const double vnorm2 = vn.squaredNorm();
    const double test_lim = std::max(toler * toler * vnorm2, std::numeric_limits<double>::min());
    double step = 0.05;

    for (int gamma_cyc = 0; gamma_cyc < max_gamma_cyc; ++gamma_cyc, gamma+=step) {
      if (logger)
        *logger << "Trying gamma equal " << gamma << "\n";
      Eigen::VectorXd r = vn - (am * dv + gamma * dv);
      double rnorm2 = r.squaredNorm();
      if (rnorm2 < test_lim)
        break;

      Eigen::VectorXd p(n), z(n), tmp(n);
      p = precond.solve(r);
      double rho0 = r.dot(p);
      bool exit_flag = false;
      for (int itr = 0; itr < ncycle; ++itr) {
        tmp.noalias() = am * p + gamma * p;
        double alpha = rho0 / p.dot(tmp);
        dv += alpha * p;
        r -= alpha * tmp;
        rnorm2 = r.squaredNorm();
        if (rnorm2 < test_lim) {
          if (!gamma_flag) {
            if (logger)
              *logger << "Convergence reached with no gamma cycles\n";
            exit_flag = true;
            break;
          } else if (conver_flag) {
            if (logger)
              *logger << "Convergence reached with gamma equal " << gamma << "\n";
            step *= 1.01;
            exit_flag = true;
            break;
          } else {
            conver_flag = true;
            gamma_save = gamma;
            dv_save = dv;
            gamma = std::max(0., gamma - step/5.);
            step = std::max(step/1.1, 0.0001);
            if (logger)
              *logger << "Gamma decreased to " << gamma << "\n";
            exit_flag = true;
            break;
          }
        }

        z = precond.solve(r);
        double rho1 = rho0;
        rho0 = r.dot(z);
        if (rho0 > 4 * rho1) {
          if (logger)
            *logger << "Not converging with gamma equal " << gamma << "\n";
          step *= 1.05;
          break;
        }
        double beta = rho0 / rho1;
        p = z + beta * p;
      }
      if (exit_flag) break;
      gamma_flag = true;
      if (!conver_flag)
        dv.setZero();
      else {
        dv = dv_save;
        gamma = gamma_save;
        if (logger)
          *logger << "Back to gamma equal " << gamma << "\n";
      }
    }
    return dv;
  }
};

} // namespace servalcat
#endif
