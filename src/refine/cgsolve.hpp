// Author: "Keitaro Yamashita, Garib N. Murshudov"
// MRC Laboratory of Molecular Biology

#ifndef SERVALCAT_REFINE_CGSOLVE_HPP_
#define SERVALCAT_REFINE_CGSOLVE_HPP_

#include "ll.hpp"
#include "geom.hpp"
#include <ostream>
#include <gemmi/logger.hpp> // for gemmi::Logger
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

namespace servalcat {

Eigen::SparseMatrix<double>
diagonal_preconditioner(Eigen::SparseMatrix<double> &mat) {
  const int n = mat.cols();
  Eigen::SparseMatrix<double> pmat(n, n);
  std::vector<Eigen::Triplet<double>> data;
  for(int j = 0; j < mat.outerSize(); ++j) {
    Eigen::SparseMatrix<double>::InnerIterator it(mat, j);
    while(it && it.index() != j) ++it;
    if(it && it.index() == j && it.value() > 0)
      data.emplace_back(j, j, std::sqrt(1. / it.value()));
    else
      data.emplace_back(j, j, 1);
  }
  pmat.setFromTriplets(data.begin(), data.end());
  mat = (pmat * mat * pmat).eval();
  // in our case if diagonal is zero, all corresponding non-diagonals are also zero.
  // so it's safe to replace diagonal with one.
  for(int j = 0; j < mat.outerSize(); ++j) {
    Eigen::SparseMatrix<double>::InnerIterator it(mat, j);
    while(it && it.index() != j) ++it;
    if(it && it.index() == j && it.value() == 0)
      it.valueRef() = 1.;
  }
  return pmat;
}

struct CgSolve {
  const GeomTarget *geom;
  const LL *ll;
  double gamma = 0.;
  double toler = 1.e-4;
  int ncycle = 2000;
  int max_gamma_cyc = 500;
  CgSolve(const GeomTarget *geom, const LL *ll)
    : geom(geom), ll(ll) {}

  template<typename Preconditioner = Eigen::IdentityPreconditioner>
  Eigen::VectorXd solve(double weight, const gemmi::Logger& logger){
    Eigen::VectorXd vn = Eigen::VectorXd::Map(geom->vn.data(), geom->vn.size());
    Eigen::SparseMatrix<double> am = geom->make_spmat();
    logger.mesg("diag(geom) min= ", std::to_string(am.diagonal().minCoeff()),
                " max= ", std::to_string(am.diagonal().maxCoeff()));
    if (ll != nullptr) {
      auto ll_mat = ll->make_spmat();
      logger.mesg("diag(data) min= ", std::to_string(ll_mat.diagonal().minCoeff()),
                  " max= ", std::to_string(ll_mat.diagonal().maxCoeff()));
      vn += Eigen::VectorXd::Map(ll->vn.data(), ll->vn.size()) * weight;
      am += ll_mat * weight;
    }
    logger.mesg("diag(all) min= ", std::to_string(am.diagonal().minCoeff()),
                " max= ", std::to_string(am.diagonal().maxCoeff()));
    const int n = am.cols();

    // this changes am
    Eigen::SparseMatrix<double> pmat = diagonal_preconditioner(am);
    vn = (pmat * vn).eval();

    if (gamma == 0 && max_gamma_cyc == 1) {
      Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper,
                               Preconditioner> cg;
      Eigen::VectorXd dv(n);
      cg.setMaxIterations(ncycle);
      cg.setTolerance(toler);
      cg.compute(am);
      dv = cg.solve(vn);
      logger.mesg("#iterations:     ", cg.iterations(), "\n",
                  "estimated error: ", std::to_string(cg.error()));
      return pmat * dv;
    }

    // if Preconditioner is not Identity, gamma cycle should not be used.
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
      logger.mesg("Trying gamma equal ", std::to_string(gamma));
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
            logger.mesg("Convergence reached after ", itr+1, " iterations with no gamma cycles");
            exit_flag = true;
            break;
          } else if (conver_flag) {
            logger.mesg("Convergence reached with gamma equal ", std::to_string(gamma));
            step *= 1.01;
            exit_flag = true;
            break;
          } else {
            conver_flag = true;
            gamma_save = gamma;
            dv_save = dv;
            gamma = std::max(0., gamma - step/5.);
            step = std::max(step/1.1, 0.0001);
            logger.mesg("Gamma decreased to ", std::to_string(gamma));
            exit_flag = true;
            break;
          }
        }

        z = precond.solve(r);
        double rho1 = rho0;
        rho0 = r.dot(z);
        if (rho0 > 4 * rho1) {
          logger.mesg("Not converging with gamma equal ", std::to_string(gamma));
          step *= 1.05;
          break;
        }
        double beta = rho0 / rho1;
        p = z + beta * p;
      }
      if (exit_flag) break;
      if (max_gamma_cyc == 1) break; // test
      gamma_flag = true;
      if (!conver_flag)
        dv.setZero();
      else {
        dv = dv_save;
        gamma = gamma_save;
        logger.mesg("Back to gamma equal ", std::to_string(gamma));
      }
    }
    return pmat * dv;
  }
};

} // namespace servalcat
#endif
