"""
Author: "Keitaro Yamashita, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""
from __future__ import absolute_import, division, print_function, generators
from servalcat.utils import logger
import numpy

def cgsolve_rm(A, v, M, gamma=0., ncycl=2000, toler=1.e-4):
    gamma_save = None
    step_flag = False
    gamma_flag = False
    conver_flag = False
    #f = numpy.zeros(len(v))
    dv = numpy.zeros(len(v))
    dv_save = numpy.zeros(len(v))
    
    # preconditioning
    A = M.T.dot(A).dot(M)
    #print("precond_A=")
    #print(A.toarray())
    v = M.T.dot(v)

    vnorm = numpy.sqrt(numpy.dot(v, v))
    test_lim = toler * vnorm
    max_gamma_cyc = 500
    step = 0.05

    for gamma_cyc in range(max_gamma_cyc):
        if gamma_cyc != 0: gamma += step

        logger.writeln("Trying gamma equal {:.4e}".format(gamma))
        r = v - (A.dot(dv) + gamma * dv)
        rho = [numpy.dot(r, r)]
        if rho[0] < toler:
            break
        
        exit_flag = False
        for itr in range(ncycl):
            if itr == 0:
                p = r
                beta = 0.
            else:
                beta = rho[-1] / rho[-2]
                p = r + beta * p

            f = A.dot(p) + gamma * p
            alpha = rho[-1] / numpy.dot(p, f)
            dv += alpha * p
            if itr%20 == 0:
                r = v - (A.dot(dv) + gamma * dv)
            else:
                r = r - alpha * f

            rho.append(numpy.dot(r, r))
            #print("rho=", rho)
            if numpy.sqrt(rho[-1]) > 2 * numpy.sqrt(rho[-2]):
                logger.writeln("Not converging with gamma equal {:.4e}".format(gamma))
                step *= 1.05
                break

            if numpy.sqrt(rho[-1]) < test_lim:
                if not gamma_flag:
                    logger.writeln("Convergence reached with no gamma cycles")
                    exit_flag = True
                    break # goto 120
                elif conver_flag:
                    logger.writeln("Convergence reached with gamma equal {:.4e}".format(gamma))
                    step *= 1.01
                    exit_flag = True
                    break # goto 120
                else:
                    conver_flag = True
                    step_flag = True
                    gamma_save = gamma
                    dv_save = numpy.copy(dv)
                    gamma = max(0, gamma - step/5.)
                    step = max(step/1.1, 0.0001)
                    logger.writeln("Gamma decreased to {:.4e}".format(gamma))
                    exit_flag = True
                    break
        # end of inner loop
        if exit_flag: break
        
        gamma_flag = True
        if not conver_flag:
            dv[:] = 0.
        else:
            dv = numpy.copy(dv_save)
            gamma = gamma_save
            logger.writeln("Back to gamma equal {:.4e}".format(gamma))


    # postconditioning
    dv = M.dot(dv)
    return dv, gamma
# cgsolve_rm()
