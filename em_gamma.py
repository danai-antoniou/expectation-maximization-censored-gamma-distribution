import numpy as np
import scipy.stats as st
import math
from scipy import special as stsp
import mpmath
from scipy.optimize import newton

# Test data to set up EM
shape, scale = 2.5, 2
num = 1000
real_demand = st.gamma.rvs(a=shape, scale=scale, size=num)  # vector with true observations
lim = st.gamma.rvs(a=shape, scale=scale, size=num)
# Observed demand (flag)
demand = real_demand * (real_demand <= lim) + lim * (real_demand > lim)  # vector with some censored observations

# Flag to indicate which observations are censored
flag = real_demand > lim
flag = flag * 1


def em_gamma(demand, flag, tol=1e-8):  # tol = tolerance level for convergence condition
    num_constrained = np.sum(flag == 1)  # Constrained demand
    num_unconstrained = np.sum(flag == 0)  # Unconstrained demand
    n = num_constrained + num_unconstrained  # Total num obs
    constrained_demand = demand[flag == 1]

    # Initialise kappa and theta at moment estimators
    theta = ((n - 1) / n * np.var(demand[flag == 0])) / np.mean(demand[flag == 0])
    kappa = np.mean(demand[flag == 0]) / theta

    # Store EM updates for params
    store_params = [theta, kappa]
    iter_num = 1
    theta_update = theta
    kappa_update = kappa
    while (abs(theta_update - theta) >= tol and abs(kappa_update - kappa) >= tol) or iter_num == 1:
        theta = theta_update
        kappa = kappa_update

        ##########
        # E-step #
        ##########
        # Compute the sufficient statistics for the gamma distribution
        # E[x|x>c, theta, kappa] = z
        numerator = stsp.gammaincc(kappa + 1, constrained_demand / theta) * stsp.gamma(
            kappa + 1)  # undo scipy's regularisation
        denominator = stsp.gammaincc(kappa, constrained_demand / theta) * stsp.gamma(
            kappa)  # undo scipy's regularisation

        z = theta * numerator / denominator

        # Now compute E[ln(x)|x>c, theta, kappa]
        # This is tricky because it needs the partial derivative wrt to kappa of the incomplete
        # gamma function: Wolfram Alpha derivation: http://functions.wolfram.com/GammaBetaErf/Gamma2/20/01/01/0001/
        # or for general form: https://link.springer.com/content/pdf/10.1007%2FBF01810298.pdf - page 156
        apply_meijerg = np.frompyfunc(lambda x: float(mpmath.meijerg(a_s=[[], [1, 1]], b_s=[[0, 0, kappa], []], z=x)),
                                      1, 1)
        calc_meijerg = apply_meijerg(constrained_demand / theta).astype(float)
        # Compute E[ln(x)|x>c, theta, kappa] = lnz
        lnz = math.log(theta) + np.log(constrained_demand / theta) + \
              calc_meijerg / (stsp.gammaincc(kappa, constrained_demand / theta) * stsp.gamma(kappa))

        # Compute sufficient statistics needed for the M-step
        # First, Σxi = sum(true uncensored) + sum(conditional expectations)
        sum_x = sum(demand[flag == 0]) + sum(z)

        # Second, Σln(xi) - same logic
        sum_lnx = sum(np.log(demand[flag == 0])) + sum(lnz)

        demand[flag == 1] = z
        ##########
        # M-step #
        ##########
        # Solve system of eqs for theta and sub to equation for kappa
        # The updates for kappa (kappa hat) will come from Newton Raphson - there's no closed form solution
        # then sub back in to get theta hat
        trigamma = lambda x: stsp.polygamma(1, x)  # derivative of digamma function

        # Need the roots of this thing
        def f(x):
            return n * math.log(x) - n * math.log(sum_x / n) + sum_lnx - n * stsp.digamma(x)

        # First derivative of the thing
        def fprime(x):
            return n / x - n * trigamma(x)

        # Initial guess for root, shouldn't make a massive difference
        x_bar = np.mean(demand)
        x2_bar = np.mean(demand ** 2)

        kappa_init = x_bar ** 2 / (x2_bar - x_bar ** 2)
        newton_kappa = [kappa_init]
        maxiter = 101  # Newton-Rapson iterations

        # Solve score eqs numerically
        for i in range(0, maxiter):
            kappa_hat = newton_kappa[i] - f(newton_kappa[i]) / fprime(newton_kappa[i])
            newton_kappa.append(kappa_hat)

        kappa_update = newton_kappa[maxiter]
        # kappa_update = newton(func=f, x0=kappa_init, fprime=fprime, maxiter=int(100)) #scipy NR

        theta_update = (sum_x / n) / kappa_update

        new_vals = [theta_update, kappa_update]
        store_params = np.vstack([store_params, new_vals])
        iter_num += 1
        print("Iteration Number", iter_num)
    print("The final parameter vals are", new_vals)
    print("Params path", store_params)
    return demand


df = em_gamma(demand, flag)