import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def poly_exp_r2rho(r, a, b, c, d, g):
    return r**2*(a + b*r + c*r**2 + d*r**4)*np.exp(-g*r)

def poly_exp_rho(r, a, b, c, g):
    return (a + b*r + c*r**2)*np.exp(-g*r)

data = np.loadtxt("/home/soumikd/symbolic_regression/SymbolicRegression.jl/soumikd/vxc_exc/Be/vxc_exc_rho_matrix.txt",
                  delimiter="\t")

R = np.array(data[:, 0])
rho = np.array(data[:, 1])

popt, pcov = curve_fit(poly_exp_r2rho, R, R**2*rho, p0=np.array([35, 4, 5, 7, 10]))

print(popt)

# plt.plot(R, rho)
# plt.plot(R, poly_exp_rho(R, *popt))

###############################################################################
### R^2 (26.23767052 -56.15001832 R + 32.18549571 R^2) exp(- 4.06034866 R) ####
###############################################################################

plt.plot(R, R**2*rho, 'b-')
plt.plot(R, poly_exp_r2rho(R, *popt), 'r-')
plt.xlim(-1, 8)
plt.legend(["$r^2\\rho$", "fit of $r^2*(a + br + cr^2)e^{-gr}$"])

plt.show()
