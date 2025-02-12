import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def poly_exp_r2rho(r, a, b, g, h):
    return r**2*(a*np.exp(-g*r) + b*r**2*np.exp(-h*r))

data = np.loadtxt("/home/soumikd/symbolic_regression/SymbolicRegression.jl/soumikd/vxc_exc/Be/vxc_exc_rho_matrix.txt",
                  delimiter="\t")

R = np.array(data[:, 0])
rho = np.array(data[:, 1])

popt, pcov = curve_fit(poly_exp_r2rho, R, R**2*rho, p0=np.array([35, 7, 4, 1]))

print(popt)

# plt.plot(R, rho)
# plt.plot(R, poly_exp_rho(R, *popt))

######################################################################################
###  rho(r) = #### 33.82613608 exp(- 7.58284232 r) + 0.19754759 r^2 exp(-1.95816344 r) ###
######################################################################################

plt.plot(R, R**2*rho, 'b-')
plt.plot(R, poly_exp_r2rho(R, *popt), 'r-')
plt.xlim(-1, 8)
plt.legend(["$r^2\\rho$", "fit of $r^2*(ae^{-gr} + br^2e^{-hr})$"])

plt.show()
