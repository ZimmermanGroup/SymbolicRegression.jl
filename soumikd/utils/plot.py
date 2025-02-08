import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("/home/soumikd/symbolic_regression/SymbolicRegression.jl/soumikd/vxc_exc/Be/vxc_exc_rho_matrix.txt",
                  delimiter="\t")

R = np.array(data[:, 0])
rho = np.array(data[:, 1])
vxc_actual = np.array(data[:, 2])
exc_actual = np.array(data[:, -1])


# Change as per equation
# exc_per_particle = (3.9145868755050124 / ((R * -7.829173644499098) - 2.2209752841995134)) - (np.exp(R * (-4.231219098189861 * R)) / 0.8689364571490701)
# exc_per_particle = (-2.0490218914757468 / (np.exp(R / 0.33181632420249013) - (R / 0.39961550468630547))) - 0.1858816045163806
exc_per_particle = -0.49995 * (np.exp(-14.559 / (R - 29.473)) / (0.33876 + R)) # rho
# exc_per_particle = ((1.24 / (-0.1833 - R)) - np.exp(-0.2183 * R)) * 0.40316 # r2rho
# exc_per_particle = (0.98622 / (R + 0.25336)) / ((R - 19.305) / ((-29.735 + R) * -0.50706)) # r2rho

exc_predicted = exc_per_particle * rho

# plt.plot(R, exc_actual, '-o', markersize=2)
# plt.plot(R, exc_predicted)
# plt.plot(R[150:], -rho[150:]/(2*R[150:]))
# plt.xlim(-1, 4)
# plt.legend(['$\\epsilon_{xc}^{OA}$', '$\\epsilon_{xc}^{SR}$', '$-\\frac{\\rho}{2r}$'], loc="lower right")
# # plt.savefig('exc_fit.png')

L2_exc = np.sum((exc_actual - exc_predicted)**2)
MSE_exc = np.sum((exc_actual - exc_predicted)**2)/(len(exc_predicted))
weighted_MSE_exc = np.sum(rho*(exc_actual - exc_predicted)**2)/(len(exc_predicted)*np.sum(rho))

idx = np.where(rho >= 1e-7)[0][-1]
truncated_R, truncated_rho, truncated_vxc, truncated_exc = R[:idx+1], rho[:idx+1], vxc_actual[:idx+1], exc_actual[:idx+1]
truncated_exc_per_particle = exc_per_particle[:idx+1]

dexc_dr = (truncated_exc_per_particle[1:] - truncated_exc_per_particle[:-1])/(truncated_R[1:] - truncated_R[:-1])
drho_dr = (truncated_rho[1:]- truncated_rho[:-1])/(truncated_R[1:] - truncated_R[:-1])

vxc_predicted = truncated_exc_per_particle[:-1] + (truncated_rho[:-1]*dexc_dr/(drho_dr + 1e-6))

L2_vxc = np.sum((truncated_vxc[:-1] - vxc_predicted)**2)
MSE_vxc = np.sum((truncated_vxc[:-1] - vxc_predicted)**2)/len(vxc_predicted)
weighted_MSE_vxc = np.sum(truncated_rho[:-1]*(truncated_vxc[:-1] - vxc_predicted)**2)/(len(vxc_predicted)*np.sum(truncated_rho[:-1]))

print(L2_exc, MSE_exc, weighted_MSE_exc)
print(L2_vxc, MSE_vxc, weighted_MSE_vxc)

plt.plot(truncated_R, truncated_vxc, '-o', markersize=2)
plt.plot(truncated_R[:-1], vxc_predicted)
plt.plot(R[200:], -1/(R[200:]))
plt.xlim(-1, 10)
plt.legend(['$v_{xc}^{OA}$', '$v_{xc}^{SR}$', '$-\\frac{1}{r}$'], loc="lower right")
# plt.savefig('vxc_fit.png')
plt.show()