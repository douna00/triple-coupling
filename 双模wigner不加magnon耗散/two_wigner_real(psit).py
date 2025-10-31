import numpy as np
import matplotlib.pyplot as plt
from qutip import coherent, tensor, Qobj,basis,qeye,mesolve,destroy
import pandas as pd

def get_jw_point(rho, fracse, N, a1, a2):
    a1_power = np.zeros(N, dtype=complex)
    a2_power = np.zeros(N, dtype=complex)
    matrix1 = np.zeros((N, N), dtype=complex)
    matrix2 = np.zeros((N, N), dtype=complex)

    a1_power[0] = 1
    a2_power[0] = 1

    for k in range(1, N):
        a1_power[k] = 2.0 * a1 * a1_power[k - 1] / k
        a2_power[k] = 2.0 * a2 * a2_power[k - 1] / k

    for kc in range(N):
        for kl in range(N):
            temp_coe = 0
            temp_coe2 = 0
            for k in range(min(kl, kc) + 1):
                temp_coe3 = 2 * np.sqrt(fracse[kl] * fracse[kc]) / (fracse[k] * 3.141592654)
                temp_coe += temp_coe3 * (-1) ** k * a1_power[kl - k] * np.conjugate(a1_power[kc - k])
                temp_coe2 += temp_coe3 * (-1) ** k * a2_power[kl - k] * np.conjugate(a2_power[kc - k])

            matrix1[kl][kc] = temp_coe * np.exp(-np.real(a1_power[1] * np.conjugate(a1_power[1])) / 2)
            matrix2[kl][kc] = temp_coe2 * np.exp(-np.real(a2_power[1] * np.conjugate(a2_power[1])) / 2)

    wigner = 0
    for kc1 in range(N):
        for kc2 in range(N):
            for kl1 in range(N):
                for kl2 in range(N):
                    wigner += np.real(matrix1[kl1][kc1] * matrix2[kl2][kc2] * rho[kl1 + N * kl2][kc1 + N * kc2])

    return wigner


def get_wigner(joint_wigner, rho, N, Nw, range_val):
    fracse = np.zeros(N)
    a1 = 0
    a2 = 0

    fracse[0] = 1
    for k in range(1, N):
        fracse[k] = k * fracse[k - 1]

    for kw1 in range(Nw):
        for kw2 in range(Nw):
            a1 = complex(-range_val / 2 + range_val * kw1 / Nw,0)
            a2 = complex(-range_val / 2 + range_val * kw2 / Nw,0)
            joint_wigner[kw1][kw2] = get_jw_point(rho, fracse, N, a1, a2)

    return joint_wigner


# Define parameters15
N = 15 # Number of Fock states
Nw = 100  # Resolution for Wigner function
range_val = 4 # Range for the plot

# Create the superposition state
s_a1 = 1
s_a2 = 1
s_b = -2.4j
lam1 = 0.8
lam2 = 0.8

kerr = 0.75
delta_b = 0#先定lam=0，调出猫态sa,kerr再改lam,sb,集体耗散怎么影响态

psi0 = tensor(basis(N,0),basis(N,0),basis(N,0))
a1 = tensor(qeye(N),destroy(N),qeye(N))
a2 = tensor(qeye(N),qeye(N),destroy(N))
b = tensor(destroy(N),qeye(N),qeye(N))
H = (s_a1.conjugate()*a1.dag()*a1.dag()+s_a1*a1*a1+s_b.conjugate()*b.dag()*b.dag()+s_b*b*b+delta_b*b.dag()*b+kerr*(a1.dag()*a1.dag()*a1*a1)+
     s_a2.conjugate()*a2.dag()*a2.dag()+s_a2*a2*a2+kerr*(a2.dag()*a2.dag()*a2*a2))
t = np.linspace(0,20,100)

result = mesolve(H, psi0, t,
                 c_ops=[lam1*(b+b.dag())*(a1+a2),
                        lam2*(b-b.dag())*(a1-a2)*1j,
                        np.sqrt(10)*b]
                 )

# Create the density matrix
rho = result.states[99].ptrace([1,2])

# Initialize the joint Wigner function array
joint_wigner = np.zeros((Nw, Nw), dtype=complex)

# Calculate the Wigner function
joint_wigner = get_wigner(joint_wigner, rho.full(), N, Nw, range_val)

# Prepare for plotting
x_range = np.linspace(-range_val / 2, range_val / 2, Nw)
y_range = np.linspace(-range_val / 2, range_val / 2, Nw)
X, Y = np.meshgrid(x_range, y_range)



# x_range 和 y_range 是你定义的横纵坐标

df_data = pd.DataFrame(np.real(joint_wigner), index=y_range, columns=x_range)

# 保存到 CSV
df_data.to_csv('E:/two-mode-wigner/real/re_-2.4j_1_0.8_0.75_15range.csv')

# Plot the Wigner function
plt.figure()
plt.contourf(X, Y, np.real(joint_wigner), levels=100, cmap='RdBu')
plt.colorbar()
plt.title('Dual-Mode Wigner Function')
plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal')
plt.show()
