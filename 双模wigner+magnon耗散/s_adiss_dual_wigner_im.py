import numpy as np
import matplotlib.pyplot as plt
from qutip import tensor, qeye, destroy, basis, mesolve
import pandas as pd

# ----------------------------------------------------
# 计算单点的双模 Wigner 函数
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
            temp_coe1 = 0
            temp_coe2 = 0
            for k in range(min(kl, kc) + 1):
                temp_coef = 2 * np.sqrt(fracse[kl] * fracse[kc]) / (fracse[k] * np.pi)
                temp_coe1 += temp_coef * (-1) ** k * a1_power[kl - k] * np.conjugate(a1_power[kc - k])
                temp_coe2 += temp_coef * (-1) ** k * a2_power[kl - k] * np.conjugate(a2_power[kc - k])

            matrix1[kl][kc] = temp_coe1 * np.exp(-np.real(a1_power[1] * np.conjugate(a1_power[1])) / 2)
            matrix2[kl][kc] = temp_coe2 * np.exp(-np.real(a2_power[1] * np.conjugate(a2_power[1])) / 2)

    wigner = 0
    for kc1 in range(N):
        for kc2 in range(N):
            for kl1 in range(N):
                for kl2 in range(N):
                    wigner += np.real(matrix1[kl1][kc1] * matrix2[kl2][kc2] *
                                       rho[kl1 + N * kl2][kc1 + N * kc2])

    return wigner


# ----------------------------------------------------
# 计算双模 Wigner 函数矩阵
def get_wigner(rho, N, Nw, range_val):
    joint_wigner = np.zeros((Nw, Nw), dtype=float)
    fracse = np.zeros(N)
    fracse[0] = 1
    for k in range(1, N):
        fracse[k] = k * fracse[k - 1]

    for kw1 in range(Nw):
        for kw2 in range(Nw):
            a1 = complex(0, -range_val / 2 + range_val * kw1 / Nw)
            a2 = complex(0, -range_val / 2 + range_val * kw2 / Nw)
            joint_wigner[kw1][kw2] = get_jw_point(rho, fracse, N, a1, a2)

    return joint_wigner


# ----------------------------------------------------
# 参数设置
N = 12
Nw = 100
range_val = 8

# 初始态与算符
psi0 = tensor(basis(N, 0), basis(N, 0), basis(N, 0))
a1 = tensor(qeye(N), destroy(N), qeye(N))
a2 = tensor(qeye(N), qeye(N), destroy(N))
b = tensor(destroy(N), qeye(N), qeye(N))

# 哈密顿量与演化
s_a1 = 4.2
s_a2 = 4.2
s_b = -2.4j
lam1 = 0.8
lam2 = 0.8
lam = 2
kerr = 0.75
delta_b = 0

H = (s_a1.conjugate() * a1.dag() * a1.dag() + s_a1 * a1 * a1 +
     s_b.conjugate() * b.dag() * b.dag() + s_b * b * b + delta_b * b.dag() * b +
     kerr * (a1.dag() * a1.dag() * a1 * a1) +
     s_a2.conjugate() * a2.dag() * a2.dag() + s_a2 * a2 * a2 +
     kerr * (a2.dag() * a2.dag() * a2 * a2))

t = np.linspace(0, 20, 100)
result = mesolve(H, psi0, t,
                  c_ops=[
                      lam1 * (b + b.dag()) * (a1 + a2),
                      lam2 * (b - b.dag()) * (a1 - a2) * 1j,
                      np.sqrt(10) * b,
                      lam * (a1 + a2),
                      lam * (a1 - a2)
                  ])

# 取末态的双模子系统
rho = result.states[-1].ptrace([1, 2])
rho_matrix = rho.full()

# ----------------------------------------------------
# 计算双模 Wigner 函数
joint_wigner = get_wigner(rho_matrix, N, Nw, range_val)

# ----------------------------------------------------
# 坐标
x_range = np.linspace(-range_val / 2, range_val / 2, Nw)
y_range = np.linspace(-range_val / 2, range_val / 2, Nw)

# ----------------------------------------------------
# 保存为 CSV 文件，第一行是 x 坐标，第一列是 y 坐标
# 拼接坐标
joint_wigner_csv = np.zeros((Nw + 1, Nw + 1))
joint_wigner_csv[1:, 0] = y_range
joint_wigner_csv[0, 1:] = x_range
joint_wigner_csv[1:, 1:] = np.real(joint_wigner)

# 保存
csv_path = r"E:\s_a_diss_dualwigner\im\dual_wigner_4.2_2_-2.4j_0.8_12.csv"
np.savetxt(csv_path, joint_wigner_csv, delimiter=',')
print("已保存为 CSV 文件：", csv_path)

# ----------------------------------------------------
# 绘制
X, Y = np.meshgrid(x_range, y_range)
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, np.real(joint_wigner), levels=100, cmap='RdBu')
plt.colorbar()
plt.xlabel('p1')
plt.ylabel('p2')
plt.title('Dual-Mode Wigner Function_4.2_2_-2.4j_0.8_12')
plt.axis('equal')
plt.show()
