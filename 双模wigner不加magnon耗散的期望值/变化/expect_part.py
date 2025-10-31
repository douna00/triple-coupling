from qutip import (basis, destroy, mesolve, qeye, tensor, Options)
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd

# ----------------------------------------------------
# 参数设置
s_a1 = 0.6
s_a2 = 0.6
lam1 = 0.8
lam2 = 0.8
N = 12
kerr = 0.75
delta_b = 0
t = np.linspace(0, 40, 100)  # 0-40, 100步

# ----------------------------------------------------
# 时间依赖哈密顿量
def time_dependent_hamiltonian(t, args=None):
    return 1j if t < 20 else -1j

# ----------------------------------------------------
# 初始态
psi0 = tensor(basis(N, 0), basis(N, 0), basis(N, 0))

# ----------------------------------------------------
# 算符
a1 = tensor(qeye(N), destroy(N), qeye(N))
a2 = tensor(qeye(N), qeye(N), destroy(N))
b = tensor(destroy(N), qeye(N), qeye(N))

# ----------------------------------------------------
# 哈密顿量
H0 = (
    s_a1.conjugate() * a1.dag() * a1.dag() + s_a1 * a1 * a1 + delta_b * b.dag() * b
    + kerr * (a1.dag() * a1.dag() * a1 * a1)
    + s_a2.conjugate() * a2.dag() * a2.dag() + s_a2 * a2 * a2
    + kerr * (a2.dag() * a2.dag() * a2 * a2)
)
H1 = b * b - b.dag() * b.dag()
H = [H0, [H1, time_dependent_hamiltonian]]

# ----------------------------------------------------
# 观测量
e_ops = [
    ((a1 + a1.dag()) * (a2 + a2.dag())) / 2,            # 位移期望值乘积
    -1 * ((a1 - a1.dag()) * (a2 - a2.dag())) / 2        # 动量期望值乘积
]

# ----------------------------------------------------
# 求解
options = Options(store_states=True)
time1 = datetime.datetime.now()
print("开始时间:", time1)

result = mesolve(
    H, psi0, t,
    c_ops=[
        lam1 * (b + b.dag()) * (a1 + a2),
        lam2 * (b - b.dag()) * (a1 - a2) * 1j,
        np.sqrt(10) * b
    ],
    e_ops=e_ops,
    options=options
)

time2 = datetime.datetime.now()
print("结束时间:", time2)

# ----------------------------------------------------
# 提取结果
expect1 = np.real(result.expect[0])  # 位移乘积
expect2 = np.real(result.expect[1])  # 动量乘积

# ----------------------------------------------------
# 保存为 CSV 文件
data = pd.DataFrame({
    "Time": t,
    "Displacement_Product": expect1,
    "Momentum_Product": expect2
})
data.to_csv("expectation_products.csv", index=False)
print("已保存为 expect_products.csv 文件")

# ----------------------------------------------------
# 绘图
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(t, expect1, color='red')
plt.xlabel('Time')
plt.ylabel('Displacement Product')
plt.title('x')

plt.subplot(1, 2, 2)
plt.plot(t, expect2, color='blue')
plt.xlabel('Time')
plt.ylabel('Momentum Product')
plt.title('p')

plt.tight_layout()
plt.show()
