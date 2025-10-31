from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# 打印当前时间
time1 = datetime.datetime.now()
print("Start time:", time1)

# 参数设置（从第二段代码中来）
N = 12
s_a1 = 4.2
s_a2 = 4.2
lam1 = 0.8
lam2 = 0.8
lam = 2
kerr = 0.75
delta_b = 0

# 初始态
psi0 = tensor(basis(N, 0), basis(N, 0), basis(N, 0))

# 算符
a1 = tensor(qeye(N), destroy(N), qeye(N))
a2 = tensor(qeye(N), qeye(N), destroy(N))
b = tensor(destroy(N), qeye(N), qeye(N))

# 哈密顿量结构
H0 = (
    s_a1.conjugate() * a1.dag() * a1.dag() + s_a1 * a1 * a1 +
    delta_b * b.dag() * b +
    kerr * (a1.dag() * a1.dag() * a1 * a1) +
    s_a2.conjugate() * a2.dag() * a2.dag() + s_a2 * a2 * a2 +
    kerr * (a2.dag() * a2.dag() * a2 * a2)
)
H1 = b * b - b.dag() * b.dag()

# 塌缩项
c_ops = [
    lam1 * (b + b.dag()) * (a1 + a2),
    lam2 * (b - b.dag()) * (a1 - a2) * 1j,
    np.sqrt(10) * b,
    lam * (a1 + a2),
    lam * (a1 - a2)
]

# ----------------------------------------
# 第一段演化：s_b = 2.4j, t ∈ [0, 10]
# ----------------------------------------
t1 = np.linspace(0, 10, 50)
s_b1 = -2.4j
H_part1 = H0 + s_b1 * H1

res1 = mesolve(H_part1, psi0, t1, c_ops=c_ops,
               e_ops=[(a1+a1.dag())*(a2+a2.dag())/2,
                        -1*(a1-a1.dag())*(a2-a2.dag())/2],
               options=Options(store_states=True))

# ----------------------------------------
# 第二段演化：s_b = -2.4j, t ∈ [10, 20]
# ----------------------------------------
t2 = np.linspace(10, 20, 50)
s_b2 = 2.4j
H_part2 = H0 + s_b2 * H1

res2 = mesolve(H_part2, res1.states[-1], t2, c_ops=c_ops,
               e_ops=[((a1+a1.dag())*(a2+a2.dag()))/2,
                        -1*((a1-a1.dag())*(a2-a2.dag()))/2],
               options=Options(store_states=True))

# ----------------------------------------
# 合并时间和期望值
# ----------------------------------------
t_full = np.concatenate([t1, t2])
a1a2_full = np.concatenate([np.real(res1.expect[0]), np.real(res2.expect[0])])
a2a1_full = np.concatenate([np.real(res1.expect[1]), np.real(res2.expect[1])])


# 保存 x1x2
df_x = pd.DataFrame({'Time': t_full, 'x1x2_expect': a1a2_full})
df_x.to_csv(r'E:\s_adiss_expect_part\nega\sa_diss_expect_xp_partialH_s_b_switch-12_x.csv', index=False)


# 保存 p1p2
df_p = pd.DataFrame({'Time': t_full, 'x1x2_expect': a2a1_full})
df_p.to_csv(r'E:\s_adiss_expect_part\nega\sa_diss_expect_xp_partialH_s_b_switch-12_p.csv', index=False)

# ----------------------------------------
# 分开绘图
# ----------------------------------------
plt.figure(figsize=(12, 5))

# ⟨a₁†a₂⟩
plt.subplot(1, 2, 1)
plt.plot(t_full, a1a2_full, 'r-')
plt.title('x1*x2 over time')
plt.xlabel('Time')
plt.ylabel('Expectation Value')

# ⟨a₂†a₁⟩
plt.subplot(1, 2, 2)
plt.plot(t_full, a2a1_full, 'b--')
plt.title('p1*p2 over time')
plt.xlabel('Time')
plt.ylabel('Expectation Value')

plt.tight_layout()
plt.show()

# 打印结束时间
time2 = datetime.datetime.now()
print("End time:", time2)
