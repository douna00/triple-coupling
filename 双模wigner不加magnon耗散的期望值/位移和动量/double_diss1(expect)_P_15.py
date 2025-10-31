from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd

# 打印当前时间
time1 = datetime.datetime.now()
print(time1)

# 参数设置
s_a1 = 0.6
s_a2 = 0.6
s_b = 1j
lam1 = 0.8
lam2 = 0.8
N = 15
kerr = 0.75
delta_b = 0

# 初始态和算符
psi0 = tensor(basis(N, 0), basis(N, 0), basis(N, 0))
a1 = tensor(qeye(N), destroy(N), qeye(N))
a2 = tensor(qeye(N), qeye(N), destroy(N))
b = tensor(destroy(N), qeye(N), qeye(N))

# 哈密顿量
H = (s_a1.conjugate() * a1.dag() * a1.dag() + s_a1 * a1 * a1 +
     s_b.conjugate() * b.dag() * b.dag() + s_b * b * b +
     delta_b * b.dag() * b +
     kerr * (a1.dag() * a1.dag() * a1 * a1) +
     s_a2.conjugate() * a2.dag() * a2.dag() + s_a2 * a2 * a2 +
     kerr * (a2.dag() * a2.dag() * a2 * a2))

# 时间和演化
t = np.linspace(0, 20, 100)

result = mesolve(H, psi0, t,
                 c_ops=[lam1 * (b + b.dag()) * (a1 + a2),
                        lam2 * (b - b.dag()) * (a1 - a2) * 1j,
                        np.sqrt(10) * b],
                 e_ops=[((a1 + a1.dag()) * (a2 + a2.dag())) / 2,
                        -1 * ((a1 - a1.dag()) * (a2 - a2.dag())) / 2])

# 保存 x1x2
df_x = pd.DataFrame({'Time': t, 'x1x2_expect': np.real(result.expect[0])})
df_x.to_csv(r'E:\dualmode_expect_XP_15\posi\x1x2_expect_0.6_1j_0.8_x.csv', index=False)

# 保存 p1p2
df_p = pd.DataFrame({'Time': t, 'p1p2_expect': np.real(result.expect[1])})
df_p.to_csv(r'E:\dualmode_expect_XP_15\posi\p1p2_expect_0.6_1j_0.8_p.csv', index=False)


# 绘图
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(t, np.real(result.expect[0]), color='red')
plt.title('⟨x₁ x₂⟩')
plt.xlabel('Time')
plt.ylabel('Expectation')

plt.subplot(1, 2, 2)
plt.plot(t, np.real(result.expect[1]), color='blue')
plt.title('⟨p₁ p₂⟩')
plt.xlabel('Time')
plt.ylabel('Expectation')

plt.tight_layout()
plt.show()

# 打印结束时间
time2 = datetime.datetime.now()
print(time2)
