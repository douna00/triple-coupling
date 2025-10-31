import matplotlib.colors
import qutip
from qutip import (basis, destroy, mesolve, qeye, tensor, wigner)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cmap = plt.cm.RdBu_r
s = -2.4j
delta = 0
N = 25
kerr = 0.75
psi0 = basis(N, 0)
a = destroy(N)

# 哈密顿量
H = s * a.dag() * a.dag() + s.conjugate() * a * a

# 时间
t = np.linspace(0, 20, 100)

# 动力学演化
result = mesolve(H, psi0, t, [np.sqrt(10) * a])

# 计算最终时刻的 Wigner 函数
vecx = np.linspace(-6, 6, 100)
W = wigner(result.states[99], vecx, vecx)

# 保存为 CSV
wigner_df = pd.DataFrame(W, index=vecx, columns=vecx)
wigner_df.to_csv(r"E:\result_data\squeezed_wigner_1\nega\wigner_t40_squeeze_-_66.csv")


# 绘图
fig, ax = plt.subplots()
co = ax.contourf(vecx, vecx, W, 100, cmap=cmap)
cbar = plt.colorbar(co)
cbar.set_label('Value range')
plt.title('Squeezed vacuum state_n')
ax.set_xlabel('x')
ax.set_ylabel('p')
plt.tight_layout()
plt.show()
