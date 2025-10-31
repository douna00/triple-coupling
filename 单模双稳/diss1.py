<<<<<<< HEAD
from qutip import (about, basis, concurrence, destroy, expect, fidelity,
                   ket2dm, mesolve, ptrace, qeye, sigmaz,
                   tensor, sigmam, sigmap, wigner)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

s_a = 1
s_b = 1.5j
lam = 0.8
N = 20
kerr = 0.75
delta_b = 0  # 先定 lam=0，调出猫态 sa, kerr 再改 lam, sb，集体耗散怎么影响态

psi0 = tensor(basis(N, 0), basis(N, 0))
a = tensor(destroy(N), qeye(N))
b = tensor(qeye(N), destroy(N))

H = (
    s_a.conjugate() * a.dag() * a.dag() + s_a * a * a +
    s_b.conjugate() * b.dag() * b.dag() + s_b * b * b +
    delta_b * b.dag() * b +
    kerr * (a.dag() * a.dag() * a * a)
)

t = np.linspace(0, 20, 100)
result = mesolve(H, psi0, t, [lam * (b + b.dag()) * a, np.sqrt(10) * b])

vecx = np.linspace(-3, 3, 1000)
W = wigner(result.states[99].ptrace(0), vecx, vecx)

# 保存 Wigner 数据为 CSV
wigner_df = pd.DataFrame(W, index=vecx, columns=vecx)
wigner_df.to_csv(r"E:\result_data\single_mode_bistable\posi\bistable_1_1.5j_0.8_0.75_20.csv")

# 绘图
fig, ax = plt.subplots()
co = ax.contourf(vecx, vecx, W, 100, levels=np.linspace(-0.25, 0.25, 100), cmap='RdBu_r')
plt.colorbar(co, ax=ax)
plt.title('cat state (p)')
ax.set_xlabel('x')
ax.set_ylabel('p')
plt.tight_layout()
plt.show()
=======
from qutip import (about, basis, concurrence, destroy, expect, fidelity,
                   ket2dm, mesolve, ptrace, qeye, sigmaz,
                   tensor, sigmam, sigmap, wigner)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

s_a = 1
s_b = 1.5j
lam = 0.8
N = 20
kerr = 0.75
delta_b = 0  # 先定 lam=0，调出猫态 sa, kerr 再改 lam, sb，集体耗散怎么影响态

psi0 = tensor(basis(N, 0), basis(N, 0))
a = tensor(destroy(N), qeye(N))
b = tensor(qeye(N), destroy(N))

H = (
    s_a.conjugate() * a.dag() * a.dag() + s_a * a * a +
    s_b.conjugate() * b.dag() * b.dag() + s_b * b * b +
    delta_b * b.dag() * b +
    kerr * (a.dag() * a.dag() * a * a)
)

t = np.linspace(0, 20, 100)
result = mesolve(H, psi0, t, [lam * (b + b.dag()) * a, np.sqrt(10) * b])

vecx = np.linspace(-3, 3, 1000)
W = wigner(result.states[99].ptrace(0), vecx, vecx)

# 保存 Wigner 数据为 CSV
wigner_df = pd.DataFrame(W, index=vecx, columns=vecx)
wigner_df.to_csv(r"E:\result_data\single_mode_bistable\posi\bistable_1_1.5j_0.8_0.75_20.csv")

# 绘图
fig, ax = plt.subplots()
co = ax.contourf(vecx, vecx, W, 100, levels=np.linspace(-0.25, 0.25, 100), cmap='RdBu_r')
plt.colorbar(co, ax=ax)
plt.title('cat state (p)')
ax.set_xlabel('x')
ax.set_ylabel('p')
plt.tight_layout()
plt.show()
>>>>>>> bedfc09 (python程序设计)
