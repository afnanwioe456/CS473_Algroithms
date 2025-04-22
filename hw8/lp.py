import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# Primal: maximize x1+x2  <=> minimize -[1,1] @ x
c = [-1, -1]
A = [[3,2],
     [0,1],
     [1,5],
     [2,1]]
b = [27, 8, 35, 17]
res = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method='highs')
print("Primal optimal x:", res.x)
print("Primal optimal value:", -res.fun)

# Dual: minimize [27,8,35,17] @ y  s.t. A^T y >= [1,1]
c_dual = [27, 8, 35, 17]
A_dual = np.array(A).T
b_dual = [1,1]
res_dual = linprog(c_dual, A_ub=-A_dual, b_ub=[-1,-1], bounds=(0, None), method='highs')
print("Dual optimal y:", res_dual.x)
print("Dual optimal value:", res_dual.fun)


# 设置 x1 和 x2 的范围
x1 = np.linspace(0, 20, 400)
x2 = np.linspace(0, 20, 400)
X1, X2 = np.meshgrid(x1, x2)

# 各约束对应的函数
c1 = 3*X1 + 2*X2 <= 27
c2 = X2 <= 8
c3 = X1 + 5*X2 <= 35
c4 = 2*X1 + X2 <= 17
c5 = X1 >= 0
c6 = X2 >= 0

# 可行区域（所有约束同时满足）
feasible = c1 & c2 & c3 & c4 & c5 & c6

# 绘图
plt.figure(figsize=(10, 8))
plt.contourf(X1, X2, feasible, levels=[0.5, 1], colors=['#ccf2ff'], alpha=0.7)

# 绘制每条边界线
plt.plot(x1, (27 - 3*x1)/2, label=r'$3x_1 + 2x_2 \\le 27$')
plt.plot(x1, np.full_like(x1, 8), label=r'$x_2 \\le 8$')
plt.plot(x1, (35 - x1)/5, label=r'$x_1 + 5x_2 \\le 35$')
plt.plot(x1, 17 - 2*x1, label=r'$2x_1 + x_2 \\le 17$')

# 绘制目标函数等值线（x1 + x2 = 11）
plt.plot(x1, 11 - x1, 'k--', label=r'$x_1 + x_2 = 11$ (objective)')

# 标出最优点 (5, 6)
plt.plot(5, 6, 'ro', label='Optimal point (5, 6)')

plt.xlim(0, 15)
plt.ylim(0, 10)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Feasible Region for the LP')
plt.legend()
plt.grid(True)
plt.show()
