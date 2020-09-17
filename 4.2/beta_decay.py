#!/usr/bin/env python
# coding: utf-8

# # Отчет по лабораторной работе 4.2
# 
# ## Исследование энергетического спектра β-частиц и определение их максимальной энергии при помощи магнитного спектрометра
# Конкс Эрик, Б01-818

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy import odr

N = []
p = []
T = []
pd.DataFrame({'N': N, 'p, кэВ/с': p, 'T, кэВ': T})


# In[ ]:


y = np.sqrt(N) / p
y_error = []
x = T
x_error = []
pd.DataFrame({'sqrt(N)/p': x, 'Δ(sqrt(N)/p)': x_err, 'Ee-E': y, 'Δ(Ee-E)': y_err})


# In[ ]:


font = {'size'   : 20}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = [18, 14]


# $$\frac{\sqrt{N(p)}}{p} \approx E_e - E$$

# In[ ]:


f = lambda p, x: p[0] * x + p[1]
quad_model = odr.Model(f)
data = odr.RealData(x, y, sx=x_error, sy=y_error)
modr = odr.ODR(data, quad_model, beta0=[0.002, 0.002])
out = modr.run()
beta_opt = out.beta
beta_err = out.sd_beta
beta_name = ['a', 'b']
print('Fit parameter 1-sigma error y = a * x + b')
print('———————————–—————————————————————————————')
for i in range(len(beta_opt)):
    print(f"{beta_name[i]} = {beta_opt[i]} +- {beta_err[i]}")
    print("    {:.5f} +- {:.5f}".format(beta_opt[i], beta_err[i]))
    
print(f"chisq = {out.res_var * (len(x) - len(beta_opt))}")


# In[ ]:


plt.plot(x, f(beta_opt, x), color='black', linewidth=4, label='fit curve')
plt.plot(x, y, 'ro', label='data points', markersize=12)
plt.errorbar(x, y, xerr=x_error, yerr=y_error, fmt="none", linewidth=4)
plt.xlabel('sqrt(N) / p)')
plt.ylabel('Ee - E')
plt.grid(linewidth=2)
plt.legend()
plt.title('Fermi–Kurie plot')
plt.show()

