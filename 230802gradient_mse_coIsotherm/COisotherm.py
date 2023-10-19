import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time
# 데이터 예시
zero = pd.read_excel("./20degree_.xlsx")
x_data = zero.iloc[:, 0].values
y_data = zero.iloc[:, 1].values

# Toth 식 정의
def toth_function(x, a, b, c):
    return c * b* x / (1 +(b* x) ** a) ** (1 / a)

# 평균 제곱 오차(MSE) 계산 함수
def mean_squared_error(params):
    a_val, b_val, c_val = params
    y_pred = toth_function(x_data, a_val, b_val, c_val)
    return np.mean((y_pred - y_data) ** 2)

# 경사하강법으로 최적의 파라미터 찾기
initial_params = np.array([55 ,10,1276])  # 초기값 설정
methods_list = [
    'Nelder-Mead',
    'Powell',
    'CG',
    'BFGS',
    'Newton-CG',
    'L-BFGS-B',
    'TNC',
    'COBYLA',
    'SLSQP',
    'trust-constr',
    'dogleg',
    'trust-ncg',
    'trust-exact',
    'trust-krylov'
]
a_t = time.time()
result = minimize(mean_squared_error, initial_params, method=methods_list[0], options={'maxiter': 10000})
print(time.time() - a_t)
optimal_a, optimal_b, optimal_c = result.x
min_mse = result.fun
print(result)

print("최적의 파라미터: a =", optimal_a, "b =", optimal_b, "c =", optimal_c)
print(result.x)
print("최적의 MSE:", min_mse*len(x_data))
import matplotlib.pyplot as plt
plt.plot(x_data, y_data)
plt.plot(x_data, toth_function(x_data, optimal_a,optimal_b, optimal_c))
plt.show()