import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """ S: valeur actuelle de l'action, K: strike, T: maturité en années, r: taux sans risque, sigma: volatilité prix, option_type: 'call' ou 'put' """
    d1= (np.log(S/K) + (r + 0.5*sigma**2)*T)/ (sigma*np.sqrt(T))
    d2= d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else: 
        price = K*np.exp(-r*T)*norm.cdf(-d2)-S *norm.cdf(-d1)

    return price, d1, d2

def greeks (S,K,T,r,sigma): 
    d1= (np.log(S/K) + (r + 0.5*sigma**2)*T)/ (sigma*np.sqrt(T))
    d2= d1 - sigma*np.sqrt(T)
    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    vega = S*norm.pdf(d1)*np.sqrt(T)/10
    return delta_call, delta_put, gamma, vega    

#Exemple : 
S=100 
K=105 
T=1 
r=0.05 
sigma=0.2

call_price, d1, d2 = black_scholes(S,K,T,r,sigma,'call')
put_price, _,_ = black_scholes(S,K,T,r,sigma,'put')
delta_call, delta_put, gamma, vega = greeks(S,K,T,r,sigma)

print(f"call price: {call_price:.2f}")
print(f"put price: {put_price:.2f}")
print(f"delta call:{delta_call:.3f}")
print(f"delta put :{delta_put:.3f}")
print(f"gamma:{gamma:.5f}")
print(f"vega : {vega:.3f}")

ST = np.linspace(50,150,100)
payoff_call=np.maximum(ST-K,0)
payoff_put=np.maximum(K-ST,0)

plt.plot(ST, payoff_call, label='call payoff')
plt.plot(ST, payoff_put, label='put payoff')
plt.axvline(K, color='black', linestyle='--', label='Strike (K)')
plt.text(K, 5, 'ATM', ha='center', color='black', fontsize=12)
plt.text(K - 20, 10, 'Put ITM\nCall OTM', ha='center', color='red', fontsize=10)
plt.text(K + 20, 10, 'Call ITM\nPut OTM', ha='center', color='blue', fontsize=10)
plt.axhline(0, color='black', lw=0.5)
plt.xlabel('Prix du sous-jacent à maturité')
plt.ylabel('payoff')
plt.title('Payoff des options Call et Put')
plt.legend()
plt.show()
