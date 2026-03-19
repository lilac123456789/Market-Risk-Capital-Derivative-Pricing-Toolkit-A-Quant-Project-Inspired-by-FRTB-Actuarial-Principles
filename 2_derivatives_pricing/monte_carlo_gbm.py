import numpy as np

def monte_carlo_call(S, K, r, sigma, T, n_sims=10000):
    np.random.seed(42)
    Z = np.random.normal(0, 1, n_sims)
    ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)

    payoff = np.maximum(ST - K, 0)
    price = np.exp(-r*T) * np.mean(payoff)
    return price


if __name__ == "__main__":
    price = monte_carlo_call(100, 100, 0.05, 0.2, 1)
    print("Monte Carlo call price:", price)
