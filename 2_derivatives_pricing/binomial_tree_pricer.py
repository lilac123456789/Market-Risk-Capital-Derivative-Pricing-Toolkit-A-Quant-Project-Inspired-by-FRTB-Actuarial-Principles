import numpy as np

def binomial_tree(S, K, r, sigma, T, steps=100, option_type="call", american=False):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt) - d) / (u - d)

    # Build underlying price tree
    ST = np.zeros((steps+1, steps+1))
    for i in range(steps+1):
        for j in range(i+1):
            ST[j, i] = S * (u**(i-j)) * (d**j)

    # Option values at maturity
    if option_type == "call":
        opt = np.maximum(ST[:, steps] - K, 0)
    else:
        opt = np.maximum(K - ST[:, steps], 0)

    # Backward induction
    for i in range(steps-1, -1, -1):
        for j in range(i+1):
            continuation = np.exp(-r*dt) * (p*opt[j] + (1-p)*opt[j+1])
            if american:
                intrinsic = (
                    max(ST[j,i] - K, 0) if option_type == "call" 
                    else max(K - ST[j,i], 0)
                )
                opt[j] = max(continuation, intrinsic)
            else:
                opt[j] = continuation

    return opt[0]


# Example usage
if __name__ == "__main__":
    price = binomial_tree(100, 100, 0.05, 0.2, 1, steps=200, option_type="call")
    print("Binomial call price:", price)
