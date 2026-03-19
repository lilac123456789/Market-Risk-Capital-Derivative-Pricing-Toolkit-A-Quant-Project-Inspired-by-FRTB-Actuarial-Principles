import numpy as np

def efficient_frontier(returns, cov, num_points=50):
    """
    Compute efficient frontier.
    
    returns: expected returns vector
    cov: covariance matrix
    num_points: number of portfolios to generate
    """

    n = len(returns)
    w = np.zeros((num_points, n))
    port_returns = np.zeros(num_points)
    port_vols = np.zeros(num_points)

    for i in range(num_points):
        weights = np.random.random(n)
        weights /= np.sum(weights)

        w[i] = weights
        port_returns[i] = np.dot(weights, returns)
        port_vols[i] = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

    return w, port_returns, port_vols


# Example Usage
if __name__ == "__main__":
    returns = np.array([0.08, 0.12, 0.10])
    cov = np.array([
        [0.04, 0.006, 0.004],
        [0.006, 0.09, 0.008],
        [0.004, 0.008, 0.025]
    ])

    w, r, v = efficient_frontier(returns, cov)
    print("Example frontier generated!")
