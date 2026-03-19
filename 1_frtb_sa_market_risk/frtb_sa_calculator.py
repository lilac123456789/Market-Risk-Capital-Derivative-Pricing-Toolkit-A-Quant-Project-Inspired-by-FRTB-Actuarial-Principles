import numpy as np
import pandas as pd

class FRTB_SA:
    def __init__(self, sensitivities, risk_weights, correlations):
        """
        sensitivities: dict {tenor_bucket: value}
        risk_weights: dict {tenor_bucket: RW}
        correlations: 2D matrix for inter-bucket correlations
        """
        self.sens = sensitivities
        self.rw = risk_weights
        self.corr = correlations

    def weighted_sensitivities(self):
        w = {}
        for bucket in self.sens:
            w[bucket] = self.sens[bucket] * self.rw[bucket]
        return w

    def capital_charge(self):
        w = self.weighted_sensitivities()
        buckets = list(w.keys())
        matrix = np.zeros((len(buckets), len(buckets)))

        for i, b1 in enumerate(buckets):
            for j, b2 in enumerate(buckets):
                matrix[i][j] = self.corr[i][j] * w[b1] * w[b2]

        return np.sqrt(matrix.sum())

# Example usage
if __name__ == "__main__":
    sens = {"0-1y": 100000, "1-5y": -50000, "5-30y": 20000}
    rw = {"0-1y": 1.7, "1-5y": 1.3, "5-30y": 1.1}
    corr = [
        [1, 0.7, 0.5],
        [0.7, 1, 0.75],
        [0.5, 0.75, 1]
    ]

    model = FRTB_SA(sens, rw, corr)
    print("Capital charge:", model.capital_charge())
