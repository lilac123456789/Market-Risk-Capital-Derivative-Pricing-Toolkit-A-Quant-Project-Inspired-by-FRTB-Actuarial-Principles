import numpy as np

def survival_probability(lambda_t, t):
    """
    Exponential survival model:
    lambda_t: hazard rate
    t: time horizon (years)
    """
    return np.exp(-lambda_t * t)

def present_value_survival(benefit, lambda_t, r, t):
    """
    PV of a life-contingent benefit.
    
    benefit: cashflow amount
    lambda_t: hazard rate
    r: interest rate
    t: time horizon
    """
    S = survival_probability(lambda_t, t)
    discount = np.exp(-r * t)
    return benefit * S * discount


# Example usage
if __name__ == "__main__":
    # A benefit paid if alive at year 5
    pv = present_value_survival(100000, lambda_t=0.02, r=0.03, t=5)
    print("PV of survival-contingent benefit:", pv)
