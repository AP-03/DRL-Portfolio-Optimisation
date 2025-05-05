import numpy as np
from portfolioEnv import DifferentialSharpeReward   # your class

dsh = DifferentialSharpeReward()
for _ in range(1_000_000):
    r = np.random.normal(0, 0.05)          # simulate 20 % daily σ
    if np.random.rand() < 1e-5:            # inject random inf / nan
        r = np.inf * np.random.choice([-1, 1]) if np.random.rand() < 0.5 else np.nan
    d = dsh.compute(r)
    assert np.isfinite(d)                  # will raise if any nan/inf survives
print("✅ stress-test passed")

