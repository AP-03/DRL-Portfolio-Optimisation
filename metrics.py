import numpy as np
import pandas as pd

def compute_metrics(returns: pd.Series):
    cumulative_return = np.exp(returns.cumsum())[-1] - 1
    annual_return = (1 + returns.mean()) ** 252 - 1
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    calmar = annual_return / abs((returns.cummax() - returns).min())
    stability = returns.cumsum().corr(pd.Series(np.arange(len(returns)), index=returns.index))
    max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()

    downside_std = returns[returns < 0].std()
    sortino = returns.mean() / (downside_std + 1e-8) * np.sqrt(252)

    omega = (returns[returns > 0].sum() / abs(returns[returns < 0].sum()))
    tail_ratio = (returns[returns > returns.quantile(0.95)].mean() /
                  abs(returns[returns < returns.quantile(0.05)].mean()))

    value_at_risk = returns.quantile(0.01)
    skew = returns.skew()
    kurt = returns.kurtosis()

    return {
        "Annual return": annual_return,
        "Cumulative returns": cumulative_return,
        "Annual volatility": annual_vol,
        "Sharpe ratio": sharpe,
        "Calmar ratio": calmar,
        "Stability": stability,
        "Max drawdown": max_drawdown,
        "Omega ratio": omega,
        "Sortino ratio": sortino,
        "Skew": skew,
        "Kurtosis": kurt,
        "Tail ratio": tail_ratio,
        "Daily value at risk": value_at_risk
    }
