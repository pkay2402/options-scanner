import pandas as pd
from datetime import datetime, timedelta

def filter_options_chain(chain, current_price):
    strike_min = current_price * 0.8
    strike_max = current_price * 1.2
    expiry_today = datetime.now().date()
    expiry_limit = expiry_today + timedelta(weeks=12)
    options = []
    for expiry, contracts in chain['options'].items():
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        if expiry_today <= expiry_date <= expiry_limit:
            for contract in contracts:
                strike = contract['strikePrice']
                if strike_min <= strike <= strike_max:
                    options.append({
                        'expiry': expiry_date,
                        'strike': strike,
                        'type': contract['putCall'],
                        'delta': contract.get('delta'),
                        'gamma': contract.get('gamma'),
                        'theta': contract.get('theta'),
                        'vega': contract.get('vega'),
                        'bid': contract.get('bid'),
                        'ask': contract.get('ask'),
                        'last': contract.get('last'),
                        'openInterest': contract.get('openInterest'),
                        'volume': contract.get('totalVolume'),
                    })
    return pd.DataFrame(options)

def aggregate_greeks(df):
    greek_cols = ['delta', 'gamma', 'theta', 'vega']
    return df.groupby('expiry')[greek_cols].sum()

def generate_greek_insights(df, aggs):
    insights = []
    # Pin risk: high gamma
    high_gamma = df[df['gamma'] == df['gamma'].max()]
    if not high_gamma.empty:
        strike = high_gamma.iloc[0]['strike']
        expiry = high_gamma.iloc[0]['expiry']
        insights.append(f"High gamma concentration at strike ${strike:.2f} for expiry {expiry}. Expect volatility and possible pin risk near this level.")
    # Directional bias: net delta
    net_delta = aggs['delta'].sum()
    if net_delta > 0:
        insights.append("Net positive delta: Option market is positioned bullishly.")
    elif net_delta < 0:
        insights.append("Net negative delta: Option market is positioned bearishly.")
    else:
        insights.append("Net delta is neutral: No strong directional bias.")
    # Premium decay: high theta
    high_theta = df[df['theta'] == df['theta'].min()]
    if not high_theta.empty:
        strike = high_theta.iloc[0]['strike']
        expiry = high_theta.iloc[0]['expiry']
        insights.append(f"Highest theta decay at strike ${strike:.2f} for expiry {expiry}. Consider premium selling strategies here.")
    # Volatility zones: high vega
    high_vega = df[df['vega'] == df['vega'].max()]
    if not high_vega.empty:
        strike = high_vega.iloc[0]['strike']
        expiry = high_vega.iloc[0]['expiry']
        insights.append(f"High vega at strike ${strike:.2f} for expiry {expiry}. These options are most sensitive to volatility changes.")
    return insights
