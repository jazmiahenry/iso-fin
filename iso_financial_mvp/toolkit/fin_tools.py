import math
import numpy as np
from typing import List, Union, Tuple, Dict, Optional
import scipy.stats as stats  # Used in some statistical functions like t-distribution

def capital_market_line(risk_free_rate: float, market_return: float, market_stddev: float, 
                      portfolio_stddev: float) -> float:
    """
    Calculate expected return using capital market line
    
    Args:
        risk_free_rate: Risk-free rate (as a decimal)
        market_return: Expected return of market portfolio (as a decimal)
        market_stddev: Standard deviation of market portfolio
        portfolio_stddev: Standard deviation of portfolio
    
    Returns:
        Expected return of portfolio
    """
    return risk_free_rate + ((market_return - risk_free_rate) / market_stddev) * portfolio_stddev


def asset_beta(covariance_with_market: float, market_variance: float) -> float:
    """
    Calculate beta of an asset
    
    Args:
        covariance_with_market: Covariance of asset with market
        market_variance: Variance of market
    
    Returns:
        Beta of the asset
    """
    return covariance_with_market / market_variance


def asset_beta_correlation(correlation: float, asset_stddev: float, market_stddev: float) -> float:
    """
    Calculate beta of an asset using correlation
    
    Args:
        correlation: Correlation between asset and market
        asset_stddev: Standard deviation of asset
        market_stddev: Standard deviation of market
    
    Returns:
        Beta of the asset
    """
    return correlation * (asset_stddev / market_stddev)


def portfolio_beta(weights: List[float], betas: List[float]) -> float:
    """
    Calculate portfolio beta
    
    Args:
        weights: List of weights
        betas: List of betas
    
    Returns:
        Portfolio beta
    """
    return sum(w * b for w, b in zip(weights, betas))


def asset_total_risk(beta: float, market_stddev: float, specific_risk: float) -> float:
    """
    Calculate total risk of an asset
    
    Args:
        beta: Beta of the asset
        market_stddev: Standard deviation of market
        specific_risk: Standard deviation of non-systematic risk
    
    Returns:
        Total risk of the asset
    """
    systematic_risk = beta * market_stddev
    return math.sqrt((systematic_risk ** 2) + (specific_risk ** 2))


def capital_asset_pricing_model(risk_free_rate: float, beta: float, market_risk_premium: float) -> float:
    """
    Calculate expected return using CAPM
    
    Args:
        risk_free_rate: Risk-free rate (as a decimal)
        beta: Beta of the asset
        market_risk_premium: Market risk premium (as a decimal)
    
    Returns:
        Expected return of the asset
    """
    return risk_free_rate + beta * market_risk_premium


def sharpe_ratio(portfolio_return: float, risk_free_rate: float, portfolio_stddev: float) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        portfolio_return: Portfolio return (as a decimal)
        risk_free_rate: Risk-free rate (as a decimal)
        portfolio_stddev: Portfolio standard deviation
    
    Returns:
        Sharpe ratio
    """
    return (portfolio_return - risk_free_rate) / portfolio_stddev


def treynor_ratio(portfolio_return: float, risk_free_rate: float, portfolio_beta: float) -> float:
    """
    Calculate Treynor ratio
    
    Args:
        portfolio_return: Portfolio return (as a decimal)
        risk_free_rate: Risk-free rate (as a decimal)
        portfolio_beta: Portfolio beta
    
    Returns:
        Treynor ratio
    """
    return (portfolio_return - risk_free_rate) / portfolio_beta


def m_squared(portfolio_return: float, risk_free_rate: float, portfolio_stddev: float, 
            market_stddev: float) -> float:
    """
    Calculate M-squared (M²)
    
    Args:
        portfolio_return: Portfolio return (as a decimal)
        risk_free_rate: Risk-free rate (as a decimal)
        portfolio_stddev: Portfolio standard deviation
        market_stddev: Market standard deviation
    
    Returns:
        M-squared
    """
    return ((portfolio_return - risk_free_rate) * (market_stddev / portfolio_stddev)) + risk_free_rate


def m_squared_alpha(portfolio_return: float, risk_free_rate: float, portfolio_stddev: float, 
                  market_stddev: float, market_return: float) -> float:
    """
    Calculate M-squared alpha
    
    Args:
        portfolio_return: Portfolio return (as a decimal)
        risk_free_rate: Risk-free rate (as a decimal)
        portfolio_stddev: Portfolio standard deviation
        market_stddev: Market standard deviation
        market_return: Market return (as a decimal)
    
    Returns:
        M-squared alpha
    """
    m2 = ((portfolio_return - risk_free_rate) * (market_stddev / portfolio_stddev)) + risk_free_rate
    return m2 - market_return


def jensen_alpha(portfolio_return: float, risk_free_rate: float, portfolio_beta: float, 
               market_return: float) -> float:
    """
    Calculate Jensen's alpha
    
    Args:
        portfolio_return: Portfolio return (as a decimal)
        risk_free_rate: Risk-free rate (as a decimal)
        portfolio_beta: Portfolio beta
        market_return: Market return (as a decimal)
    
    Returns:
        Jensen's alpha
    """
    return portfolio_return - (risk_free_rate + portfolio_beta * (market_return - risk_free_rate))


def information_ratio(alpha: float, tracking_error: float) -> float:
    """
    Calculate information ratio
    
    Args:
        alpha: Alpha (excess return over benchmark)
        tracking_error: Tracking error (standard deviation of the difference in returns)
    
    Returns:
        Information ratio
    """
    return alpha / tracking_error


### TECHNICAL ANALYSIS ###

def head_and_shoulders_target_price(neckline: float, head: float) -> float:
    """
    Calculate target price for head and shoulders pattern
    
    Args:
        neckline: Neckline level
        head: Head level
    
    Returns:
        Target price
    """
    return neckline - (head - neckline)


def inverse_head_and_shoulders_target_price(neckline: float, head: float) -> float:
    """
    Calculate target price for inverse head and shoulders pattern
    
    Args:
        neckline: Neckline level
        head: Head level
    
    Returns:
        Target price
    """
    return neckline + (neckline - head)


def double_top_target_price(valley_low: float, valley_high: float) -> float:
    """
    Calculate target price for double top pattern
    
    Args:
        valley_low: Valley low level
        valley_high: Valley high level
    
    Returns:
        Target price
    """
    return valley_low - (valley_high - valley_low)


def macaulay_duration_coupon_bond(coupon_rate: float, face_value: float, ytm: float, 
                                periods_to_maturity: float, payment_frequency: int = 1) -> float:
    """
    Calculate Macaulay duration for a coupon bond
    
    Args:
        coupon_rate: Annual coupon rate (as a decimal)
        face_value: Face value
        ytm: Yield to maturity (as a decimal)
        periods_to_maturity: Periods to maturity
        payment_frequency: Payment frequency (e.g., 2 for semiannual)
    
    Returns:
        Macaulay duration
    """
    # Adjust rates for payment frequency
    periodic_coupon_rate = coupon_rate / payment_frequency
    periodic_ytm = ytm / payment_frequency
    
    # Calculate coupon payment per period
    coupon_payment = periodic_coupon_rate * face_value
    
    # Calculate present value factors
    pvf_coupons = (1 - (1 + periodic_ytm) ** (-periods_to_maturity)) / periodic_ytm
    pvf_principal = (1 + periodic_ytm) ** (-periods_to_maturity)
    
    # Calculate bond price
    bond_price = coupon_payment * pvf_coupons + face_value * pvf_principal
    
    # Calculate duration
    duration = ((coupon_payment * pvf_coupons - coupon_payment * periods_to_maturity * 
                pvf_principal) / periodic_ytm + 
               periods_to_maturity * face_value * pvf_principal) / bond_price
    
    # Convert to years
    return duration / payment_frequency


def modified_duration(macaulay_duration: float, ytm: float, 
                    payment_frequency: int = 1) -> float:
    """
    Calculate modified duration
    
    Args:
        macaulay_duration: Macaulay duration
        ytm: Yield to maturity (as a decimal)
        payment_frequency: Payment frequency (e.g., 2 for semiannual)
    
    Returns:
        Modified duration
    """
    return macaulay_duration / (1 + (ytm / payment_frequency))


def approximate_modified_duration(price_minus: float, price_plus: float, price0: float, 
                                yield_change: float) -> float:
    """
    Calculate approximate modified duration
    
    Args:
        price_minus: Bond price after yield decrease
        price_plus: Bond price after yield increase
        price0: Initial bond price
        yield_change: Change in yield (in decimal form)
    
    Returns:
        Approximate modified duration
    """
    return (price_minus - price_plus) / (2 * yield_change * price0)


def effective_duration(price_minus: float, price_plus: float, price0: float, 
                     curve_shift: float) -> float:
    """
    Calculate effective duration
    
    Args:
        price_minus: Bond price after yield curve decrease
        price_plus: Bond price after yield curve increase
        price0: Initial bond price
        curve_shift: Shift in yield curve (in decimal form)
    
    Returns:
        Effective duration
    """
    return (price_minus - price_plus) / (2 * curve_shift * price0)


def portfolio_duration(weights: List[float], durations: List[float]) -> float:
    """
    Calculate portfolio duration
    
    Args:
        weights: List of market value weights
        durations: List of durations for each component
    
    Returns:
        Portfolio duration
    """
    return sum(w * d for w, d in zip(weights, durations))


def money_duration(modified_duration: float, full_price: float) -> float:
    """
    Calculate money duration (dollar duration)
    
    Args:
        modified_duration: Modified duration
        full_price: Full price of the bond
    
    Returns:
        Money duration
    """
    return modified_duration * full_price


def price_change_percentage(modified_duration: float, yield_change: float) -> float:
    """
    Estimate percentage price change using duration
    
    Args:
        modified_duration: Modified duration
        yield_change: Change in yield (in decimal form)
    
    Returns:
        Estimated percentage price change
    """
    return -modified_duration * yield_change


def price_change_money(money_duration: float, yield_change: float) -> float:
    """
    Estimate price change in money terms using duration
    
    Args:
        money_duration: Money duration (dollar duration)
        yield_change: Change in yield (in decimal form)
    
    Returns:
        Estimated price change in money terms
    """
    return -money_duration * yield_change


def approximate_convexity(price_minus: float, price_plus: float, price0: float, 
                        yield_change: float) -> float:
    """
    Calculate approximate convexity
    
    Args:
        price_minus: Bond price after yield decrease
        price_plus: Bond price after yield increase
        price0: Initial bond price
        yield_change: Change in yield (in decimal form)
    
    Returns:
        Approximate convexity
    """
    return (price_minus + price_plus - 2 * price0) / ((yield_change ** 2) * price0)


def effective_convexity(price_minus: float, price_plus: float, price0: float, 
                      curve_shift: float) -> float:
    """
    Calculate effective convexity
    
    Args:
        price_minus: Bond price after yield curve decrease
        price_plus: Bond price after yield curve increase
        price0: Initial bond price
        curve_shift: Shift in yield curve (in decimal form)
    
    Returns:
        Effective convexity
    """
    return (price_minus + price_plus - 2 * price0) / ((curve_shift ** 2) * price0)


def price_change_with_convexity(modified_duration: float, convexity: float, 
                              yield_change: float, full_price: float) -> float:
    """
    Estimate price change using duration and convexity
    
    Args:
        modified_duration: Modified duration
        convexity: Convexity
        yield_change: Change in yield (in decimal form)
        full_price: Full price of the bond
    
    Returns:
        Estimated price change
    """
    duration_effect = -modified_duration * yield_change * full_price
    convexity_effect = 0.5 * convexity * (yield_change ** 2) * full_price
    
    return duration_effect + convexity_effect


def duration_gap(macaulay_duration: float, investment_horizon: float) -> float:
    """
    Calculate duration gap
    
    Args:
        macaulay_duration: Macaulay duration
        investment_horizon: Investment horizon
    
    Returns:
        Duration gap
    """
    return macaulay_duration - investment_horizon


###############################
# DERIVATIVES
###############################

### DERIVATIVE MARKET AND INSTRUMENTS ###

def long_call_payoff(spot_price: float, strike_price: float) -> float:
    """
    Calculate payoff of a long call option at expiration
    
    Args:
        spot_price: Spot price of the underlying asset at expiration
        strike_price: Strike price of the option
    
    Returns:
        Payoff at expiration
    """
    return max(0, spot_price - strike_price)


def long_call_profit(spot_price: float, strike_price: float, premium: float) -> float:
    """
    Calculate profit of a long call option at expiration
    
    Args:
        spot_price: Spot price of the underlying asset at expiration
        strike_price: Strike price of the option
        premium: Option premium (cost)
    
    Returns:
        Profit at expiration
    """
    return max(0, spot_price - strike_price) - premium


def short_call_payoff(spot_price: float, strike_price: float) -> float:
    """
    Calculate payoff of a short call option at expiration
    
    Args:
        spot_price: Spot price of the underlying asset at expiration
        strike_price: Strike price of the option
    
    Returns:
        Payoff at expiration
    """
    return -max(0, spot_price - strike_price)


def short_call_profit(spot_price: float, strike_price: float, premium: float) -> float:
    """
    Calculate profit of a short call option at expiration
    
    Args:
        spot_price: Spot price of the underlying asset at expiration
        strike_price: Strike price of the option
        premium: Option premium (received)
    
    Returns:
        Profit at expiration
    """
    return -max(0, spot_price - strike_price) + premium


def long_put_payoff(spot_price: float, strike_price: float) -> float:
    """
    Calculate payoff of a long put option at expiration
    
    Args:
        spot_price: Spot price of the underlying asset at expiration
        strike_price: Strike price of the option
    
    Returns:
        Payoff at expiration
    """
    return max(0, strike_price - spot_price)


def long_put_profit(spot_price: float, strike_price: float, premium: float) -> float:
    """
    Calculate profit of a long put option at expiration
    
    Args:
        spot_price: Spot price of the underlying asset at expiration
        strike_price: Strike price of the option
        premium: Option premium (cost)
    
    Returns:
        Profit at expiration
    """
    return max(0, strike_price - spot_price) - premium


def short_put_payoff(spot_price: float, strike_price: float) -> float:
    """
    Calculate payoff of a short put option at expiration
    
    Args:
        spot_price: Spot price of the underlying asset at expiration
        strike_price: Strike price of the option
    
    Returns:
        Payoff at expiration
    """
    return -max(0, strike_price - spot_price)


def short_put_profit(spot_price: float, strike_price: float, premium: float) -> float:
    """
    Calculate profit of a short put option at expiration
    
    Args:
        spot_price: Spot price of the underlying asset at expiration
        strike_price: Strike price of the option
        premium: Option premium (received)
    
    Returns:
        Profit at expiration
    """
    return -max(0, strike_price - spot_price) + premium


### BASICS OF DERIVATIVE PRICING AND VALUATION ###

def risky_asset_pricing(expected_value: float, risk_premium: float, 
                      risk_free_rate: float, time_period: float) -> float:
    """
    Calculate price of a risky asset
    
    Args:
        expected_value: Expected future value
        risk_premium: Risk premium (as a decimal)
        risk_free_rate: Risk-free rate (as a decimal)
        time_period: Time period (in years)
    
    Returns:
        Risky asset price
    """
    return expected_value / ((1 + risk_free_rate + risk_premium) ** time_period)


def forward_contract_price(spot_price: float, benefits_pv: float, costs_pv: float, 
                         risk_free_rate: float, time_period: float) -> float:
    """
    Calculate forward contract price
    
    Args:
        spot_price: Spot price of the underlying asset
        benefits_pv: Present value of benefits
        costs_pv: Present value of costs
        risk_free_rate: Risk-free rate (as a decimal)
        time_period: Time period (in years)
    
    Returns:
        Forward contract price
    """
    return (spot_price - benefits_pv + costs_pv) * ((1 + risk_free_rate) ** time_period)


def forward_contract_value_during_life(spot_price: float, forward_price: float, 
                                     benefits_t: float, costs_t: float, 
                                     risk_free_rate: float, time_remaining: float) -> float:
    """
    Calculate value of a forward contract during its life
    
    Args:
        spot_price: Current spot price of the underlying asset
        forward_price: Original forward price
        benefits_t: Benefits at time t
        costs_t: Costs at time t
        risk_free_rate: Risk-free rate (as a decimal)
        time_remaining: Time remaining until expiration (in years)
    
    Returns:
        Forward contract value
    """
    return spot_price - benefits_t + costs_t - (forward_price / ((1 + risk_free_rate) ** time_remaining))


def intrinsic_value_call(spot_price: float, strike_price: float) -> float:
    """
    Calculate intrinsic value of a call option
    
    Args:
        spot_price: Spot price of the underlying asset
        strike_price: Strike price of the option
    
    Returns:
        Intrinsic value
    """
    return max(0, spot_price - strike_price)


def intrinsic_value_put(spot_price: float, strike_price: float) -> float:
    """
    Calculate intrinsic value of a put option
    
    Args:
        spot_price: Spot price of the underlying asset
        strike_price: Strike price of the option
    
    Returns:
        Intrinsic value
    """
    return max(0, strike_price - spot_price)


def option_time_value(option_price: float, intrinsic_value: float) -> float:
    """
    Calculate time value of an option
    
    Args:
        option_price: Market price of the option
        intrinsic_value: Intrinsic value of the option
    
    Returns:
        Time value
    """
    return option_price - intrinsic_value


def put_call_parity(call_price: float, put_price: float, spot_price: float, 
                  strike_price: float, risk_free_rate: float, time_to_expiry: float) -> bool:
    """
    Check if put-call parity holds
    
    Args:
        call_price: Call option price
        put_price: Put option price
        spot_price: Spot price of the underlying asset
        strike_price: Strike price of the options
        risk_free_rate: Risk-free rate (as a decimal)
        time_to_expiry: Time to expiration (in years)
    
    Returns:
        True if parity holds, False otherwise
    """
    lhs = call_price + (strike_price / ((1 + risk_free_rate) ** time_to_expiry))
    rhs = put_price + spot_price
    
    # Allow for small floating point differences
    return abs(lhs - rhs) < 1e-10


def put_call_forward_parity(call_price: float, put_price: float, forward_price: float, 
                          strike_price: float, risk_free_rate: float, time_to_expiry: float) -> bool:
    """
    Check if put-call-forward parity holds
    
    Args:
        call_price: Call option price
        put_price: Put option price
        forward_price: Forward price
        strike_price: Strike price of the options
        risk_free_rate: Risk-free rate (as a decimal)
        time_to_expiry: Time to expiration (in years)
    
    Returns:
        True if parity holds, False otherwise
    """
    lhs = call_price + (strike_price / ((1 + risk_free_rate) ** time_to_expiry))
    rhs = put_price + (forward_price / ((1 + risk_free_rate) ** time_to_expiry))
    
    # Allow for small floating point differences
    return abs(lhs - rhs) < 1e-10


def synthetic_call(put_price: float, stock_price: float, strike_price: float, 
                 risk_free_rate: float, time_to_expiry: float) -> float:
    """
    Calculate synthetic call price using put-call parity
    
    Args:
        put_price: Put option price
        stock_price: Stock price
        strike_price: Strike price of the options
        risk_free_rate: Risk-free rate (as a decimal)
        time_to_expiry: Time to expiration (in years)
    
    Returns:
        Synthetic call price
    """
    return put_price + stock_price - (strike_price / ((1 + risk_free_rate) ** time_to_expiry))


def synthetic_put(call_price: float, strike_price: float, stock_price: float, 
                risk_free_rate: float, time_to_expiry: float) -> float:
    """
    Calculate synthetic put price using put-call parity
    
    Args:
        call_price: Call option price
        strike_price: Strike price of the options
        stock_price: Stock price
        risk_free_rate: Risk-free rate (as a decimal)
        time_to_expiry: Time to expiration (in years)
    
    Returns:
        Synthetic put price
    """
    return call_price + (strike_price / ((1 + risk_free_rate) ** time_to_expiry)) - stock_price


def lower_bound_european_call(stock_price: float, strike_price: float, 
                            risk_free_rate: float, time_to_expiry: float) -> float:
    """
    Calculate lower bound of a European call option
    
    Args:
        stock_price: Stock price
        strike_price: Strike price of the option
        risk_free_rate: Risk-free rate (as a decimal)
        time_to_expiry: Time to expiration (in years)
    
    Returns:
        Lower bound of European call option
    """
    return max(0, stock_price - (strike_price / ((1 + risk_free_rate) ** time_to_expiry)))


def lower_bound_european_put(stock_price: float, strike_price: float, 
                           risk_free_rate: float, time_to_expiry: float) -> float:
    """
    Calculate lower bound of a European put option
    
    Args:
        stock_price: Stock price
        strike_price: Strike price of the option
        risk_free_rate: Risk-free rate (as a decimal)
        time_to_expiry: Time to expiration (in years)
    
    Returns:
        Lower bound of European put option
    """
    return max(0, (strike_price / ((1 + risk_free_rate) ** time_to_expiry)) - stock_price)


def lower_bound_american_put(stock_price: float, strike_price: float) -> float:
    """
    Calculate lower bound of an American put option
    
    Args:
        stock_price: Stock price
        strike_price: Strike price of the option
    
    Returns:
        Lower bound of American put option
    """
    return max(0, strike_price - stock_price)


def binomial_call_price(up_price: float, down_price: float, risk_free_rate: float, 
                      up_prob: float, strike_price: float) -> float:
    """
    Calculate call option price using one-period binomial model
    
    Args:
        up_price: Up-state price of the underlying asset
        down_price: Down-state price of the underlying asset
        risk_free_rate: Risk-free rate (as a decimal)
        up_prob: Risk-neutral probability of up-state
        strike_price: Strike price of the option
    
    Returns:
        Call option price
    """
    call_up = max(0, up_price - strike_price)
    call_down = max(0, down_price - strike_price)
    
    return (up_prob * call_up + (1 - up_prob) * call_down) / (1 + risk_free_rate)


def binomial_put_price(up_price: float, down_price: float, risk_free_rate: float, 
                     up_prob: float, strike_price: float) -> float:
    """
    Calculate put option price using one-period binomial model
    
    Args:
        up_price: Up-state price of the underlying asset
        down_price: Down-state price of the underlying asset
        risk_free_rate: Risk-free rate (as a decimal)
        up_prob: Risk-neutral probability of up-state
        strike_price: Strike price of the option
    
    Returns:
        Put option price
    """
    put_up = max(0, strike_price - up_price)
    put_down = max(0, strike_price - down_price)
    
    return (up_prob * put_up + (1 - up_prob) * put_down) / (1 + risk_free_rate)


def risk_neutral_probability(risk_free_rate: float, up_factor: float, down_factor: float) -> float:
    """
    Calculate risk-neutral probability in binomial model
    
    Args:
        risk_free_rate: Risk-free rate (as a decimal)
        up_factor: Up-state factor
        down_factor: Down-state factor
    
    Returns:
        Risk-neutral probability of up-state
    """
    return (1 + risk_free_rate - down_factor) / (up_factor - down_factor)


def hedge_ratio(up_call_price: float, down_call_price: float, up_stock_price: float, 
              down_stock_price: float) -> float:
    """
    Calculate hedge ratio for a call option
    
    Args:
        up_call_price: Call option price in up-state
        down_call_price: Call option price in down-state
        up_stock_price: Stock price in up-state
        down_stock_price: Stock price in down-state
    
    Returns:
        Hedge ratio
    """
    return (up_call_price - down_call_price) / (up_stock_price - down_stock_price)


###############################
# ALTERNATIVE INVESTMENTS
###############################

### INTRODUCTION TO ALTERNATIVE INVESTMENTS ###

def management_fee_beginning_market_value(management_fee_pct: float, 
                                        beginning_market_value: float) -> float:
    """
    Calculate management fee based on beginning market value
    
    Args:
        management_fee_pct: Management fee percentage (as a decimal)
        beginning_market_value: Beginning market value
    
    Returns:
        Management fee
    """
    return management_fee_pct * beginning_market_value


def management_fee_ending_market_value(management_fee_pct: float, 
                                     ending_market_value: float) -> float:
    """
    Calculate management fee based on ending market value
    
    Args:
        management_fee_pct: Management fee percentage (as a decimal)
        ending_market_value: Ending market value
    
    Returns:
        Management fee
    """
    return management_fee_pct * ending_market_value


def incentive_fee_independent(incentive_fee_pct: float, gain: float) -> float:
    """
    Calculate incentive fee independent of management fee
    
    Args:
        incentive_fee_pct: Incentive fee percentage (as a decimal)
        gain: Gain
    
    Returns:
        Incentive fee
    """
    if gain > 0:
        return incentive_fee_pct * gain
    else:
        return 0


def incentive_fee_net_of_management_fee(incentive_fee_pct: float, gain: float, 
                                      management_fee: float) -> float:
    """
    Calculate incentive fee net of management fee
    
    Args:
        incentive_fee_pct: Incentive fee percentage (as a decimal)
        gain: Gain
        management_fee: Management fee
    
    Returns:
        Incentive fee
    """
    net_gain = gain - management_fee
    if net_gain > 0:
        return incentive_fee_pct * net_gain
    else:
        return 0


def incentive_fee_hard_hurdle_independent(incentive_fee_pct: float, gain: float, 
                                        hurdle: float) -> float:
    """
    Calculate incentive fee with hard hurdle (independent of management fee)
    
    Args:
        incentive_fee_pct: Incentive fee percentage (as a decimal)
        gain: Gain
        hurdle: Hurdle
    
    Returns:
        Incentive fee
    """
    excess_gain = gain - hurdle
    if excess_gain > 0:
        return incentive_fee_pct * excess_gain
    else:
        return 0


def incentive_fee_hard_hurdle_net_of_management_fee(incentive_fee_pct: float, gain: float, 
                                                 management_fee: float, hurdle: float) -> float:
    """
    Calculate incentive fee with hard hurdle (net of management fee)
    
    Args:
        incentive_fee_pct: Incentive fee percentage (as a decimal)
        gain: Gain
        management_fee: Management fee
        hurdle: Hurdle
    
    Returns:
        Incentive fee
    """
    net_gain = gain - management_fee
    excess_gain = net_gain - hurdle
    if excess_gain > 0:
        return incentive_fee_pct * excess_gain
    else:
        return 0


def real_estate_income_approach(expected_noi: float, cap_rate: float) -> float:
    """
    Calculate property value using income approach
    
    Args:
        expected_noi: Expected annual net operating income (NOI)
        cap_rate: Capitalization rate (as a decimal)
    
    Returns:
        Property value
    """
    return expected_noi / cap_rate


def real_estate_gordon_growth_model(noi1: float, required_return: float, growth_rate: float) -> float:
    """
    Calculate property value using Gordon Growth Model
    
    Args:
        noi1: Next period's net operating income (NOI)
        required_return: Required rate of return (as a decimal)
        growth_rate: Expected growth rate of NOI (as a decimal)
    
    Returns:
        Property value
    """
    return noi1 / (required_return - growth_rate)


###############################
# PORTFOLIO MANAGEMENT
###############################

### PORTFOLIO RISK AND RETURN: PART 1 ###

def holding_period_return(price_end: float, price_begin: float, income: float) -> float:
    """
    Calculate holding period return
    
    Args:
        price_end: Ending price
        price_begin: Beginning price
        income: Income received during the period
    
    Returns:
        Holding period return (as a decimal)
    """
    return (price_end - price_begin + income) / price_begin


def real_return(nominal_return: float, inflation_rate: float) -> float:
    """
    Calculate real return
    
    Args:
        nominal_return: Nominal return (as a decimal)
        inflation_rate: Inflation rate (as a decimal)
    
    Returns:
        Real return (as a decimal)
    """
    return (1 + nominal_return) / (1 + inflation_rate) - 1


def utility_function(expected_return: float, variance: float, risk_aversion: float) -> float:
    """
    Calculate utility using mean-variance utility function
    
    Args:
        expected_return: Expected return (as a decimal)
        variance: Variance (as a decimal)
        risk_aversion: Measure of risk aversion
    
    Returns:
        Utility
    """
    return expected_return - 0.5 * risk_aversion * variance


def capital_allocation_line(risk_free_rate: float, portfolio_return: float, 
                          portfolio_stddev: float, portfolio_allocation_stddev: float) -> float:
    """
    Calculate expected return using capital allocation line
    
    Args:
        risk_free_rate: Risk-free rate (as a decimal)
        portfolio_return: Expected return of risky portfolio (as a decimal)
        portfolio_stddev: Standard deviation of risky portfolio
        portfolio_allocation_stddev: Standard deviation of allocation portfolio
    
    Returns:
        Expected return of allocation portfolio
    """
    return risk_free_rate + ((portfolio_return - risk_free_rate) / portfolio_stddev) * portfolio_allocation_stddev


def portfolio_expected_return(weights: List[float], returns: List[float]) -> float:
    """
    Calculate portfolio expected return
    
    Args:
        weights: List of weights
        returns: List of expected returns
    
    Returns:
        Portfolio expected return
    """
    return sum(w * r for w, r in zip(weights, returns))


def portfolio_variance_two_assets(weight1: float, stddev1: float, weight2: float, 
                                stddev2: float, correlation: float) -> float:
    """
    Calculate portfolio variance for two assets
    
    Args:
        weight1: Weight of asset 1
        stddev1: Standard deviation of asset 1
        weight2: Weight of asset 2
        stddev2: Standard deviation of asset 2
        correlation: Correlation between assets
    
    Returns:
        Portfolio variance
    """
    return (weight1 ** 2) * (stddev1 ** 2) + (weight2 ** 2) * (stddev2 ** 2) + 2 * weight1 * weight2 * correlation * stddev1 * stddev2


def portfolio_std_dev(variance: float) -> float:
    """
    Calculate portfolio standard deviation
    
    Args:
        variance: Portfolio variance
    
    Returns:
        Portfolio standard deviation
    """
    return math.sqrt(variance)


def portfolio_variance_many_assets(average_variance: float, average_correlation: float, 
                                 n: int) -> float:
    """
    Calculate portfolio variance for many equally weighted assets
    
    Args:
        average_variance: Average variance of all assets
        average_correlation: Average correlation between assets
        n: Number of assets
    
    Returns:
        Portfolio variance
    """
    return (average_variance / n) + ((n - 1) / n) * average_correlation * average_variance


def new_asset_condition(new_return: float, new_stddev: float, portfolio_return: float, 
                      portfolio_stddev: float, correlation: float) -> bool:
    """
    Check condition for adding a new asset to portfolio
    
    Args:
        new_return: Expected return of new asset
        new_stddev: Standard deviation of new asset
        portfolio_return: Expected return of current portfolio
        portfolio_stddev: Standard deviation of current portfolio
        correlation: Correlation between new asset and current portfolio
    
    Returns:
        True if condition is met, False otherwise
    """
    lhs = (new_return - portfolio_return) / new_stddev
    rhs = ((portfolio_return - 0) / portfolio_stddev) * correlation  # Assuming risk-free rate is 0
    
    return lhs > rhs


### PORTFOLIO RISK AND RETURN: PART II ###

def capital_market_line(risk_free_rate: float, market_return: float, market_stddev: float, 
                      portfolio_stddev: float) -> float:
    """
    Calculate expected return using capital market line
    
    Args:
        risk_free_rate: Risk-free rate (as a decimal)
        market_return: Expected return of market portfolio (as a decimal)
        market_stddev: Standard deviation of market portfolio
        portfolio_stddev: Standard deviation of portfolio
    
    Returns:
        Expected return of portfolio    Calculate cost of equity using CAPM with country risk premium
    
    Args:
        risk_free_rate: Risk-free rate (as a decimal)
        beta: Beta of the stock
        market_risk_premium: Market risk premium (Rm - Rf)
        country_risk_premium: Country risk premium
    
    Returns:
        Cost of equity
    """
    return risk_free_rate + beta * (market_risk_premium + country_risk_premium)


def break_point_marginal_cost_of_capital(amount: float, proportion: float) -> float:
    """
    Calculate break point for marginal cost of capital schedule
    
    Args:
        amount: Amount of capital at which source's cost of capital changes
        proportion: Proportion of new capital raised from the source
    
    Returns:
        Break point
    """
    return amount / proportion


def cost_of_equity_with_flotation_absolute(dividend: float, price: float, flotation_cost: float, 
                                        growth_rate: float) -> float:
    """
    Calculate cost of equity with flotation cost (absolute amount)
    
    Args:
        dividend: Next period's dividend per share (D₁)
        price: Current stock price per share
        flotation_cost: Flotation cost (absolute amount)
        growth_rate: Expected dividend growth rate
    
    Returns:
        Cost of equity
    """
    return (dividend / (price - flotation_cost)) + growth_rate


def cost_of_equity_with_flotation_percentage(dividend: float, price: float, flotation_rate: float, 
                                          growth_rate: float) -> float:
    """
    Calculate cost of equity with flotation cost (percentage)
    
    Args:
        dividend: Next period's dividend per share (D₁)
        price: Current stock price per share
        flotation_rate: Flotation cost as a percentage of price
        growth_rate: Expected dividend growth rate
    
    Returns:
        Cost of equity
    """
    return (dividend / (price * (1 - flotation_rate))) + growth_rate


### MEASURES OF LEVERAGE ###

def degree_of_operating_leverage(price: float, variable_cost: float, quantity: float, 
                               fixed_cost: float) -> float:
    """
    Calculate degree of operating leverage (DOL)
    
    Args:
        price: Price per unit
        variable_cost: Variable operating cost per unit
        quantity: Number of units sold
        fixed_cost: Fixed operating cost
    
    Returns:
        Degree of operating leverage
    """
    return (quantity * (price - variable_cost)) / (quantity * (price - variable_cost) - fixed_cost)


def degree_of_financial_leverage(price: float, variable_cost: float, quantity: float, 
                               fixed_cost: float, interest: float) -> float:
    """
    Calculate degree of financial leverage (DFL)
    
    Args:
        price: Price per unit
        variable_cost: Variable operating cost per unit
        quantity: Number of units sold
        fixed_cost: Fixed operating cost
        interest: Fixed financial cost (interest)
    
    Returns:
        Degree of financial leverage
    """
    return (quantity * (price - variable_cost) - fixed_cost) / (quantity * (price - variable_cost) - fixed_cost - interest)


def degree_of_total_leverage(price: float, variable_cost: float, quantity: float, 
                           fixed_cost: float, interest: float) -> float:
    """
    Calculate degree of total leverage (DTL)
    
    Args:
        price: Price per unit
        variable_cost: Variable operating cost per unit
        quantity: Number of units sold
        fixed_cost: Fixed operating cost
        interest: Fixed financial cost (interest)
    
    Returns:
        Degree of total leverage
    """
    return (quantity * (price - variable_cost)) / (quantity * (price - variable_cost) - fixed_cost - interest)


def break_even_units(fixed_cost: float, interest: float, price: float, variable_cost: float) -> float:
    """
    Calculate break-even units (net income = 0)
    
    Args:
        fixed_cost: Fixed operating cost
        interest: Fixed financial cost (interest)
        price: Price per unit
        variable_cost: Variable operating cost per unit
    
    Returns:
        Break-even units
    """
    return (fixed_cost + interest) / (price - variable_cost)


def operating_break_even_units(fixed_cost: float, price: float, variable_cost: float) -> float:
    """
    Calculate operating break-even units (operating income = 0)
    
    Args:
        fixed_cost: Fixed operating cost
        price: Price per unit
        variable_cost: Variable operating cost per unit
    
    Returns:
        Operating break-even units
    """
    return fixed_cost / (price - variable_cost)


### WORKING CAPITAL MANAGEMENT ###

def operating_cycle(days_inventory: float, days_receivables: float) -> float:
    """
    Calculate operating cycle
    
    Args:
        days_inventory: Number of days of inventory
        days_receivables: Number of days of receivables
    
    Returns:
        Operating cycle
    """
    return days_inventory + days_receivables


def net_operating_cycle(days_inventory: float, days_receivables: float, days_payables: float) -> float:
    """
    Calculate net operating cycle (cash conversion cycle)
    
    Args:
        days_inventory: Number of days of inventory
        days_receivables: Number of days of receivables
        days_payables: Number of days of payables
    
    Returns:
        Net operating cycle
    """
    return days_inventory + days_receivables - days_payables


def float_factor(avg_daily_float: float, avg_daily_deposit: float) -> float:
    """
    Calculate float factor
    
    Args:
        avg_daily_float: Average daily float
        avg_daily_deposit: Average daily deposit
    
    Returns:
        Float factor
    """
    return avg_daily_float / avg_daily_deposit


def cost_of_trade_credit(discount_percentage: float, days_beyond_discount: float) -> float:
    """
    Calculate cost of trade credit
    
    Args:
        discount_percentage: Discount percentage (as a decimal)
        days_beyond_discount: Days beyond discount period
    
    Returns:
        Cost of trade credit (annualized)
    """
    return ((1 + discount_percentage / (1 - discount_percentage)) ** (365 / days_beyond_discount)) - 1


def money_market_yield(face_value: float, purchase_price: float, days_to_maturity: float) -> float:
    """
    Calculate money market yield
    
    Args:
        face_value: Face value
        purchase_price: Purchase price
        days_to_maturity: Days to maturity
    
    Returns:
        Money market yield (annualized)
    """
    return ((face_value - purchase_price) / purchase_price) * (360 / days_to_maturity)


def bond_equivalent_yield(face_value: float, purchase_price: float, days_to_maturity: float) -> float:
    """
    Calculate bond equivalent yield
    
    Args:
        face_value: Face value
        purchase_price: Purchase price
        days_to_maturity: Days to maturity
    
    Returns:
        Bond equivalent yield (annualized)
    """
    return ((face_value - purchase_price) / purchase_price) * (365 / days_to_maturity)


def discount_basis_yield(face_value: float, purchase_price: float, days_to_maturity: float) -> float:
    """
    Calculate discount basis yield
    
    Args:
        face_value: Face value
        purchase_price: Purchase price
        days_to_maturity: Days to maturity
    
    Returns:
        Discount basis yield (annualized)
    """
    return ((face_value - purchase_price) / face_value) * (360 / days_to_maturity)


def line_of_credit_cost(interest: float, commitment_fee: float, loan_amount: float) -> float:
    """
    Calculate cost of borrowing using line of credit
    
    Args:
        interest: Interest
        commitment_fee: Commitment fee
        loan_amount: Loan amount
    
    Returns:
        Cost of borrowing
    """
    return (interest + commitment_fee) / loan_amount


def all_inclusive_cost(interest: float, loan_amount: float) -> float:
    """
    Calculate all-inclusive cost of borrowing
    
    Args:
        interest: Interest
        loan_amount: Loan amount (net of interest)
    
    Returns:
        All-inclusive cost
    """
    return interest / (loan_amount - interest)


def commercial_paper_cost(interest: float, dealers_commission: float, backup_costs: float, 
                        loan_amount: float) -> float:
    """
    Calculate cost of commercial paper
    
    Args:
        interest: Interest
        dealers_commission: Dealers commission
        backup_costs: Backup costs
        loan_amount: Loan amount (net of interest)
    
    Returns:
        Cost of commercial paper
    """
    return (interest + dealers_commission + backup_costs) / (loan_amount - interest)


###############################
# EQUITY INVESTMENTS
###############################

### MARKET ORGANIZATION AND STRUCTURE ###

def maximum_leverage_ratio(minimum_margin_requirement: float) -> float:
    """
    Calculate maximum leverage ratio
    
    Args:
        minimum_margin_requirement: Minimum margin requirement (as a decimal)
    
    Returns:
        Maximum leverage ratio
    """
    return 1 / minimum_margin_requirement


def total_return_leveraged_stock(sales_proceeds: float, dividends: float, loan: float, 
                              margin_interest: float, sales_commission: float, 
                              initial_equity: float, purchase_commission: float) -> float:
    """
    Calculate total return on leveraged stock investment
    
    Args:
        sales_proceeds: Sales proceeds
        dividends: Dividends
        loan: Loan
        margin_interest: Margin interest
        sales_commission: Sales commission
        initial_equity: Initial equity
        purchase_commission: Purchase commission
    
    Returns:
        Total return on leveraged stock investment
    """
    return ((sales_proceeds + dividends - loan - margin_interest - sales_commission) / 
            (initial_equity + purchase_commission)) - 1


def initial_equity(minimum_margin_requirement: float, total_purchase_price: float) -> float:
    """
    Calculate initial equity
    
    Args:
        minimum_margin_requirement: Minimum margin requirement (as a decimal)
        total_purchase_price: Total purchase price
    
    Returns:
        Initial equity
    """
    return minimum_margin_requirement * total_purchase_price


def margin_call_price(current_price: float, initial_margin: float, 
                    maintenance_margin: float) -> float:
    """
    Calculate margin call price
    
    Args:
        current_price: Current price
        initial_margin: Initial margin (as a decimal)
        maintenance_margin: Maintenance margin (as a decimal)
    
    Returns:
        Margin call price
    """
    return (current_price * (1 - initial_margin)) / (1 - maintenance_margin)


### SECURITY MARKET INDICES ###

def price_return_index(constituent_prices: List[float], constituent_quantities: List[float], 
                     divisor: float) -> float:
    """
    Calculate price return index
    
    Args:
        constituent_prices: List of unit prices of constituent securities
        constituent_quantities: List of quantities of constituent securities held in index
        divisor: Value of the divisor
    
    Returns:
        Price return index
    """
    numerator = sum(p * q for p, q in zip(constituent_prices, constituent_quantities))
    return numerator / divisor


def price_return(index_value_end: float, index_value_beginning: float) -> float:
    """
    Calculate price return of an index
    
    Args:
        index_value_end: Value of the price return index at the end of the period
        index_value_beginning: Value of the price return index at the beginning of the period
    
    Returns:
        Price return
    """
    return (index_value_end - index_value_beginning) / index_value_beginning


def total_return_index(index_value_end: float, index_value_beginning: float, income: float) -> float:
    """
    Calculate total return index
    
    Args:
        index_value_end: Value of the price return index at the end of the period
        index_value_beginning: Value of the price return index at the beginning of the period
        income: Total income (dividends and/or interest) from all securities in the index
    
    Returns:
        Total return index
    """
    return (index_value_end - index_value_beginning + income) / index_value_beginning


def price_weighting(price: float, all_prices: List[float]) -> float:
    """
    Calculate price weighting
    
    Args:
        price: Price of security
        all_prices: List of prices of all securities in the index
    
    Returns:
        Price weight
    """
    return price / sum(all_prices)


def equal_weighting(n: int) -> float:
    """
    Calculate equal weighting
    
    Args:
        n: Number of securities in the index
    
    Returns:
        Equal weight
    """
    return 1 / n


def market_cap_weighting(shares: float, price: float, all_shares: List[float], 
                       all_prices: List[float]) -> float:
    """
    Calculate market capitalization weighting
    
    Args:
        shares: Number of shares outstanding of security
        price: Price of security
        all_shares: List of shares outstanding of all securities in the index
        all_prices: List of prices of all securities in the index
    
    Returns:
        Market capitalization weight
    """
    return (shares * price) / sum(s * p for s, p in zip(all_shares, all_prices))


def float_adjusted_market_cap_weighting(float_factor: float, shares: float, price: float, 
                                      all_float_factors: List[float], all_shares: List[float], 
                                      all_prices: List[float]) -> float:
    """
    Calculate float-adjusted market capitalization weighting
    
    Args:
        float_factor: Float factor (fraction of shares in market float)
        shares: Number of shares outstanding of security
        price: Price of security
        all_float_factors: List of float factors of all securities in the index
        all_shares: List of shares outstanding of all securities in the index
        all_prices: List of prices of all securities in the index
    
    Returns:
        Float-adjusted market capitalization weight
    """
    return (float_factor * shares * price) / sum(f * s * p for f, s, p in zip(all_float_factors, all_shares, all_prices))


def fundamental_weighting(fundamental_measure: float, all_fundamental_measures: List[float]) -> float:
    """
    Calculate fundamental weighting
    
    Args:
        fundamental_measure: Fundamental size measure of company
        all_fundamental_measures: List of fundamental size measures of all companies in the index
    
    Returns:
        Fundamental weight
    """
    return fundamental_measure / sum(all_fundamental_measures)


### OVERVIEW OF EQUITY SECURITIES ###

def return_on_equity_average(net_income: float, beginning_equity: float, ending_equity: float) -> float:
    """
    Calculate return on equity using average total book value of equity
    
    Args:
        net_income: Net income
        beginning_equity: Beginning book value of equity
        ending_equity: Ending book value of equity
    
    Returns:
        Return on equity
    """
    return net_income / ((beginning_equity + ending_equity) / 2)


def return_on_equity_beginning(net_income: float, beginning_equity: float) -> float:
    """
    Calculate return on equity using beginning book value of equity
    
    Args:
        net_income: Net income
        beginning_equity: Beginning book value of equity
    
    Returns:
        Return on equity
    """
    return net_income / beginning_equity


### EQUITY VALUATION: CONCEPTS AND BASIC TOOLS ###

def intrinsic_value(dividends: List[float], terminal_value: float, required_return: float) -> float:
    """
    Calculate intrinsic value of a share at t=0
    
    Args:
        dividends: List of expected dividends in each future year
        terminal_value: Expected price per share at terminal period
        required_return: Required rate of return on stock (as a decimal)
    
    Returns:
        Intrinsic value
    """
    value = 0
    for t, div in enumerate(dividends, 1):
        value += div / ((1 + required_return) ** t)
    
    value += terminal_value / ((1 + required_return) ** len(dividends))
    
    return value


def preferred_stock_value_noncallable_nonconvertible_maturity(dividend: float, 
                                                            required_return: float) -> float:
    """
    Calculate value of preferred stock (non-callable, non-convertible, with maturity)
    
    Args:
        dividend: Preferred stock dividend per share
        required_return: Required rate of return (as a decimal)
    
    Returns:
        Value of preferred stock
    """
    return dividend / required_return


def preferred_stock_value_perpetual(expected_dividends: List[float], terminal_value: float, 
                                  required_return: float) -> float:
    """
    Calculate value of perpetual preferred stock
    
    Args:
        expected_dividends: List of expected dividends in each future year
        terminal_value: Par value (terminal value)
        required_return: Required rate of return (as a decimal)
    
    Returns:
        Value of perpetual preferred stock
    """
    value = 0
    for t, div in enumerate(expected_dividends, 1):
        value += div / ((1 + required_return) ** t)
    
    value += terminal_value / ((1 + required_return) ** len(expected_dividends))
    
    return value


def gordon_growth_model(dividend_next_year: float, required_return: float, growth_rate: float) -> float:
    """
    Calculate stock value using Gordon Growth Model
    
    Args:
        dividend_next_year: Expected dividend in next year (D₁)
        required_return: Required rate of return (as a decimal)
        growth_rate: Expected dividend growth rate (as a decimal)
    
    Returns:
        Stock value
    """
    return dividend_next_year / (required_return - growth_rate)


def gordon_growth_model_with_current_dividend(current_dividend: float, required_return: float, 
                                            growth_rate: float) -> float:
    """
    Calculate stock value using Gordon Growth Model with current dividend
    
    Args:
        current_dividend: Current dividend (D₀)
        required_return: Required rate of return (as a decimal)
        growth_rate: Expected dividend growth rate (as a decimal)
    
    Returns:
        Stock value
    """
    dividend_next_year = current_dividend * (1 + growth_rate)
    return dividend_next_year / (required_return - growth_rate)


def two_stage_dividend_discount_model(current_dividend: float, short_term_growth: float, 
                                    long_term_growth: float, periods: int, 
                                    required_return: float) -> float:
    """
    Calculate stock value using two-stage dividend discount model
    
    Args:
        current_dividend: Current dividend (D₀)
        short_term_growth: Short-term growth rate (as a decimal)
        long_term_growth: Long-term stable growth rate (as a decimal)
        periods: Number of years in high-growth stage
        required_return: Required rate of return (as a decimal)
    
    Returns:
        Stock value
    """
    # Calculate PV of dividends during high growth stage
    value = 0
    for t in range(1, periods + 1):
        dividend = current_dividend * ((1 + short_term_growth) ** t)
        value += dividend / ((1 + required_return) ** t)
    
    # Calculate terminal value using Gordon Growth Model
    dividend_terminal = current_dividend * ((1 + short_term_growth) ** periods) * (1 + long_term_growth)
    terminal_value = dividend_terminal / (required_return - long_term_growth)
    
    # Add PV of terminal value
    value += terminal_value / ((1 + required_return) ** periods)
    
    return value


def justified_forward_pe(dividend_payout_ratio: float, required_return: float, 
                       growth_rate: float) -> float:
    """
    Calculate justified forward P/E ratio
    
    Args:
        dividend_payout_ratio: Dividend payout ratio (1 - retention ratio)
        required_return: Required rate of return (as a decimal)
        growth_rate: Expected growth rate (as a decimal)
    
    Returns:
        Justified forward P/E ratio
    """
    return dividend_payout_ratio / (required_return - growth_rate)


def enterprise_value(equity_market_value: float, preferred_stock_value: float, 
                   debt_market_value: float, cash: float) -> float:
    """
    Calculate enterprise value
    
    Args:
        equity_market_value: Market value of equity
        preferred_stock_value: Market value of preferred stock
        debt_market_value: Market value of debt
        cash: Cash and short-term investments
    
    Returns:
        Enterprise value
    """
    return equity_market_value + preferred_stock_value + debt_market_value - cash


def adjusted_book_value(market_value_assets: float, market_value_liabilities: float) -> float:
    """
    Calculate adjusted book value
    
    Args:
        market_value_assets: Market value of assets
        market_value_liabilities: Market value of liabilities
    
    Returns:
        Adjusted book value
    """
    return market_value_assets - market_value_liabilities


###############################
# FIXED INCOME
###############################

### FIXED-INCOME SECURITIES: DEFINING ELEMENTS ###

def conversion_ratio(par_value: float, conversion_price: float) -> float:
    """
    Calculate conversion ratio for convertible bonds
    
    Args:
        par_value: Par value of the bond
        conversion_price: Conversion price
    
    Returns:
        Conversion ratio
    """
    return par_value / conversion_price


def conversion_value(share_price: float, conversion_ratio: float) -> float:
    """
    Calculate conversion value for convertible bonds
    
    Args:
        share_price: Current share price
        conversion_ratio: Conversion ratio
    
    Returns:
        Conversion value
    """
    return share_price * conversion_ratio


def conversion_premium(bond_price: float, conversion_value: float) -> float:
    """
    Calculate conversion premium for convertible bonds
    
    Args:
        bond_price: Convertible bond price
        conversion_value: Conversion value
    
    Returns:
        Conversion premium
    """
    return bond_price - conversion_value


### INTRODUCTION TO FIXED-INCOME VALUATION ###

def bond_price_zero_coupon(face_value: float, yield_to_maturity: float, time_to_maturity: float) -> float:
    """
    Calculate the price of a zero-coupon bond
    
    Args:
        face_value: Face value
        yield_to_maturity: Yield to maturity (as a decimal)
        time_to_maturity: Time to maturity (in years)
    
    Returns:
        Bond price
    """
    return face_value / ((1 + yield_to_maturity) ** time_to_maturity)


def bond_price_spot_rates(coupon_payments: List[float], face_value: float, 
                        spot_rates: List[float]) -> float:
    """
    Calculate bond price using spot rates
    
    Args:
        coupon_payments: List of coupon payments
        face_value: Face value
        spot_rates: List of spot rates (zero rates) for each period
    
    Returns:
        Bond price
    """
    bond_price = 0
    for t, (coupon, rate) in enumerate(zip(coupon_payments, spot_rates), 1):
        bond_price += coupon / ((1 + rate) ** t)
    
    # Add present value of face value (using the last spot rate)
    bond_price += face_value / ((1 + spot_rates[-1]) ** len(coupon_payments))
    
    return bond_price


def bond_price_full(bond_price_flat: float, accrued_interest: float) -> float:
    """
    Calculate full bond price
    
    Args:
        bond_price_flat: Flat bond price
        accrued_interest: Accrued interest
    
    Returns:
        Full bond price
    """
    return bond_price_flat + accrued_interest


def accrued_interest(coupon: float, days_since_last_coupon: float, 
                   days_in_coupon_period: float) -> float:
    """
    Calculate accrued interest
    
    Args:
        coupon: Full coupon payment
        days_since_last_coupon: Number of days from the last coupon payment to settlement date
        days_in_coupon_period: Number of days in the coupon period
    
    Returns:
        Accrued interest
    """
    return coupon * (days_since_last_coupon / days_in_coupon_period)


def floating_rate_note_value(index_rate: float, quoted_margin: float, discount_margin: float, 
                           face_value: float, periods: int, frequency: int) -> float:
    """
    Calculate value of a floating rate note
    
    Args:
        index_rate: Index rate
        quoted_margin: Quoted margin
        discount_margin: Discount margin
        face_value: Face value
        periods: Number of periods
        frequency: Periodicity (number of payments per year)
    
    Returns:
        Floating rate note value
    """
    value = 0
    for t in range(1, periods + 1):
        # Calculate the t-th coupon payment
        coupon_rate = (index_rate + quoted_margin) / frequency
        coupon_payment = coupon_rate * face_value
        
        # Discount at the rate index + discount margin
        discount_rate = (index_rate + discount_margin) / frequency
        value += coupon_payment / ((1 + discount_rate) ** t)
    
    # Add present value of face value
    value += face_value / ((1 + discount_rate) ** periods)
    
    return value


def convert_periodic_rates(rate_m: float, m: int, rate_n: float, n: int) -> float:
    """
    Convert between periodic rates
    
    Args:
        rate_m: Annual percentage rate with m compounding periods
        m: Number of compounding periods per year (for rate_m)
        rate_n: Annual percentage rate with n compounding periods (output)
        n: Number of compounding periods per year (for rate_n)
    
    Returns:
        Equivalent rate with n compounding periods
    """
    return ((1 + rate_m / m) ** m) ** (1 / n) - 1


def current_yield(annual_coupon: float, flat_price: float) -> float:
    """
    Calculate current yield
    
    Args:
        annual_coupon: Annual coupon payment
        flat_price: Flat price of the bond
    
    Returns:
        Current yield
    """
    return annual_coupon / flat_price


def simple_yield(coupon: float, face_value: float, flat_price: float, years_to_maturity: float) -> float:
    """
    Calculate simple yield
    
    Args:
        coupon: Annual coupon payment
        face_value: Face value
        flat_price: Flat price of the bond
        years_to_maturity: Years to maturity
    
    Returns:
        Simple yield
    """
    return (coupon + ((face_value - flat_price) / years_to_maturity)) / flat_price


def callable_bond_price(option_free_price: float, embedded_call_option: float) -> float:
    """
    Calculate price of callable bond
    
    Args:
        option_free_price: Price of option-free bond
        embedded_call_option: Value of embedded call option
    
    Returns:
        Price of callable bond
    """
    return option_free_price - embedded_call_option


def forward_rate(spot_rate_longer: float, spot_rate_shorter: float, 
               longer_maturity: float, shorter_maturity: float) -> float:
    """
    Calculate implied forward rate
    
    Args:
        spot_rate_longer: Spot rate for the longer maturity
        spot_rate_shorter: Spot rate for the shorter maturity
        longer_maturity: Longer maturity period
        shorter_maturity: Shorter maturity period
    
    Returns:
        Implied forward rate
    """
    # Calculate using the relationship: (1 + zB)^B = (1 + zA)^A × (1 + IFR)^(B-A)
    return ((1 + spot_rate_longer) ** longer_maturity / 
            (1 + spot_rate_shorter) ** shorter_maturity) ** (1 / (longer_maturity - shorter_maturity)) - 1


def g_spread(corporate_ytm: float, government_ytm: float) -> float:
    """
    Calculate G-spread
    
    Args:
        corporate_ytm: Yield to maturity of corporate bond
        government_ytm: Yield to maturity of government bond
    
    Returns:
        G-spread
    """
    return corporate_ytm - government_ytm


def i_spread(corporate_ytm: float, swap_rate: float) -> float:
    """
    Calculate I-spread
    
    Args:
        corporate_ytm: Yield to maturity of corporate bond
        swap_rate: Swap rate
    
    Returns:
        I-spread
    """
    return corporate_ytm - swap_rate


def option_adjusted_spread(z_spread: float, option_value: float) -> float:
    """
    Calculate option-adjusted spread (OAS)
    
    Args:
        z_spread: Z-spread
        option_value: Option value (in basis points)
    
    Returns:
        Option-adjusted spread
    """
    return z_spread - option_value


### INTRODUCTION TO ASSET-BACKED SECURITIES ###

def single_monthly_mortality_rate(prepayment: float, beginning_balance: float, 
                                scheduled_principal: float) -> float:
    """
    Calculate single monthly mortality rate (SMM)
    
    Args:
        prepayment: Prepayment for the month
        beginning_balance: Beginning outstanding mortgage balance for the month
        scheduled_principal: Scheduled principal repayment for the month
    
    Returns:
        Single monthly mortality rate
    """
    return prepayment / (beginning_balance - scheduled_principal)


### UNDERSTANDING FIXED-INCOME RISK AND RETURN ###

def horizon_yield(purchase_price: float, reinvested_coupon_value: float, 
                sale_price: float, holding_period: float) -> float:
    """
    Calculate horizon yield
    
    Args:
        purchase_price: Bond purchase price
        reinvested_coupon_value: Sum of reinvested coupon payments
        sale_price: Sale price or redemption amount
        holding_period: Holding period (in years)
    
    Returns:
        Horizon yield
    """
    future_value = reinvested_coupon_value + sale_price
    return (future_value / purchase_price) ** (1 / holding_period) - 1

###############################
# QUANTITATIVE METHODS
###############################

### TIME VALUE OF MONEY ###

def future_value_single_cf(present_value: float, rate: float, periods: int, 
                          compounding_frequency: int = 1) -> float:
    """
    Calculate future value of a single cash flow
    
    Args:
        present_value: Present value (initial investment)
        rate: Interest rate per year (as a decimal)
        periods: Number of years
        compounding_frequency: Number of times compounding occurs per year
    
    Returns:
        Future value
    """
    adj_rate = rate / compounding_frequency
    adj_periods = periods * compounding_frequency
    return present_value * (1 + adj_rate) ** adj_periods


def present_value_single_cf(future_value: float, rate: float, periods: int, 
                           compounding_frequency: int = 1) -> float:
    """
    Calculate present value of a single future cash flow
    
    Args:
        future_value: Future value
        rate: Interest rate per year (as a decimal)
        periods: Number of years
        compounding_frequency: Number of times compounding occurs per year
    
    Returns:
        Present value
    """
    adj_rate = rate / compounding_frequency
    adj_periods = periods * compounding_frequency
    return future_value / ((1 + adj_rate) ** adj_periods)


def effective_annual_rate(rate: float, compounding_frequency: int) -> float:
    """
    Calculate the effective annual rate given a nominal rate
    
    Args:
        rate: Nominal interest rate per year (as a decimal)
        compounding_frequency: Number of times compounding occurs per year
    
    Returns:
        Effective annual rate
    """
    return (1 + rate / compounding_frequency) ** compounding_frequency - 1


def future_value_continuous(present_value: float, rate: float, periods: int) -> float:
    """
    Calculate future value with continuous compounding
    
    Args:
        present_value: Present value (initial investment)
        rate: Continuously compounded rate (as a decimal)
        periods: Number of years
    
    Returns:
        Future value
    """
    return present_value * math.exp(rate * periods)


def present_value_continuous(future_value: float, rate: float, periods: int) -> float:
    """
    Calculate present value with continuous compounding
    
    Args:
        future_value: Future value
        rate: Continuously compounded rate (as a decimal)
        periods: Number of years
        
    Returns:
        Present value
    """
    return future_value * math.exp(-rate * periods)


def future_value_ordinary_annuity(payment: float, rate: float, periods: int) -> float:
    """
    Calculate future value of an ordinary annuity
    
    Args:
        payment: Annuity payment per period
        rate: Interest rate per period (as a decimal)
        periods: Number of periods
    
    Returns:
        Future value
    """
    return payment * ((1 + rate) ** periods - 1) / rate


def present_value_ordinary_annuity(payment: float, rate: float, periods: int) -> float:
    """
    Calculate present value of an ordinary annuity
    
    Args:
        payment: Annuity payment per period
        rate: Interest rate per period (as a decimal)
        periods: Number of periods
    
    Returns:
        Present value
    """
    return payment * (1 - 1 / ((1 + rate) ** periods)) / rate


def future_value_annuity_due(payment: float, rate: float, periods: int) -> float:
    """
    Calculate future value of an annuity due
    
    Args:
        payment: Annuity payment per period
        rate: Interest rate per period (as a decimal)
        periods: Number of periods
    
    Returns:
        Future value
    """
    return payment * ((1 + rate) ** periods - 1) / rate * (1 + rate)


def present_value_annuity_due(payment: float, rate: float, periods: int) -> float:
    """
    Calculate present value of an annuity due
    
    Args:
        payment: Annuity payment per period
        rate: Interest rate per period (as a decimal)
        periods: Number of periods
    
    Returns:
        Present value
    """
    return payment * (1 - 1 / ((1 + rate) ** periods)) / rate * (1 + rate)


def present_value_perpetuity(payment: float, rate: float, immediate: bool = False) -> float:
    """
    Calculate present value of a perpetuity
    
    Args:
        payment: Perpetuity payment per period
        rate: Interest rate per period (as a decimal)
        immediate: If True, first payment occurs immediately; otherwise, first payment in one period
    
    Returns:
        Present value
    """
    if immediate:
        return payment / rate * (1 + rate)
    else:
        return payment / rate


### STATISTICAL CONCEPTS AND MARKET RETURNS ###

def arithmetic_mean(values: List[float]) -> float:
    """
    Calculate arithmetic mean
    
    Args:
        values: List of values
    
    Returns:
        Arithmetic mean
    """
    return sum(values) / len(values)


def weighted_mean(values: List[float], weights: List[float]) -> float:
    """
    Calculate weighted mean
    
    Args:
        values: List of values
        weights: List of weights
    
    Returns:
        Weighted mean
    """
    return sum(w * x for w, x in zip(weights, values))


def geometric_mean(values: List[float]) -> float:
    """
    Calculate geometric mean
    
    Args:
        values: List of non-negative values
    
    Returns:
        Geometric mean
    """
    if any(x < 0 for x in values):
        raise ValueError("All values must be non-negative")
    
    return np.prod(values) ** (1 / len(values))


def geometric_mean_returns(returns: List[float]) -> float:
    """
    Calculate geometric mean of returns
    
    Args:
        returns: List of returns (as decimals)
    
    Returns:
        Geometric mean return
    """
    return np.prod([1 + r for r in returns]) ** (1 / len(returns)) - 1


def harmonic_mean(values: List[float]) -> float:
    """
    Calculate harmonic mean
    
    Args:
        values: List of values
    
    Returns:
        Harmonic mean
    """
    return len(values) / sum(1 / x for x in values)


def median_position(n: int) -> float:
    """
    Calculate position of median in a sorted list
    
    Args:
        n: Number of values
    
    Returns:
        Position of median
    """
    return (n + 1) / 2


def percentile_position(n: int, percentile: float) -> float:
    """
    Calculate position of percentile in a sorted list
    
    Args:
        n: Number of values
        percentile: Percentile as a decimal (0.01 for 1%)
    
    Returns:
        Position of percentile
    """
    return (n + 1) * percentile


def range_statistic(values: List[float]) -> float:
    """
    Calculate range statistic
    
    Args:
        values: List of values
    
    Returns:
        Range (maximum - minimum)
    """
    return max(values) - min(values)


def mean_absolute_deviation(values: List[float]) -> float:
    """
    Calculate mean absolute deviation
    
    Args:
        values: List of values
    
    Returns:
        Mean absolute deviation
    """
    mean = arithmetic_mean(values)
    return sum(abs(x - mean) for x in values) / len(values)


def population_variance(values: List[float]) -> float:
    """
    Calculate population variance
    
    Args:
        values: List of values
    
    Returns:
        Population variance
    """
    mean = arithmetic_mean(values)
    return sum((x - mean) ** 2 for x in values) / len(values)


def sample_variance(values: List[float]) -> float:
    """
    Calculate sample variance
    
    Args:
        values: List of values
    
    Returns:
        Sample variance
    """
    mean = arithmetic_mean(values)
    return sum((x - mean) ** 2 for x in values) / (len(values) - 1)


def population_std_dev(values: List[float]) -> float:
    """
    Calculate population standard deviation
    
    Args:
        values: List of values
    
    Returns:
        Population standard deviation
    """
    return math.sqrt(population_variance(values))


def sample_std_dev(values: List[float]) -> float:
    """
    Calculate sample standard deviation
    
    Args:
        values: List of values
    
    Returns:
        Sample standard deviation
    """
    return math.sqrt(sample_variance(values))


def semivariance(values: List[float]) -> float:
    """
    Calculate semivariance
    
    Args:
        values: List of values
    
    Returns:
        Semivariance
    """
    mean = arithmetic_mean(values)
    below_mean = [x for x in values if x <= mean]
    return sum((x - mean) ** 2 for x in below_mean) / (len(values) - 1)


def semideviation(values: List[float]) -> float:
    """
    Calculate semideviation
    
    Args:
        values: List of values
    
    Returns:
        Semideviation
    """
    return math.sqrt(semivariance(values))


def target_semivariance(values: List[float], target: float) -> float:
    """
    Calculate target semivariance
    
    Args:
        values: List of values
        target: Target level
    
    Returns:
        Target semivariance
    """
    below_target = [x for x in values if x <= target]
    return sum((x - target) ** 2 for x in below_target) / (len(values) - 1)


def target_semideviation(values: List[float], target: float) -> float:
    """
    Calculate target semideviation
    
    Args:
        values: List of values
        target: Target level
    
    Returns:
        Target semideviation
    """
    return math.sqrt(target_semivariance(values, target))


def chebyshev_inequality(k: float) -> float:
    """
    Calculate probability from Chebyshev's inequality
    
    Args:
        k: Number of standard deviations from mean (k > 1)
    
    Returns:
        Lower bound on probability
    """
    if k <= 1:
        raise ValueError("k must be greater than 1")
    
    return 1 - 1 / (k ** 2)


def coefficient_of_variation(values: List[float]) -> float:
    """
    Calculate coefficient of variation
    
    Args:
        values: List of values
    
    Returns:
        Coefficient of variation
    """
    return sample_std_dev(values) / arithmetic_mean(values)


def sample_skewness(values: List[float]) -> float:
    """
    Calculate sample skewness
    
    Args:
        values: List of values
    
    Returns:
        Sample skewness
    """
    n = len(values)
    mean = arithmetic_mean(values)
    std_dev = sample_std_dev(values)
    return (n / ((n - 1) * (n - 2))) * sum(((x - mean) / std_dev) ** 3 for x in values)


def sample_excess_kurtosis(values: List[float]) -> float:
    """
    Calculate sample excess kurtosis
    
    Args:
        values: List of values
    
    Returns:
        Sample excess kurtosis
    """
    n = len(values)
    mean = arithmetic_mean(values)
    std_dev = sample_std_dev(values)
    
    numerator = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
    sum_term = sum(((x - mean) / std_dev) ** 4 for x in values)
    correction = 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    
    return numerator * sum_term - correction


def geometric_mean_approximation(arithmetic_mean: float, variance: float) -> float:
    """
    Approximate geometric mean from arithmetic mean and variance
    
    Args:
        arithmetic_mean: Arithmetic mean
        variance: Variance
    
    Returns:
        Approximated geometric mean
    """
    return arithmetic_mean - 0.5 * variance


### PROBABILITY CONCEPTS ###

def odds_for_event(probability: float) -> float:
    """
    Calculate odds for an event
    
    Args:
        probability: Probability of event (as a decimal)
    
    Returns:
        Odds for event
    """
    return probability / (1 - probability)


def odds_against_event(probability: float) -> float:
    """
    Calculate odds against an event
    
    Args:
        probability: Probability of event (as a decimal)
    
    Returns:
        Odds against event
    """
    return (1 - probability) / probability


def probability_from_odds(odds_ratio: Tuple[float, float]) -> float:
    """
    Calculate probability from odds ratio
    
    Args:
        odds_ratio: Tuple of (a, b) representing odds of a:b
    
    Returns:
        Probability
    """
    a, b = odds_ratio
    return a / (a + b)


def conditional_probability(joint_prob: float, condition_prob: float) -> float:
    """
    Calculate conditional probability
    
    Args:
        joint_prob: Probability of A and B
        condition_prob: Probability of the condition (B)
    
    Returns:
        Conditional probability P(A|B)
    """
    return joint_prob / condition_prob


def bayes_rule(prior: float, likelihood: float, evidence: float) -> float:
    """
    Apply Bayes' rule to calculate posterior probability
    
    Args:
        prior: Prior probability P(A)
        likelihood: Likelihood P(B|A)
        evidence: Evidence P(B)
    
    Returns:
        Posterior probability P(A|B)
    """
    return (likelihood * prior) / evidence


def expected_value(values: List[float], probabilities: List[float]) -> float:
    """
    Calculate expected value of a discrete random variable
    
    Args:
        values: List of possible values
        probabilities: List of corresponding probabilities
    
    Returns:
        Expected value
    """
    return sum(v * p for v, p in zip(values, probabilities))


def variance_discrete_rv(values: List[float], probabilities: List[float]) -> float:
    """
    Calculate variance of a discrete random variable
    
    Args:
        values: List of possible values
        probabilities: List of corresponding probabilities
    
    Returns:
        Variance
    """
    ev = expected_value(values, probabilities)
    return sum(p * ((x - ev) ** 2) for x, p in zip(values, probabilities))


def std_dev_discrete_rv(values: List[float], probabilities: List[float]) -> float:
    """
    Calculate standard deviation of a discrete random variable
    
    Args:
        values: List of possible values
        probabilities: List of corresponding probabilities
    
    Returns:
        Standard deviation
    """
    return math.sqrt(variance_discrete_rv(values, probabilities))


def covariance_discrete_rv(x_values: List[float], y_values: List[float], 
                          joint_probabilities: List[float]) -> float:
    """
    Calculate covariance of two discrete random variables
    
    Args:
        x_values: List of all possible x values
        y_values: List of all possible y values
        joint_probabilities: Joint probability of each (x,y) pair
    
    Returns:
        Covariance
    """
    # Calculate expected values for X and Y
    x_probs = [sum(joint_probabilities[i] for i in range(len(joint_probabilities)) 
                 if x_values[i // len(y_values)] == x) for x in set(x_values)]
    y_probs = [sum(joint_probabilities[i] for i in range(len(joint_probabilities)) 
                 if y_values[i % len(y_values)] == y) for y in set(y_values)]
    
    ex = expected_value(list(set(x_values)), x_probs)
    ey = expected_value(list(set(y_values)), y_probs)
    
    return sum(p * (x - ex) * (y - ey) for x, y, p in zip(x_values, y_values, joint_probabilities))


def correlation_coefficient(x_values: List[float], y_values: List[float], 
                           joint_probabilities: List[float]) -> float:
    """
    Calculate correlation coefficient of two discrete random variables
    
    Args:
        x_values: List of all possible x values
        y_values: List of all possible y values
        joint_probabilities: Joint probability of each (x,y) pair
    
    Returns:
        Correlation coefficient
    """
    cov = covariance_discrete_rv(x_values, y_values, joint_probabilities)
    
    # Calculate standard deviations
    x_probs = [sum(joint_probabilities[i] for i in range(len(joint_probabilities)) 
                 if x_values[i // len(y_values)] == x) for x in set(x_values)]
    y_probs = [sum(joint_probabilities[i] for i in range(len(joint_probabilities)) 
                 if y_values[i % len(y_values)] == y) for y in set(y_values)]
    
    std_x = std_dev_discrete_rv(list(set(x_values)), x_probs)
    std_y = std_dev_discrete_rv(list(set(y_values)), y_probs)
    
    return cov / (std_x * std_y)


def portfolio_return_two_asset(w1: float, r1: float, w2: float, r2: float) -> float:
    """
    Calculate expected return of a two-asset portfolio
    
    Args:
        w1: Weight of asset 1
        r1: Expected return of asset 1
        w2: Weight of asset 2
        r2: Expected return of asset 2
    
    Returns:
        Expected portfolio return
    """
    return w1 * r1 + w2 * r2


def portfolio_variance_two_asset(w1: float, sigma1: float, w2: float, sigma2: float, 
                               correlation: float) -> float:
    """
    Calculate variance of a two-asset portfolio
    
    Args:
        w1: Weight of asset 1
        sigma1: Standard deviation of asset 1
        w2: Weight of asset 2
        sigma2: Standard deviation of asset 2
        correlation: Correlation coefficient between assets
    
    Returns:
        Portfolio variance
    """
    cov = correlation * sigma1 * sigma2
    return (w1 ** 2) * (sigma1 ** 2) + (w2 ** 2) * (sigma2 ** 2) + 2 * w1 * w2 * cov


def portfolio_return_three_asset(w1: float, r1: float, w2: float, r2: float, 
                               w3: float, r3: float) -> float:
    """
    Calculate expected return of a three-asset portfolio
    
    Args:
        w1: Weight of asset 1
        r1: Expected return of asset 1
        w2: Weight of asset 2
        r2: Expected return of asset 2
        w3: Weight of asset 3
        r3: Expected return of asset 3
    
    Returns:
        Expected portfolio return
    """
    return w1 * r1 + w2 * r2 + w3 * r3


def portfolio_variance_three_asset(w1: float, sigma1: float, w2: float, sigma2: float, 
                                 w3: float, sigma3: float, corr12: float, 
                                 corr13: float, corr23: float) -> float:
    """
    Calculate variance of a three-asset portfolio
    
    Args:
        w1: Weight of asset 1
        sigma1: Standard deviation of asset 1
        w2: Weight of asset 2
        sigma2: Standard deviation of asset 2
        w3: Weight of asset 3
        sigma3: Standard deviation of asset 3
        corr12: Correlation coefficient between assets 1 and 2
        corr13: Correlation coefficient between assets 1 and 3
        corr23: Correlation coefficient between assets 2 and 3
    
    Returns:
        Portfolio variance
    """
    cov12 = corr12 * sigma1 * sigma2
    cov13 = corr13 * sigma1 * sigma3
    cov23 = corr23 * sigma2 * sigma3
    
    return (w1 ** 2) * (sigma1 ** 2) + (w2 ** 2) * (sigma2 ** 2) + (w3 ** 2) * (sigma3 ** 2) + \
           2 * w1 * w2 * cov12 + 2 * w1 * w3 * cov13 + 2 * w2 * w3 * cov23


def combination(n: int, r: int) -> int:
    """
    Calculate number of combinations (nCr)
    
    Args:
        n: Total number of items
        r: Number of items to choose
    
    Returns:
        Number of combinations
    """
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))


def permutation(n: int, r: int) -> int:
    """
    Calculate number of permutations (nPr)
    
    Args:
        n: Total number of items
        r: Number of items to arrange
    
    Returns:
        Number of permutations
    """
    return math.factorial(n) // math.factorial(n - r)


### COMMON PROBABILITY DISTRIBUTIONS ###

def binomial_probability(n: int, r: int, p: float) -> float:
    """
    Calculate binomial probability
    
    Args:
        n: Number of trials
        r: Number of successes
        p: Probability of success on a single trial
    
    Returns:
        Probability of exactly r successes in n trials
    """
    return combination(n, r) * (p ** r) * ((1 - p) ** (n - r))


def binomial_mean(n: int, p: float) -> float:
    """
    Calculate mean of binomial distribution
    
    Args:
        n: Number of trials
        p: Probability of success on a single trial
    
    Returns:
        Mean of binomial distribution
    """
    return n * p


def binomial_variance(n: int, p: float) -> float:
    """
    Calculate variance of binomial distribution
    
    Args:
        n: Number of trials
        p: Probability of success on a single trial
    
    Returns:
        Variance of binomial distribution
    """
    return n * p * (1 - p)


def binomial_std_dev(n: int, p: float) -> float:
    """
    Calculate standard deviation of binomial distribution
    
    Args:
        n: Number of trials
        p: Probability of success on a single trial
    
    Returns:
        Standard deviation of binomial distribution
    """
    return math.sqrt(binomial_variance(n, p))


def uniform_pdf(x: float, a: float, b: float) -> float:
    """
    Calculate PDF of continuous uniform distribution
    
    Args:
        x: Value
        a: Lower bound of uniform distribution
        b: Upper bound of uniform distribution
    
    Returns:
        Probability density at x
    """
    if a <= x <= b:
        return 1 / (b - a)
    else:
        return 0


def uniform_cdf(x: float, a: float, b: float) -> float:
    """
    Calculate CDF of continuous uniform distribution
    
    Args:
        x: Value
        a: Lower bound of uniform distribution
        b: Upper bound of uniform distribution
    
    Returns:
        Cumulative probability at x
    """
    if x < a:
        return 0
    elif a <= x <= b:
        return (x - a) / (b - a)
    else:
        return 1


def uniform_mean(a: float, b: float) -> float:
    """
    Calculate mean of continuous uniform distribution
    
    Args:
        a: Lower bound of uniform distribution
        b: Upper bound of uniform distribution
    
    Returns:
        Mean of uniform distribution
    """
    return (a + b) / 2


def uniform_variance(a: float, b: float) -> float:
    """
    Calculate variance of continuous uniform distribution
    
    Args:
        a: Lower bound of uniform distribution
        b: Upper bound of uniform distribution
    
    Returns:
        Variance of uniform distribution
    """
    return ((b - a) ** 2) / 12


def normal_cdf(x: float, mean: float = 0, std_dev: float = 1) -> float:
    """
    Calculate CDF of normal distribution
    
    Args:
        x: Value
        mean: Mean of normal distribution
        std_dev: Standard deviation of normal distribution
    
    Returns:
        Cumulative probability at x
    """
    z = (x - mean) / std_dev
    return (1 + math.erf(z / math.sqrt(2))) / 2


def standard_normal_cdf(z: float) -> float:
    """
    Calculate CDF of standard normal distribution
    
    Args:
        z: Standard normal value
    
    Returns:
        Cumulative probability at z
    """
    return normal_cdf(z)


def roys_safety_first_ratio(expected_return: float, minimum_return: float, std_dev: float) -> float:
    """
    Calculate Roy's safety-first ratio
    
    Args:
        expected_return: Expected return
        minimum_return: Minimum acceptable return
        std_dev: Standard deviation
    
    Returns:
        Roy's safety-first ratio
    """
    return (expected_return - minimum_return) / std_dev


def shortfall_risk(safety_first_ratio: float) -> float:
    """
    Calculate shortfall risk using Roy's safety-first ratio
    
    Args:
        safety_first_ratio: Roy's safety-first ratio
    
    Returns:
        Shortfall risk probability
    """
    return normal_cdf(-safety_first_ratio)


def lognormal_mean(mu: float, sigma_squared: float) -> float:
    """
    Calculate mean of lognormal random variable
    
    Args:
        mu: Mean of underlying normal distribution
        sigma_squared: Variance of underlying normal distribution
    
    Returns:
        Mean of lognormal distribution
    """
    return math.exp(mu + 0.5 * sigma_squared)


def lognormal_variance(mu: float, sigma_squared: float) -> float:
    """
    Calculate variance of lognormal random variable
    
    Args:
        mu: Mean of underlying normal distribution
        sigma_squared: Variance of underlying normal distribution
    
    Returns:
        Variance of lognormal distribution
    """
    return math.exp(2 * mu + sigma_squared) * (math.exp(sigma_squared) - 1)


### SAMPLING AND ESTIMATION ###

def confidence_interval_known_sigma(sample_mean: float, sigma: float, n: int, 
                                  confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for population mean (σ known)
    
    Args:
        sample_mean: Sample mean
        sigma: Known population standard deviation
        n: Sample size
        confidence_level: Confidence level (default 0.95 for 95%)
    
    Returns:
        Tuple of (lower bound, upper bound)
    """
    alpha = 1 - confidence_level
    z_critical = -normal_cdf(alpha / 2)  # Using approximation
    
    margin_error = z_critical * (sigma / math.sqrt(n))
    
    return (sample_mean - margin_error, sample_mean + margin_error)


def confidence_interval_unknown_sigma(sample_mean: float, sample_std: float, n: int, 
                                    confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for population mean (σ unknown)
    
    Args:
        sample_mean: Sample mean
        sample_std: Sample standard deviation
        n: Sample size
        confidence_level: Confidence level (default 0.95 for 95%)
    
    Returns:
        Tuple of (lower bound, upper bound)
    """
    import scipy.stats as stats
    
    alpha = 1 - confidence_level
    df = n - 1
    t_critical = stats.t.ppf(1 - alpha / 2, df)
    
    margin_error = t_critical * (sample_std / math.sqrt(n))
    
    return (sample_mean - margin_error, sample_mean + margin_error)


### HYPOTHESIS TESTING ###

def z_test_statistic(sample_mean: float, pop_mean: float, pop_std: float, n: int) -> float:
    """
    Calculate Z-test statistic
    
    Args:
        sample_mean: Sample mean
        pop_mean: Hypothesized population mean
        pop_std: Known population standard deviation
        n: Sample size
    
    Returns:
        Z-test statistic
    """
    return (sample_mean - pop_mean) / (pop_std / math.sqrt(n))


def t_test_statistic(sample_mean: float, pop_mean: float, sample_std: float, n: int) -> float:
    """
    Calculate t-test statistic
    
    Args:
        sample_mean: Sample mean
        pop_mean: Hypothesized population mean
        sample_std: Sample standard deviation
        n: Sample size
    
    Returns:
        t-test statistic
    """
    if equal_var:
        # Pooled standard deviation
        sp_squared = ((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2)
        return (mean1 - mean2) / math.sqrt(sp_squared * (1/n1 + 1/n2))
    else:
        # Welch's t-test
        return (mean1 - mean2) / math.sqrt((std1 ** 2 / n1) + (std2 ** 2 / n2))


def chi_square_variance(sample_var: float, pop_var: float, n: int) -> float:
    """
    Calculate chi-square test statistic for population variance
    
    Args:
        sample_var: Sample variance
        pop_var: Hypothesized population variance
        n: Sample size
    
    Returns:
        Chi-square test statistic
    """
    return (n - 1) * sample_var / pop_var


def f_test_variances(var1: float, var2: float, n1: int, n2: int) -> float:
    """
    Calculate F-test statistic for difference between two population variances
    
    Args:
        var1: Sample variance of population 1 (larger variance)
        var2: Sample variance of population 2 (smaller variance)
        n1: Sample size of population 1
        n2: Sample size of population 2
    
    Returns:
        F-test statistic
    """
    # Ensure var1 is the larger variance
    if var1 < var2:
        var1, var2 = var2, var1
        n1, n2 = n2, n1
    
    return var1 / var2


def correlation_test_statistic(r: float, n: int) -> float:
    """
    Calculate test statistic for testing if correlation is zero
    
    Args:
        r: Sample correlation coefficient
        n: Sample size
    
    Returns:
        t-test statistic
    """
    return r * math.sqrt(n - 2) / math.sqrt(1 - r ** 2)


def spearman_rank_correlation(ranks1: List[float], ranks2: List[float]) -> float:
    """
    Calculate Spearman's rank correlation coefficient
    
    Args:
        ranks1: Ranks of first variable
        ranks2: Ranks of second variable
    
    Returns:
        Spearman's rank correlation coefficient
    """
    n = len(ranks1)
    d_squared_sum = sum((r1 - r2) ** 2 for r1, r2 in zip(ranks1, ranks2))
    
    return 1 - (6 * d_squared_sum) / (n * (n ** 2 - 1))


def spearman_test_statistic(rs: float, n: int) -> float:
    """
    Calculate test statistic for Spearman's rank correlation
    
    Args:
        rs: Spearman's rank correlation coefficient
        n: Sample size
    
    Returns:
        t-test statistic
    """
    return rs * math.sqrt(n - 2) / math.sqrt(1 - rs ** 2)


###############################
# ECONOMICS
###############################

### TOPICS IN DEMAND AND SUPPLY ###

def linear_demand_price(a: float, b: float, q: float) -> float:
    """
    Calculate price from linear demand function Q = a + bP
    
    Args:
        a: Intercept parameter of demand function
        b: Slope parameter of demand function (negative for normal goods)
        q: Quantity demanded
    
    Returns:
        Price
    """
    # P = (Q - a) / b
    return (q - a) / b


def price_elasticity_of_demand(pct_change_q: float, pct_change_p: float) -> float:
    """
    Calculate own-price elasticity of demand
    
    Args:
        pct_change_q: Percentage change in quantity demanded
        pct_change_p: Percentage change in price
    
    Returns:
        Price elasticity of demand
    """
    return pct_change_q / pct_change_p


def income_elasticity_of_demand(pct_change_q: float, pct_change_i: float) -> float:
    """
    Calculate income elasticity of demand
    
    Args:
        pct_change_q: Percentage change in quantity demanded
        pct_change_i: Percentage change in income
    
    Returns:
        Income elasticity of demand
    """
    return pct_change_q / pct_change_i


def cross_price_elasticity(pct_change_qx: float, pct_change_py: float) -> float:
    """
    Calculate cross-price elasticity of demand
    
    Args:
        pct_change_qx: Percentage change in quantity demanded of good X
        pct_change_py: Percentage change in price of good Y
    
    Returns:
        Cross-price elasticity of demand
    """
    return pct_change_qx / pct_change_py


def economic_profit(total_revenue: float, total_economic_cost: float) -> float:
    """
    Calculate economic profit
    
    Args:
        total_revenue: Total revenue
        total_economic_cost: Total economic cost
    
    Returns:
        Economic profit
    """
    return total_revenue - total_economic_cost


def normal_profit(accounting_profit: float, economic_profit: float) -> float:
    """
    Calculate normal profit
    
    Args:
        accounting_profit: Accounting profit
        economic_profit: Economic profit
    
    Returns:
        Normal profit
    """
    return accounting_profit - economic_profit


def total_revenue(price: float, quantity: float) -> float:
    """
    Calculate total revenue
    
    Args:
        price: Price per unit
        quantity: Quantity sold
    
    Returns:
        Total revenue
    """
    return price * quantity


def average_revenue(total_revenue: float, quantity: float) -> float:
    """
    Calculate average revenue
    
    Args:
        total_revenue: Total revenue
        quantity: Quantity sold
    
    Returns:
        Average revenue
    """
    return total_revenue / quantity


def marginal_revenue(change_in_revenue: float, change_in_quantity: float) -> float:
    """
    Calculate marginal revenue
    
    Args:
        change_in_revenue: Change in total revenue
        change_in_quantity: Change in quantity sold
    
    Returns:
        Marginal revenue
    """
    return change_in_revenue / change_in_quantity


def marginal_revenue_formula(price: float, quantity: float, change_in_price: float, 
                           change_in_quantity: float) -> float:
    """
    Calculate marginal revenue using slope of demand curve
    
    Args:
        price: Price per unit
        quantity: Quantity sold
        change_in_price: Change in price per unit
        change_in_quantity: Change in quantity sold
    
    Returns:
        Marginal revenue
    """
    return price + quantity * (change_in_price / change_in_quantity)


def total_cost(fixed_costs: float, variable_costs: float) -> float:
    """
    Calculate total cost
    
    Args:
        fixed_costs: Total fixed costs
        variable_costs: Total variable costs
    
    Returns:
        Total cost
    """
    return fixed_costs + variable_costs


def average_total_cost(total_cost: float, quantity: float) -> float:
    """
    Calculate average total cost
    
    Args:
        total_cost: Total cost
        quantity: Quantity produced
    
    Returns:
        Average total cost
    """
    return total_cost / quantity


def marginal_cost(change_in_cost: float, change_in_quantity: float) -> float:
    """
    Calculate marginal cost
    
    Args:
        change_in_cost: Change in total cost
        change_in_quantity: Change in quantity produced
    
    Returns:
        Marginal cost
    """
    return change_in_cost / change_in_quantity


def average_fixed_cost(fixed_cost: float, quantity: float) -> float:
    """
    Calculate average fixed cost
    
    Args:
        fixed_cost: Total fixed cost
        quantity: Quantity produced
    
    Returns:
        Average fixed cost
    """
    return fixed_cost / quantity


def average_variable_cost(variable_cost: float, quantity: float) -> float:
    """
    Calculate average variable cost
    
    Args:
        variable_cost: Total variable cost
        quantity: Quantity produced
    
    Returns:
        Average variable cost
    """
    return variable_cost / quantity


### THE FIRM AND MARKET STRUCTURES ###

def marginal_revenue_with_elasticity(price: float, elasticity: float) -> float:
    """
    Calculate marginal revenue using price elasticity of demand
    
    Args:
        price: Price per unit
        elasticity: Price elasticity of demand (absolute value)
    
    Returns:
        Marginal revenue
    """
    return price * (1 - 1 / elasticity)


def mr_from_total_revenue_function(b: float, c: float, q: float) -> float:
    """
    Calculate marginal revenue from quadratic total revenue function
    
    Args:
        b: Linear coefficient in TR = a + bQ + cQ²
        c: Quadratic coefficient in TR = a + bQ + cQ²
        q: Quantity
    
    Returns:
        Marginal revenue
    """
    return b + 2 * c * q


def concentration_ratio(market_shares: List[float]) -> float:
    """
    Calculate concentration ratio
    
    Args:
        market_shares: List of market shares (as decimals) for the largest firms
    
    Returns:
        Concentration ratio
    """
    return sum(market_shares)


def herfindahl_hirschman_index(market_shares: List[float]) -> float:
    """
    Calculate Herfindahl-Hirschman Index (HHI)
    
    Args:
        market_shares: List of market shares (as percentages)
    
    Returns:
        HHI
    """
    return sum(ms ** 2 for ms in market_shares)


### AGGREGATE OUTPUT, PRICES, AND ECONOMIC GROWTH ###

def nominal_gdp(real_gdp: float, gdp_deflator: float) -> float:
    """
    Calculate nominal GDP
    
    Args:
        real_gdp: Real GDP
        gdp_deflator: GDP deflator (as a decimal, where 1.0 = 100%)
    
    Returns:
        Nominal GDP
    """
    return real_gdp * gdp_deflator


def real_gdp(nominal_gdp: float, gdp_deflator: float) -> float:
    """
    Calculate real GDP
    
    Args:
        nominal_gdp: Nominal GDP
        gdp_deflator: GDP deflator (as a decimal, where 1.0 = 100%)
    
    Returns:
        Real GDP
    """
    return nominal_gdp / gdp_deflator


def gdp_deflator(nominal_gdp: float, real_gdp: float) -> float:
    """
    Calculate GDP deflator
    
    Args:
        nominal_gdp: Nominal GDP
        real_gdp: Real GDP
    
    Returns:
        GDP deflator (as a decimal, where 1.0 = 100%)
    """
    return nominal_gdp / real_gdp * 100


def gdp_expenditure_approach(c: float, i: float, g: float, x: float, m: float) -> float:
    """
    Calculate GDP using the expenditure approach
    
    Args:
        c: Consumer spending
        i: Business investment
        g: Government spending
        x: Exports
        m: Imports
    
    Returns:
        GDP
    """
    return c + i + g + (x - m)


def household_disposable_income(personal_income: float, personal_taxes: float) -> float:
    """
    Calculate household disposable income
    
    Args:
        personal_income: Personal income
        personal_taxes: Personal taxes
    
    Returns:
        Household disposable income
    """
    return personal_income - personal_taxes


def household_saving(disposable_income: float, consumption: float, 
                   pension_entitlements: float = 0) -> float:
    """
    Calculate household saving
    
    Args:
        disposable_income: Household disposable income
        consumption: Consumption expenditures
        pension_entitlements: Net change in pension entitlements
    
    Returns:
        Household saving
    """
    return disposable_income - consumption + pension_entitlements


def savings_rate(household_saving: float, disposable_income: float) -> float:
    """
    Calculate savings rate
    
    Args:
        household_saving: Household saving
        disposable_income: Personal disposable income
    
    Returns:
        Savings rate (as a decimal)
    """
    return household_saving / disposable_income


def domestic_saving_identity(investment: float, govt_deficit: float, net_exports: float) -> float:
    """
    Calculate domestic saving using macroeconomic identity
    
    Args:
        investment: Investment spending
        govt_deficit: Government deficit (G - T)
        net_exports: Net exports (X - M)
    
    Returns:
        Domestic saving
    """
    return investment + govt_deficit + net_exports


def marginal_propensity_to_save(marginal_propensity_to_consume: float) -> float:
    """
    Calculate marginal propensity to save
    
    Args:
        marginal_propensity_to_consume: Marginal propensity to consume
    
    Returns:
        Marginal propensity to save
    """
    return 1 - marginal_propensity_to_consume


def marginal_propensity_to_consume(change_in_consumption: float, 
                                 change_in_disposable_income: float) -> float:
    """
    Calculate marginal propensity to consume
    
    Args:
        change_in_consumption: Change in consumption
        change_in_disposable_income: Change in disposable income
    
    Returns:
        Marginal propensity to consume
    """
    return change_in_consumption / change_in_disposable_income


def unit_labor_cost_change(pct_change_nominal_wages: float, 
                         pct_change_productivity: float) -> float:
    """
    Calculate percentage change in unit labor cost
    
    Args:
        pct_change_nominal_wages: Percentage change in nominal wages
        pct_change_productivity: Percentage change in productivity
    
    Returns:
        Percentage change in unit labor cost
    """
    return pct_change_nominal_wages - pct_change_productivity


def potential_gdp_growth(labor_force_growth: float, productivity_growth: float) -> float:
    """
    Calculate potential GDP growth rate
    
    Args:
        labor_force_growth: Long-term growth rate of labor force
        productivity_growth: Long-term labor productivity growth rate
    
    Returns:
        Potential GDP growth rate
    """
    return labor_force_growth + productivity_growth


### UNDERSTANDING BUSINESS CYCLES ###

def laspeyres_price_index(current_prices: List[float], base_quantities: List[float], 
                        base_prices: List[float]) -> float:
    """
    Calculate Laspeyres Price Index
    
    Args:
        current_prices: List of current period prices
        base_quantities: List of base period quantities
        base_prices: List of base period prices
    
    Returns:
        Laspeyres Price Index
    """
    num = sum(p_curr * q_base for p_curr, q_base in zip(current_prices, base_quantities))
    denom = sum(p_base * q_base for p_base, q_base in zip(base_prices, base_quantities))
    
    return (num / denom) * 100


def paasche_price_index(current_prices: List[float], current_quantities: List[float], 
                      base_prices: List[float]) -> float:
    """
    Calculate Paasche Price Index
    
    Args:
        current_prices: List of current period prices
        current_quantities: List of current period quantities
        base_prices: List of base period prices
    
    Returns:
        Paasche Price Index
    """
    num = sum(p_curr * q_curr for p_curr, q_curr in zip(current_prices, current_quantities))
    denom = sum(p_base * q_curr for p_base, q_curr in zip(base_prices, current_quantities))
    
    return (num / denom) * 100


def fisher_price_index(laspeyres_index: float, paasche_index: float) -> float:
    """
    Calculate Fisher Price Index
    
    Args:
        laspeyres_index: Laspeyres Price Index
        paasche_index: Paasche Price Index
    
    Returns:
        Fisher Price Index
    """
    return math.sqrt(laspeyres_index * paasche_index)


### MONETARY AND FISCAL POLICY ###

def money_multiplier(reserve_requirement: float) -> float:
    """
    Calculate money multiplier
    
    Args:
        reserve_requirement: Reserve requirement (as a decimal)
    
    Returns:
        Money multiplier
    """
    return 1 / reserve_requirement


def total_money_created(new_deposit: float, reserve_requirement: float) -> float:
    """
    Calculate total money created from new deposit
    
    Args:
        new_deposit: Amount of new deposit
        reserve_requirement: Reserve requirement (as a decimal)
    
    Returns:
        Total money created
    """
    return new_deposit / reserve_requirement


def nominal_interest_rate(real_rate: float, expected_inflation: float) -> float:
    """
    Calculate nominal interest rate using Fisher equation
    
    Args:
        real_rate: Real interest rate (as a decimal)
        expected_inflation: Expected inflation rate (as a decimal)
    
    Returns:
        Nominal interest rate
    """
    return real_rate + expected_inflation


def neutral_interest_rate(trend_growth: float, inflation_target: float) -> float:
    """
    Calculate neutral interest rate
    
    Args:
        trend_growth: Trend growth of economy (as a decimal)
        inflation_target: Inflation target (as a decimal)
    
    Returns:
        Neutral interest rate
    """
    return trend_growth + inflation_target


def fiscal_multiplier(marginal_propensity_to_consume: float, tax_rate: float) -> float:
    """
    Calculate fiscal multiplier
    
    Args:
        marginal_propensity_to_consume: Marginal propensity to consume (as a decimal)
        tax_rate: Tax rate (as a decimal)
    
    Returns:
        Fiscal multiplier
    """
    return 1 / (1 - marginal_propensity_to_consume * (1 - tax_rate))


### INTERNATIONAL TRADE AND CAPITAL FLOWS ###

def current_account_balance(private_saving: float, government_saving: float, 
                          investment: float) -> float:
    """
    Calculate current account balance
    
    Args:
        private_saving: Private sector saving
        government_saving: Government savings/surplus
        investment: Investment
    
    Returns:
        Current account balance
    """
    return private_saving + government_saving - investment


def government_saving(taxes: float, govt_spending: float, transfers: float) -> float:
    """
    Calculate government saving
    
    Args:
        taxes: Tax revenue
        govt_spending: Government spending
        transfers: Transfer payments
    
    Returns:
        Government saving
    """
    return taxes - govt_spending - transfers


### CURRENCY EXCHANGE RATES ###

def real_exchange_rate(nominal_rate: float, domestic_cpi: float, foreign_cpi: float) -> float:
    """
    Calculate real exchange rate
    
    Args:
        nominal_rate: Nominal exchange rate (domestic/foreign)
        domestic_cpi: Domestic consumer price index
        foreign_cpi: Foreign consumer price index
    
    Returns:
        Real exchange rate
    """
    return nominal_rate * (foreign_cpi / domestic_cpi)


def forward_rate(spot_rate: float, domestic_rate: float, foreign_rate: float, 
               time_period: float) -> float:
    """
    Calculate forward exchange rate
    
    Args:
        spot_rate: Spot exchange rate (foreign/domestic)
        domestic_rate: Domestic risk-free rate (as a decimal)
        foreign_rate: Foreign risk-free rate (as a decimal)
        time_period: Time horizon (in years)
    
    Returns:
        Forward exchange rate
    """
    return spot_rate * ((1 + foreign_rate * time_period) / (1 + domestic_rate * time_period))


def forward_points(spot_rate: float, domestic_rate: float, foreign_rate: float, 
                 time_period: float) -> float:
    """
    Calculate forward points
    
    Args:
        spot_rate: Spot exchange rate (foreign/domestic)
        domestic_rate: Domestic risk-free rate (as a decimal)
        foreign_rate: Foreign risk-free rate (as a decimal)
        time_period: Time horizon (in years)
    
    Returns:
        Forward points
    """
    return spot_rate * ((foreign_rate - domestic_rate) / (1 + domestic_rate * time_period)) * time_period


def expenditure_change(price_elasticity: float, price_change_pct: float) -> float:
    """
    Calculate percentage change in expenditure based on price elasticity
    
    Args:
        price_elasticity: Price elasticity (absolute value)
        price_change_pct: Percentage change in price
    
    Returns:
        Percentage change in expenditure
    """
    return (1 - price_elasticity) * price_change_pct


def marshall_lerner_condition(export_weight: float, export_elasticity: float, 
                            import_weight: float, import_elasticity: float) -> bool:
    """
    Check if Marshall-Lerner condition holds
    
    Args:
        export_weight: Weight of exports in total trade
        export_elasticity: Price elasticity of foreign demand for domestic exports
        import_weight: Weight of imports in total trade
        import_elasticity: Price elasticity of domestic demand for imports
    
    Returns:
        True if condition holds, False otherwise
    """
    condition = export_weight * export_elasticity + import_weight * (import_elasticity - 1)
    return condition > 0


###############################
# FINANCIAL REPORTING & ANALYSIS
###############################

### UNDERSTANDING INCOME STATEMENTS ###

def basic_eps(net_income: float, preferred_dividends: float, avg_shares: float) -> float:
    """
    Calculate basic earnings per share (EPS)
    
    Args:
        net_income: Net income
        preferred_dividends: Preferred dividends
        avg_shares: Weighted average number of common shares outstanding
    
    Returns:
        Basic EPS
    """
    return (net_income - preferred_dividends) / avg_shares


def diluted_eps_preferred_conversion(net_income: float, weighted_shares: float, 
                                   new_shares: float) -> float:
    """
    Calculate diluted EPS with preferred share conversion
    
    Args:
        net_income: Net income
        weighted_shares: Weighted average number of common shares outstanding
        new_shares: Number of new common shares that would be issued at conversion
    
    Returns:
        Diluted EPS
    """
    return net_income / (weighted_shares + new_shares)


def diluted_eps_debt_conversion(net_income: float, after_tax_interest: float, 
                             preferred_dividends: float, weighted_shares: float, 
                             new_shares: float) -> float:
    """
    Calculate diluted EPS with convertible debt conversion
    
    Args:
        net_income: Net income
        after_tax_interest: After-tax interest on convertible debt
        preferred_dividends: Preferred dividends
        weighted_shares: Weighted average number of common shares outstanding
        new_shares: Number of new common shares that would be issued at conversion
    
    Returns:
        Diluted EPS
    """
    return (net_income + after_tax_interest - preferred_dividends) / (weighted_shares + new_shares)


def diluted_eps_options(net_income: float, preferred_dividends: float, 
                     weighted_shares: float, incremental_shares: float) -> float:
    """
    Calculate diluted EPS with options/warrants (treasury stock method)
    
    Args:
        net_income: Net income
        preferred_dividends: Preferred dividends
        weighted_shares: Weighted average number of common shares outstanding
        incremental_shares: Net incremental shares from options (accounting for shares 
                         that could be repurchased with cash received)
    
    Returns:
        Diluted EPS
    """
    return (net_income - preferred_dividends) / (weighted_shares + incremental_shares)


def net_profit_margin(net_income: float, revenue: float) -> float:
    """
    Calculate net profit margin
    
    Args:
        net_income: Net income
        revenue: Revenue
    
    Returns:
        Net profit margin (as a decimal)
    """
    return net_income / revenue


def gross_profit_margin(gross_profit: float, revenue: float) -> float:
    """
    Calculate gross profit margin
    
    Args:
        gross_profit: Gross profit
        revenue: Revenue
    
    Returns:
        Gross profit margin (as a decimal)
    """
    return gross_profit / revenue


def comprehensive_income(net_income: float, other_comprehensive_income: float) -> float:
    """
    Calculate comprehensive income
    
    Args:
        net_income: Net income
        other_comprehensive_income: Other comprehensive income
    
    Returns:
        Comprehensive income
    """
    return net_income + other_comprehensive_income


### UNDERSTANDING CASH FLOW STATEMENTS ###

def change_in_cash(cfo: float, cfi: float, cff: float) -> float:
    """
    Calculate change in cash and cash equivalents
    
    Args:
        cfo: Cash flow from operating activities
        cfi: Cash flow from investing activities
        cff: Cash flow from financing activities
    
    Returns:
        Change in cash and cash equivalents
    """
    return cfo / operating_income


def cash_flow_per_share(cfo: float, preferred_dividends: float, common_shares: float) -> float:
    """
    Calculate cash flow per share
    
    Args:
        cfo: Cash flow from operations
        preferred_dividends: Preferred dividends
        common_shares: Number of common shares outstanding
    
    Returns:
        Cash flow per share
    """
    return (cfo - preferred_dividends) / common_shares


def debt_coverage_ratio(cfo: float, total_debt: float) -> float:
    """
    Calculate debt coverage ratio
    
    Args:
        cfo: Cash flow from operations
        total_debt: Total debt
    
    Returns:
        Debt coverage ratio
    """
    return cfo / total_debt


def interest_coverage_ratio(cfo: float, interest_paid: float, taxes_paid: float = 0) -> float:
    """
    Calculate interest coverage ratio
    
    Args:
        cfo: Cash flow from operations
        interest_paid: Interest paid
        taxes_paid: Taxes paid
    
    Returns:
        Interest coverage ratio
    """
    return (cfo + interest_paid + taxes_paid) / interest_paid


def reinvestment_coverage_ratio(cfo: float, long_term_asset_purchases: float) -> float:
    """
    Calculate reinvestment coverage ratio
    
    Args:
        cfo: Cash flow from operations
        long_term_asset_purchases: Cash paid for long term assets
    
    Returns:
        Reinvestment coverage ratio
    """
    return cfo / long_term_asset_purchases


def debt_payment_coverage_ratio(cfo: float, long_term_debt_repayment: float) -> float:
    """
    Calculate debt payment coverage ratio
    
    Args:
        cfo: Cash flow from operations
        long_term_debt_repayment: Cash paid for long term debt repayment
    
    Returns:
        Debt payment coverage ratio
    """
    return cfo / long_term_debt_repayment


def dividend_payment_coverage_ratio(cfo: float, dividends_paid: float) -> float:
    """
    Calculate dividend payment coverage ratio
    
    Args:
        cfo: Cash flow from operations
        dividends_paid: Dividends paid
    
    Returns:
        Dividend payment coverage ratio
    """
    return cfo / dividends_paid


def investing_financing_coverage_ratio(cfo: float, investing_financing_outflows: float) -> float:
    """
    Calculate investing and financing coverage ratio
    
    Args:
        cfo: Cash flow from operations
        investing_financing_outflows: Cash outflows for investing and financing activities
    
    Returns:
        Investing and financing coverage ratio
    """
    return cfo / investing_financing_outflows


### FINANCIAL ANALYSIS TECHNIQUES ###

def inventory_turnover(cogs: float, average_inventory: float) -> float:
    """
    Calculate inventory turnover
    
    Args:
        cogs: Cost of goods sold
        average_inventory: Average inventory
    
    Returns:
        Inventory turnover
    """
    return cogs / average_inventory


def days_of_inventory_on_hand(average_inventory: float, cogs: float, days: int = 365) -> float:
    """
    Calculate days of inventory on hand
    
    Args:
        average_inventory: Average inventory
        cogs: Cost of goods sold
        days: Number of days in period (default 365)
    
    Returns:
        Days of inventory on hand
    """
    return (average_inventory / cogs) * days


def receivables_turnover(revenue: float, average_receivables: float) -> float:
    """
    Calculate receivables turnover
    
    Args:
        revenue: Revenue
        average_receivables: Average receivables
    
    Returns:
        Receivables turnover
    """
    return revenue / average_receivables


def days_of_sales_outstanding(average_receivables: float, revenue: float, 
                           days: int = 365) -> float:
    """
    Calculate days of sales outstanding (DSO)
    
    Args:
        average_receivables: Average receivables
        revenue: Revenue
        days: Number of days in period (default 365)
    
    Returns:
        Days of sales outstanding
    """
    return (average_receivables / revenue) * days


def payables_turnover(purchases: float, average_trade_payables: float) -> float:
    """
    Calculate payables turnover
    
    Args:
        purchases: Purchases
        average_trade_payables: Average trade payables
    
    Returns:
        Payables turnover
    """
    return purchases / average_trade_payables


def days_of_payables(average_trade_payables: float, purchases: float, 
                  days: int = 365) -> float:
    """
    Calculate days of payables
    
    Args:
        average_trade_payables: Average trade payables
        purchases: Purchases
        days: Number of days in period (default 365)
    
    Returns:
        Days of payables
    """
    return (average_trade_payables / purchases) * days


def working_capital_turnover(revenue: float, average_working_capital: float) -> float:
    """
    Calculate working capital turnover
    
    Args:
        revenue: Revenue
        average_working_capital: Average working capital
    
    Returns:
        Working capital turnover
    """
    return revenue / average_working_capital


def fixed_asset_turnover(revenue: float, average_fixed_assets: float) -> float:
    """
    Calculate fixed asset turnover
    
    Args:
        revenue: Revenue
        average_fixed_assets: Average net fixed assets
    
    Returns:
        Fixed asset turnover
    """
    return revenue / average_fixed_assets


def total_asset_turnover(revenue: float, average_total_assets: float) -> float:
    """
    Calculate total asset turnover
    
    Args:
        revenue: Revenue
        average_total_assets: Average total assets
    
    Returns:
        Total asset turnover
    """
    return revenue / average_total_assets


def current_ratio(current_assets: float, current_liabilities: float) -> float:
    """
    Calculate current ratio
    
    Args:
        current_assets: Current assets
        current_liabilities: Current liabilities
    
    Returns:
        Current ratio
    """
    return current_assets / current_liabilities


def quick_ratio(cash: float, marketable_securities: float, receivables: float, 
              current_liabilities: float) -> float:
    """
    Calculate quick (acid test) ratio
    
    Args:
        cash: Cash
        marketable_securities: Short term marketable securities
        receivables: Receivables
        current_liabilities: Current liabilities
    
    Returns:
        Quick ratio
    """
    return (cash + marketable_securities + receivables) / current_liabilities


def cash_ratio(cash: float, marketable_securities: float, current_liabilities: float) -> float:
    """
    Calculate cash ratio
    
    Args:
        cash: Cash
        marketable_securities: Short term marketable securities
        current_liabilities: Current liabilities
    
    Returns:
        Cash ratio
    """
    return (cash + marketable_securities) / current_liabilities


def defensive_interval_ratio(cash: float, marketable_securities: float, receivables: float, 
                          daily_expenditures: float) -> float:
    """
    Calculate defensive interval ratio
    
    Args:
        cash: Cash
        marketable_securities: Short term marketable securities
        receivables: Receivables
        daily_expenditures: Daily cash expenditures
    
    Returns:
        Defensive interval ratio
    """
    return (cash + marketable_securities + receivables) / daily_expenditures


def cash_conversion_cycle(days_inventory: float, days_receivables: float, 
                        days_payables: float) -> float:
    """
    Calculate cash conversion cycle (net operating cycle)
    
    Args:
        days_inventory: Days of inventory on hand
        days_receivables: Days of sales outstanding
        days_payables: Days of payables
    
    Returns:
        Cash conversion cycle
    """
    return days_inventory + days_receivables - days_payables


def debt_to_assets_ratio(total_debt: float, total_assets: float) -> float:
    """
    Calculate debt-to-assets ratio
    
    Args:
        total_debt: Total debt
        total_assets: Total assets
    
    Returns:
        Debt-to-assets ratio
    """
    return total_debt / total_assets


def debt_to_capital_ratio(total_debt: float, total_shareholders_equity: float) -> float:
    """
    Calculate debt-to-capital ratio
    
    Args:
        total_debt: Total debt
        total_shareholders_equity: Total shareholders' equity
    
    Returns:
        Debt-to-capital ratio
    """
    return total_debt / (total_debt + total_shareholders_equity)


def debt_to_equity_ratio(total_debt: float, total_shareholders_equity: float) -> float:
    """
    Calculate debt-to-equity ratio
    
    Args:
        total_debt: Total debt
        total_shareholders_equity: Total shareholders' equity
    
    Returns:
        Debt-to-equity ratio
    """
    return total_debt / total_shareholders_equity


def financial_leverage_ratio(average_total_assets: float, average_total_equity: float) -> float:
    """
    Calculate financial leverage ratio
    
    Args:
        average_total_assets: Average total assets
        average_total_equity: Average total equity
    
    Returns:
        Financial leverage ratio
    """
    return average_total_assets / average_total_equity


def debt_to_ebitda_ratio(total_debt: float, ebitda: float) -> float:
    """
    Calculate debt-to-EBITDA ratio
    
    Args:
        total_debt: Total debt
        ebitda: EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization)
    
    Returns:
        Debt-to-EBITDA ratio
    """
    return total_debt / ebitda


def interest_coverage_ratio_ebit(ebit: float, interest_payments: float) -> float:
    """
    Calculate interest coverage ratio using EBIT
    
    Args:
        ebit: EBIT (Earnings Before Interest and Taxes)
        interest_payments: Interest payments
    
    Returns:
        Interest coverage ratio
    """
    return ebit / interest_payments


def fixed_charge_coverage(ebit: float, lease_payments: float, interest_payments: float) -> float:
    """
    Calculate fixed charge coverage
    
    Args:
        ebit: EBIT (Earnings Before Interest and Taxes)
        lease_payments: Lease payments
        interest_payments: Interest payments
    
    Returns:
        Fixed charge coverage
    """
    return (ebit + lease_payments) / (interest_payments + lease_payments)


def gross_profit_margin(gross_profit: float, revenue: float) -> float:
    """
    Calculate gross profit margin
    
    Args:
        gross_profit: Gross profit
        revenue: Revenue
    
    Returns:
        Gross profit margin
    """
    return gross_profit / revenue


def operating_profit_margin(operating_income: float, revenue: float) -> float:
    """
    Calculate operating profit margin
    
    Args:
        operating_income: Operating income
        revenue: Revenue
    
    Returns:
        Operating profit margin
    """
    return operating_income / revenue


def pretax_margin(earnings_before_tax: float, revenue: float) -> float:
    """
    Calculate pretax margin
    
    Args:
        earnings_before_tax: Earnings before tax
        revenue: Revenue
    
    Returns:
        Pretax margin
    """
    return earnings_before_tax / revenue


def net_profit_margin(net_income: float, revenue: float) -> float:
    """
    Calculate net profit margin
    
    Args:
        net_income: Net income
        revenue: Revenue
    
    Returns:
        Net profit margin
    """
    return net_income / revenue


def operating_roa(operating_income: float, average_total_assets: float) -> float:
    """
    Calculate operating return on assets (ROA)
    
    Args:
        operating_income: Operating income
        average_total_assets: Average total assets
    
    Returns:
        Operating ROA
    """
    return operating_income / average_total_assets


def return_on_assets(net_income: float, average_total_assets: float) -> float:
    """
    Calculate return on assets (ROA)
    
    Args:
        net_income: Net income
        average_total_assets: Average total assets
    
    Returns:
        ROA
    """
    return net_income / average_total_assets


def return_on_total_capital(ebit: float, debt_and_equity: float) -> float:
    """
    Calculate return on total capital
    
    Args:
        ebit: EBIT (Earnings Before Interest and Taxes)
        debt_and_equity: Total debt and equity
    
    Returns:
        Return on total capital
    """
    return ebit / debt_and_equity


def return_on_equity(net_income: float, average_total_equity: float) -> float:
    """
    Calculate return on equity (ROE)
    
    Args:
        net_income: Net income
        average_total_equity: Average total equity
    
    Returns:
        ROE
    """
    return net_income / average_total_equity


def return_on_common_equity(net_income: float, preferred_dividends: float, 
                          average_common_equity: float) -> float:
    """
    Calculate return on common equity
    
    Args:
        net_income: Net income
        preferred_dividends: Preferred dividends
        average_common_equity: Average common equity
    
    Returns:
        Return on common equity
    """
    return (net_income - preferred_dividends) / average_common_equity


def dupont_roe_basic(net_income: float, revenue: float, average_total_assets: float, 
                  average_total_equity: float) -> float:
    """
    Calculate ROE using basic DuPont analysis
    
    Args:
        net_income: Net income
        revenue: Revenue
        average_total_assets: Average total assets
        average_total_equity: Average total equity
    
    Returns:
        ROE
    """
    return (net_income / revenue) * (revenue / average_total_assets) * (average_total_assets / average_total_equity)


def dividend_payout_ratio(dividends: float, earnings: float) -> float:
    """
    Calculate dividend payout ratio
    
    Args:
        dividends: Dividends
        earnings: Earnings
    
    Returns:
        Dividend payout ratio
    """
    return dividends / earnings


def retention_rate(dividend_payout_ratio: float) -> float:
    """
    Calculate retention rate
    
    Args:
        dividend_payout_ratio: Dividend payout ratio
    
    Returns:
        Retention rate
    """
    return 1 - dividend_payout_ratio


def sustainable_growth_rate(retention_rate: float, roe: float) -> float:
    """
    Calculate sustainable growth rate
    
    Args:
        retention_rate: Retention rate
        roe: Return on equity
    
    Returns:
        Sustainable growth rate
    """
    return retention_rate * roe


def ebitda_interest_coverage(ebitda: float, gross_interest: float) -> float:
    """
    Calculate EBITDA interest coverage
    
    Args:
        ebitda: EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization)
        gross_interest: Gross interest
    
    Returns:
        EBITDA interest coverage
    """
    return ebitda / gross_interest


def ffo_to_debt(ffo: float, total_debt: float) -> float:
    """
    Calculate funds from operations to debt ratio
    
    Args:
        ffo: Funds from operations
        total_debt: Total debt
    
    Returns:
        FFO to debt ratio
    """
    return ffo / total_debt


def free_operating_cash_flow_to_debt(cfo: float, capital_expenditures: float, 
                                   total_debt: float) -> float:
    """
    Calculate free operating cash flow to debt ratio
    
    Args:
        cfo: Cash flow from operations
        capital_expenditures: Capital expenditures
        total_debt: Total debt
    
    Returns:
        Free operating cash flow to debt ratio
    """
    return (cfo - capital_expenditures) / total_debt


def return_on_capital(ebit: float, average_capital: float) -> float:
    """
    Calculate return on capital
    
    Args:
        ebit: EBIT (Earnings Before Interest and Taxes)
        average_capital: Average capital
    
    Returns:
        Return on capital
    """
    return ebit / average_capital


### LONG-LIVED ASSETS ###

def straight_line_depreciation(cost: float, residual_value: float, useful_life: float) -> float:
    """
    Calculate straight line depreciation expense
    
    Args:
        cost: Cost (initial value)
        residual_value: Residual value
        useful_life: Estimated useful life
    
    Returns:
        Depreciation expense
    """
    return (cost - residual_value) / useful_life


def double_declining_balance_depreciation(cost: float, accumulated_depreciation: float, 
                                        useful_life: float) -> float:
    """
    Calculate double declining balance depreciation expense
    
    Args:
        cost: Cost (initial value)
        accumulated_depreciation: Accumulated depreciation
        useful_life: Estimated useful life
    
    Returns:
        Depreciation expense
    """
    return (2 / useful_life) * (cost - accumulated_depreciation)


def units_of_production_depreciation(cost: float, salvage_value: float, 
                                   useful_life_units: float, output_units: float) -> float:
    """
    Calculate units of production depreciation expense
    
    Args:
        cost: Cost (initial value)
        salvage_value: Salvage value
        useful_life_units: Useful life in units
        output_units: Output units for the period
    
    Returns:
        Depreciation expense
    """
    return ((cost - salvage_value) / useful_life_units) * output_units


def impairment_loss_ifrs(carrying_value: float, recoverable_amount: float) -> float:
    """
    Calculate impairment loss under IFRS
    
    Args:
        carrying_value: Carrying value
        recoverable_amount: Recoverable amount
    
    Returns:
        Impairment loss
    """
    if carrying_value > recoverable_amount:
        return carrying_value - recoverable_amount
    else:
        return 0


def impairment_loss_us_gaap(carrying_value: float, fair_value: float, 
                          undiscounted_cash_flows: float) -> float:
    """
    Calculate impairment loss under US GAAP
    
    Args:
        carrying_value: Carrying value
        fair_value: Fair value
        undiscounted_cash_flows: Sum of undiscounted cash flow
    
    Returns:
        Impairment loss
    """
    if carrying_value > undiscounted_cash_flows:
        return carrying_value - fair_value
    else:
        return 0


### INCOME TAXES ###

def income_tax_expense(income_tax_payable: float, change_in_dtl: float, 
                     change_in_dta: float) -> float:
    """
    Calculate income tax expense
    
    Args:
        income_tax_payable: Income tax payable
        change_in_dtl: Change in deferred tax liability
        change_in_dta: Change in deferred tax asset
    
    Returns:
        Income tax expense
    """
    return income_tax_payable + change_in_dtl - change_in_dta


def deferred_tax_asset_liability(carrying_value: float, tax_base: float, 
                               tax_rate: float) -> float:
    """
    Calculate deferred tax asset or liability
    
    Args:
        carrying_value: Carrying value
        tax_base: Tax base
        tax_rate: Tax rate
    
    Returns:
        Deferred tax asset (if negative) or liability (if positive)
    """
    return (carrying_value - tax_base) * tax_rate


### NON-CURRENT (LONG-TERM) LIABILITIES ###

def coupon_payment(coupon_rate: float, face_value: float) -> float:
    """
    Calculate coupon payment for a bond
    
    Args:
        coupon_rate: Coupon rate (as a decimal)
        face_value: Face value (par value)
    
    Returns:
        Coupon payment
    """
    return coupon_rate * face_value


def ending_bond_book_value(beginning_bond_book_value: float, interest_expense: float, 
                         coupon_payment: float) -> float:
    """
    Calculate ending book value of a bond
    
    Args:
        beginning_bond_book_value: Beginning book value of bond
        interest_expense: Interest expense
        coupon_payment: Coupon payment
    
    Returns:
        Ending book value of bond
    """
    return beginning_bond_book_value + interest_expense - coupon_payment


def interest_expense_effective_rate(beginning_bond_value: float, market_interest_rate: float) -> float:
    """
    Calculate interest expense using effective interest rate method
    
    Args:
        beginning_bond_value: Beginning book value of bond
        market_interest_rate: Market interest rate at issuance (as a decimal)
    
    Returns:
        Interest expense
    """
    return beginning_bond_value * market_interest_rate


### FINANCIAL STATEMENT ANALYSIS: APPLICATIONS ###

def fifo_inventory(lifo_inventory: float, lifo_reserve: float) -> float:
    """
    Calculate FIFO inventory from LIFO inventory
    
    Args:
        lifo_inventory: LIFO inventory
        lifo_reserve: LIFO reserve
    
    Returns:
        FIFO inventory
    """
    return lifo_inventory + lifo_reserve


def fifo_cogs(lifo_cogs: float, change_in_lifo_reserve: float) -> float:
    """
    Calculate FIFO COGS from LIFO COGS
    
    Args:
        lifo_cogs: LIFO COGS
        change_in_lifo_reserve: Change in LIFO reserve
    
    Returns:
        FIFO COGS
    """
    return lifo_cogs - change_in_lifo_reserve


def asset_age(accumulated_depreciation: float, depreciation_expense: float) -> float:
    """
    Calculate average age of asset base
    
    Args:
        accumulated_depreciation: Accumulated depreciation
        depreciation_expense: Depreciation expense
    
    Returns:
        Average age of asset base
    """
    return accumulated_depreciation / depreciation_expense


def remaining_useful_life(net_ppe: float, depreciation_expense: float) -> float:
    """
    Calculate remaining useful life of asset base
    
    Args:
        net_ppe: Net property, plant, and equipment
        depreciation_expense: Depreciation expense
    
    Returns:
        Remaining useful life of asset base
    """
    return net_ppe / depreciation_expense


def average_useful_life(gross_ppe: float, depreciation_expense: float) -> float:
    """
    Calculate average useful life of assets at installation
    
    Args:
        gross_ppe: Gross property, plant, and equipment
        depreciation_expense: Depreciation expense
    
    Returns:
        Average useful life of assets at installation
    """
    return gross_ppe / depreciation_expense


def asset_renewal_rate(capex: float, gross_ppe_and_capex: float) -> float:
    """
    Calculate asset renewal rate
    
    Args:
        capex: Capital expenditures
        gross_ppe_and_capex: Gross property, plant, and equipment plus capex
    
    Returns:
        Asset renewal rate
    """
    return capex / gross_ppe_and_capex


def asset_growth_rate(capex: float, asset_disposal: float) -> float:
    """
    Calculate asset growth rate
    
    Args:
        capex: Capital expenditures
        asset_disposal: Asset disposal
    
    Returns:
        Asset growth rate
    """
    return capex / asset_disposal


###############################
# CORPORATE FINANCE
###############################

### CAPITAL BUDGETING ###

def net_present_value(cash_flows: List[float], rate: float, initial_outlay: float) -> float:
    """
    Calculate net present value (NPV)
    
    Args:
        cash_flows: List of future cash flows
        rate: Discount rate (as a decimal)
        initial_outlay: Initial investment (positive value)
    
    Returns:
        Net present value
    """
    npv = -initial_outlay
    for t, cf in enumerate(cash_flows, 1):
        npv += cf / ((1 + rate) ** t)
    
    return npv


def average_accounting_rate_of_return(avg_net_income: float, avg_book_value: float) -> float:
    """
    Calculate average accounting rate of return (AAR)
    
    Args:
        avg_net_income: Average net income
        avg_book_value: Average book value
    
    Returns:
        Average accounting rate of return
    """
    return avg_net_income / avg_book_value


def profitability_index(pv_future_cash_flows: float, initial_investment: float) -> float:
    """
    Calculate profitability index (PI)
    
    Args:
        pv_future_cash_flows: Present value of future cash flows
        initial_investment: Initial investment
    
    Returns:
        Profitability index
    """
    return pv_future_cash_flows / initial_investment


def irr_approximation(npv: float, initial_investment: float) -> float:
    """
    Approximate IRR using NPV
    
    Args:
        npv: Net present value
        initial_investment: Initial investment
    
    Returns:
        Approximate IRR
    """
    return 1 + (npv / initial_investment)


### COST OF CAPITAL ###

def weighted_average_cost_of_capital(wd: float, rd: float, tax_rate: float, wp: float, rp: float, 
                                    we: float, re: float) -> float:
    """
    Calculate weighted average cost of capital (WACC)
    
    Args:
        wd: Weight of debt
        rd: Cost of debt (before tax)
        tax_rate: Tax rate
        wp: Weight of preferred stock
        rp: Cost of preferred stock
        we: Weight of equity
        re: Cost of equity
    
    Returns:
        WACC
    """
    return wd * rd * (1 - tax_rate) + wp * rp + we * re


def cost_of_preferred_stock(dividend: float, price: float) -> float:
    """
    Calculate cost of preferred stock
    
    Args:
        dividend: Preferred stock dividend per share
        price: Current preferred stock price per share
    
    Returns:
        Cost of preferred stock
    """
    return dividend / price


def cost_of_equity_capm(risk_free_rate: float, beta: float, market_risk_premium: float) -> float:
    """
    Calculate cost of equity using CAPM
    
    Args:
        risk_free_rate: Risk-free rate (as a decimal)
        beta: Beta of the stock
        market_risk_premium: Market risk premium (Rm - Rf)
    
    Returns:
        Cost of equity
    """
    return risk_free_rate + beta * market_risk_premium


def cost_of_equity_ddm(dividend: float, price: float, growth_rate: float) -> float:
    """
    Calculate cost of equity using dividend discount model (Gordon Growth Model)
    
    Args:
        dividend: Current dividend per share (D₀)
        price: Current stock price per share
        growth_rate: Expected dividend growth rate
    
    Returns:
        Cost of equity
    """
    dividend_next_period = dividend * (1 + growth_rate)
    return (dividend_next_period / price) + growth_rate


def sustainable_growth_rate(retention_rate: float, roe: float) -> float:
    """
    Calculate sustainable growth rate
    
    Args:
        retention_rate: Retention rate (1 - dividend payout ratio)
        roe: Return on equity
    
    Returns:
        Sustainable growth rate
    """
    return retention_rate * roe


def cost_of_equity_bond_yield_plus_risk_premium(debt_rate: float, risk_premium: float) -> float:
    """
    Calculate cost of equity using bond yield plus risk premium approach
    
    Args:
        debt_rate: Bond yield (cost of debt)
        risk_premium: Risk premium
    
    Returns:
        Cost of equity
    """
    return debt_rate + risk_premium


def project_beta(debt_weight: float, tax_rate: float, equity_weight: float, 
               debt_beta: float, equity_beta: float) -> float:
    """
    Calculate project beta
    
    Args:
        debt_weight: Weight of debt
        tax_rate: Tax rate
        equity_weight: Weight of equity
        debt_beta: Beta of debt
        equity_beta: Beta of equity
    
    Returns:
        Project beta
    """
    return ((debt_weight * (1 - tax_rate)) / (debt_weight * (1 - tax_rate) + equity_weight)) * debt_beta + \
           ((equity_weight) / (debt_weight * (1 - tax_rate) + equity_weight)) * equity_beta


def project_beta_zero_debt_beta(debt_to_equity: float, tax_rate: float, equity_beta: float) -> float:
    """
    Calculate project beta when debt beta is zero
    
    Args:
        debt_to_equity: Debt-to-equity ratio
        tax_rate: Tax rate
        equity_beta: Beta of equity
    
    Returns:
        Project beta
    """
    return equity_beta / (1 + (1 - tax_rate) * (debt_to_equity))


def equity_beta_from_project_beta(project_beta: float, debt_to_equity: float, 
                                tax_rate: float) -> float:
    """
    Calculate equity beta from project beta
    
    Args:
        project_beta: Project beta
        debt_to_equity: Debt-to-equity ratio
        tax_rate: Tax rate
    
    Returns:
        Equity beta
    """
    return project_beta * (1 + (1 - tax_rate) * (debt_to_equity))


def country_risk_premium(sovereign_yield_spread: float, equity_volatility: float, 
                       bond_volatility: float) -> float:
    """
    Calculate country risk premium
    
    Args:
        sovereign_yield_spread: Sovereign yield spread
        equity_volatility: Annualized standard deviation of equity index
        bond_volatility: Annualized standard deviation of sovereign bond market
    
    Returns:
        Country risk premium
    """
    return sovereign_yield_spread * (equity_volatility / bond_volatility)

def ending_accounts_receivable(beginning_ar: float, revenue: float, 
                             cash_from_customers: float) -> float:
    """
    Calculate ending accounts receivable
    
    Args:
        beginning_ar: Beginning accounts receivable
        revenue: Revenue
        cash_from_customers: Cash collected from customers
    
    Returns:
        Ending accounts receivable
    """
    return beginning_ar + revenue - cash_from_customers


def ending_accounts_payable(beginning_ap: float, purchases: float, 
                          cash_to_suppliers: float) -> float:
    """
    Calculate ending accounts payable
    
    Args:
        beginning_ap: Beginning accounts payable
        purchases: Purchases
        cash_to_suppliers: Cash paid to suppliers
    
    Returns:
        Ending accounts payable
    """
    return beginning_ap + purchases - cash_to_suppliers


def ending_inventory(beginning_inventory: float, purchases: float, 
                   cost_of_goods_sold: float) -> float:
    """
    Calculate ending inventory
    
    Args:
        beginning_inventory: Beginning inventory
        purchases: Purchases
        cost_of_goods_sold: Cost of goods sold
    
    Returns:
        Ending inventory
    """
    return beginning_inventory + purchases - cost_of_goods_sold


def ending_wages_payable(beginning_wages_payable: float, wages_expense: float, 
                      cash_paid_to_employees: float) -> float:
    """
    Calculate ending wages payable
    
    Args:
        beginning_wages_payable: Beginning wages payable
        wages_expense: Wages expense
        cash_paid_to_employees: Cash paid to employees
    
    Returns:
        Ending wages payable
    """
    return beginning_wages_payable + wages_expense - cash_paid_to_employees


def ending_interest_payable(beginning_interest_payable: float, interest_expense: float, 
                         cash_paid_for_interest: float) -> float:
    """
    Calculate ending interest payable
    
    Args:
        beginning_interest_payable: Beginning interest payable
        interest_expense: Interest expense
        cash_paid_for_interest: Cash paid for interest
    
    Returns:
        Ending interest payable
    """
    return beginning_interest_payable + interest_expense - cash_paid_for_interest


def ending_retained_earnings(beginning_retained_earnings: float, net_income: float, 
                          dividends_paid: float) -> float:
    """
    Calculate ending retained earnings
    
    Args:
        beginning_retained_earnings: Beginning retained earnings
        net_income: Net income
        dividends_paid: Dividends paid
    
    Returns:
        Ending retained earnings
    """
    return beginning_retained_earnings + net_income - dividends_paid


def free_cash_flow_to_firm(net_income: float, non_cash_charges: float, 
                        after_tax_interest: float, capex: float, 
                        working_capital_investment: float) -> float:
    """
    Calculate free cash flow to the firm (FCFF)
    
    Args:
        net_income: Net income
        non_cash_charges: Non-cash charges
        after_tax_interest: After-tax interest (interest * (1 - tax rate))
        capex: Capital expenditures
        working_capital_investment: Working capital expenditure
    
    Returns:
        Free cash flow to the firm
    """
    return net_income + non_cash_charges + after_tax_interest - capex - working_capital_investment


def free_cash_flow_to_equity(cfo: float, capex: float, net_borrowing: float) -> float:
    """
    Calculate free cash flow to equity (FCFE)
    
    Args:
        cfo: Cash flow from operations
        capex: Capital expenditures
        net_borrowing: Net borrowing (debt issued - debt repaid)
    
    Returns:
        Free cash flow to equity
    """
    return cfo - capex + net_borrowing


def cash_flow_to_revenue_ratio(cfo: float, net_revenue: float) -> float:
    """
    Calculate cash flow to revenue ratio
    
    Args:
        cfo: Cash flow from operations
        net_revenue: Net revenue
    
    Returns:
        Cash flow to revenue ratio
    """
    return cfo / net_revenue


def cash_return_on_assets(cfo: float, average_total_assets: float) -> float:
    """
    Calculate cash return on assets
    
    Args:
        cfo: Cash flow from operations
        average_total_assets: Average total assets
    
    Returns:
        Cash return on assets
    """
    return cfo / average_total_assets


def cash_return_on_equity(cfo: float, average_shareholders_equity: float) -> float:
    """
    Calculate cash return on equity
    
    Args:
        cfo: Cash flow from operations
        average_shareholders_equity: Average shareholders' equity
    
    Returns:
        Cash return on equity
    """
    return cfo / average_shareholders_equity

def t_test_two_means(mean1: float, mean2: float, std1: float, std2: float, 
                    n1: int, n2: int, equal_var: bool = False) -> float:
    """
    Calculate t-test statistic for difference between two population means
    
    Args:
        mean1: Sample mean of population 1
        mean2: Sample mean of population 2
        std1: Sample standard deviation of population 1
        std2: Sample standard deviation of population 2
        n1: Sample size of population 1
        n2: Sample size of population 2
        equal_var: If True, assume equal population variances
    
    Returns:
        t-test statistic
    """
    if equal_var:
        # Pooled standard deviation
        sp_squared = ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
        # Calculate t-statistic with pooled variance
        t_stat = (mean1 - mean2) / math.sqrt(sp_squared * (1/n1 + 1/n2))
        # Degrees of freedom = n1 + n2 - 2
    else:
        # Welch's t-test (unequal variances)
        # Calculate denominator (standard error of difference between means)
        denominator = math.sqrt((std1**2 / n1) + (std2**2 / n2))
        # Calculate t-statistic
        t_stat = (mean1 - mean2) / denominator
        # Note: Welch-Satterthwaite degrees of freedom not calculated here
        # df = ((std1**2/n1 + std2**2/n2)**2) / ((std1**2/n1)**2/(n1-1) + (std2**2/n2)**2/(n2-1))
    
    return t_stat