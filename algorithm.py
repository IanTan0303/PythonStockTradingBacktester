import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

def download_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        group_by=None  # avoid MultiIndex when possible
    )

    if isinstance(stock_data.columns, pd.MultiIndex):
        # Flatten MultiIndex to get 'Close', 'Open', etc.
        stock_data.columns = stock_data.columns.map(lambda x: x[1])

    return stock_data

def generate_signals(data, short_window=40, long_window=100):
    signals = pd.DataFrame(index=data.index)
    signals['Close'] = data['Close']  # ✅ Needed later for trade logs
    signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    
    signals['signal'] = 0.0
    signals.loc[signals.index[short_window:], 'signal'] = np.where(
        signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0
    )
    
    signals['positions'] = signals['signal'].diff()

    return signals

def build_trade_log(signals):
    trades = signals[signals['positions'].isin([1, -1])]
    trades = trades.copy()
    
    trade_log = []
    entry_date = None
    entry_price = None

    for date, row in trades.iterrows():
        if row['positions'] == 1.0:  # Buy
            entry_date = date
            entry_price = row['Close']
        elif row['positions'] == -1.0 and entry_price is not None:  # Sell
            exit_date = date
            exit_price = row['Close']
            return_pct = ((exit_price - entry_price) / entry_price) * 100

            trade_log.append({
                'Entry Date': entry_date,
                'Exit Date': exit_date,
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'Return (%)': round(return_pct, 2)
            })

            entry_date = None
            entry_price = None

    return pd.DataFrame(trade_log)


def backtest_strategy(signals, data, initial_capital=100000):
    close_prices = data["Close"]

    # Execute each signal at the next trading day's closing price.
    executed_signal = signals["signal"].shift(1).fillna(0)

    positions = pd.DataFrame(index=signals.index)
    positions["stock"] = 100 * executed_signal

    portfolio = pd.DataFrame(index=signals.index)
    portfolio["stock"] = positions["stock"] * close_prices

    # Calculate the cash spent or received when the position changes.
    position_changes = positions["stock"].diff().fillna(
        positions["stock"]
    )
    trade_cost = position_changes * close_prices

    portfolio["cash"] = initial_capital - trade_cost.cumsum()
    portfolio["total"] = portfolio["stock"] + portfolio["cash"]

    return portfolio

if __name__ == "__main__":
    # Configuration
    symbol = "AAPL"
    start_date = "2021-07-01"
    end_date = "2026-07-01"
    short_window = 40
    long_window = 100
    initial_capital = 100000

    # Download historical stock data
    data = download_stock_data(symbol, start_date, end_date)

    if data.empty:
        raise ValueError(
            f"No stock data found for {symbol}. Check the ticker or date range."
        )

    # Generate trading signals
    signals = generate_signals(
        data,
        short_window=short_window,
        long_window=long_window
    )

    # Record the positions actually executed, not just the signal dates.
    execution_signals = signals.copy()

    executed_signal = signals["signal"].shift(1).fillna(0)

    execution_signals["positions"] = (
        executed_signal.diff().fillna(executed_signal.iloc[0])
    )

    trade_log_df = build_trade_log(execution_signals)
    trade_log_df.to_csv("results/trade_log.csv", index=False)

    buy_signals = execution_signals[
        execution_signals["positions"] == 1.0
    ]
    sell_signals = execution_signals[
        execution_signals["positions"] == -1.0
    ]

    # Backtest portfolio
    portfolio = backtest_strategy(
        signals,
        data,
        initial_capital=initial_capital
    )

    # Calculate performance metrics
    final_portfolio = portfolio["total"].iloc[-1]
    total_return = (
        (final_portfolio - initial_capital) / initial_capital
    ) * 100

    num_trades = len(trade_log_df)

    if num_trades > 0:
        win_rate = (trade_log_df["Return (%)"] > 0).mean() * 100
        avg_return = trade_log_df["Return (%)"].mean()
    else:
        win_rate = 0.0
        avg_return = 0.0

    running_max = portfolio["total"].cummax()
    drawdown = (portfolio["total"] - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    # Print consolidated summary
    separator = "=" * 60

    print(f"\n{separator}")
    print("             STOCK TRADING BACKTEST SUMMARY")
    print(separator)

    print("\nCONFIGURATION")
    print(f"  Symbol              : {symbol}")
    print(f"  Period              : {start_date} to {end_date}")
    print("  Strategy            : Moving Average Crossover")
    print(f"  Short MA            : {short_window} days")
    print(f"  Long MA             : {long_window} days")
    print(f"  Initial Capital     : ${initial_capital:,.2f}")

    print("\nPERFORMANCE")
    print(f"  Final Portfolio     : ${final_portfolio:,.2f}")
    print(f"  Total Return        : {total_return:.2f}%")
    print(f"  Completed Trades    : {num_trades}")
    print(f"  Win Rate            : {win_rate:.2f}%")
    print(f"  Avg. Trade Return   : {avg_return:.2f}%")
    print(f"  Maximum Drawdown    : {max_drawdown:.2f}%")

    print("\nTRADE HISTORY")
    if trade_log_df.empty:
        print("  No completed trades.")
    else:
        display_trades = trade_log_df.copy()
        display_trades["Entry Date"] = pd.to_datetime(
            display_trades["Entry Date"]
        ).dt.strftime("%Y-%m-%d")
        display_trades["Exit Date"] = pd.to_datetime(
            display_trades["Exit Date"]
        ).dt.strftime("%Y-%m-%d")

        print(display_trades.to_string(index=False))

    print("\nOUTPUTS")
    print("  Trade log           : trade_log.csv")
    print(separator)
    print("Backtest completed.")
    print(f"{separator}\n")

    # Plot stock price, moving averages, and signals
    plt.figure(figsize=(12, 6))

    plt.plot(
        data["Close"],
        label=f"{symbol} Close Price",
        color="blue"
    )
    plt.plot(
        signals["short_mavg"],
        label=f"{short_window}-day MA",
        color="orange",
        alpha=0.7
    )
    plt.plot(
        signals["long_mavg"],
        label=f"{long_window}-day MA",
        color="magenta",
        alpha=0.7
    )

    plt.scatter(
        buy_signals.index,
        data["Close"].loc[buy_signals.index],
        label="Buy Signal",
        marker="^",
        color="green",
        s=100
    )
    plt.scatter(
        sell_signals.index,
        data["Close"].loc[sell_signals.index],
        label="Sell Signal",
        marker="v",
        color="red",
        s=100
    )

    plt.title(f"Trading Signals and {symbol} Price")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/trading_signals.png", dpi=150, bbox_inches="tight")
    plt.show()
