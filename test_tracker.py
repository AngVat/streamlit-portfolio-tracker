import os
import io
import pickle
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import time
import warnings
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import streamlit as st

warnings.filterwarnings('ignore')

# ===================== SIDEBAR CONFIGURATION =====================
st.sidebar.title("Portfolio Analysis Configuration")
separate_profits = st.sidebar.checkbox("Separate Realized & Unrealized Profits", value=True)
include_capital = st.sidebar.checkbox("Include Invested Capital", value=False)
currency_convert = st.sidebar.checkbox("Convert USD to EUR", value=True)
time_series_frequency = st.sidebar.selectbox(
    "Time Series Frequency",
    options=["Daily", "Weekly", "Monthly", "Yearly"],
    index=0
)
analysis_region = st.sidebar.selectbox("Analysis Region", options=["All", "US", "Greek"], index=0)

st.sidebar.markdown("---")
st.sidebar.write("This app downloads historical data, computes portfolio performance, and produces various plots.")

# ===================== CURRENCY CONVERSION SETUP =====================
if currency_convert:
    try:
        eurusd_data = yf.download("EURUSD=X", period="1d")['Close']
        eurusd_rate = float(eurusd_data.iloc[-1])
        conversion_factor = 1 / eurusd_rate  # Convert USD to EUR
        st.sidebar.write(f"Conversion factor (USD to EUR): {conversion_factor:.4f}")
    except Exception as e:
        st.sidebar.error(f"Failed to fetch conversion rate: {e}")
        conversion_factor = 1.0
else:
    conversion_factor = 1.0

# ===================== HELPER FUNCTIONS =====================
def is_us_stock(stock):
    """Heuristic: if ticker does not end with '.AT', consider it US-based."""
    return not stock.endswith('.AT')

def maybe_convert(value, stock):
    """Convert value from USD to EUR if enabled and stock is US-based."""
    if currency_convert and is_us_stock(stock):
        return value * conversion_factor
    return value

# ===================== SESSION STATE INITIALIZATION =====================
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = pd.DataFrame(columns=[
        'Date', 'Stock', 'Action', 'Quantity', 'Price per Share', 'Expenses', 'Total Cost'
    ])
if 'dividend_log' not in st.session_state:
    st.session_state.dividend_log = pd.DataFrame(columns=[
        'Date', 'Stock', 'Dividend Received'
    ])

# ===================== LOGGING FUNCTIONS =====================
def log_trade(date, stock, action, quantity, price_per_share, expenses):
    total_cost = (quantity * price_per_share + expenses) if action.lower() == 'buy' else -(quantity * price_per_share - expenses)
    trade = pd.DataFrame([{
        'Date': pd.to_datetime(date),
        'Stock': stock.upper(),
        'Action': action.capitalize(),
        'Quantity': quantity,
        'Price per Share': price_per_share,
        'Expenses': expenses,
        'Total Cost': total_cost
    }])
    st.session_state.trade_log = pd.concat([st.session_state.trade_log, trade], ignore_index=True)

def log_dividend(date, stock, dividend_received):
    dividend = pd.DataFrame([{
        'Date': pd.to_datetime(date),
        'Stock': stock.upper(),
        'Dividend Received': dividend_received
    }])
    st.session_state.dividend_log = pd.concat([st.session_state.dividend_log, dividend], ignore_index=True)

# ===================== EXPORT / IMPORT LOGS =====================
st.header("Export / Import Logs")

# Export logs: Combine trade and dividend logs into an Excel file with two sheets.
if st.button("Export Logs to Excel"):
    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
        st.session_state.trade_log.to_excel(writer, index=False, sheet_name="Trade Log")
        st.session_state.dividend_log.to_excel(writer, index=False, sheet_name="Dividend Log")
    towrite.seek(0)
    st.download_button(
        label="Download Logs as Excel",
        data=towrite,
        file_name="portfolio_logs.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Import logs: Upload CSV files for trade log and dividend log.
st.subheader("Import Logs")
trade_file = st.file_uploader("Upload Trade Log Excel", type=["xlsx"], key="trade_import")
if trade_file is not None:
    try:
        imported_trade_log = pd.read_excel(trade_file, sheet_name="Trade Log", parse_dates=['Date'])
        st.session_state.trade_log = imported_trade_log
        st.success("Trade log imported successfully!")
    except Exception as e:
        st.error(f"Error importing trade log: {e}")


dividend_file = st.file_uploader("Upload Dividend Log Excel", type=["xlsx"], key="dividend_import")
if dividend_file is not None:
    try:
        imported_dividend_log = pd.read_excel(dividend_file, sheet_name="Dividend Log", parse_dates=['Date'])
        st.session_state.dividend_log = imported_dividend_log
        st.success("Dividend log imported successfully!")
    except Exception as e:
        st.error(f"Error importing dividend log: {e}")


# ===================== DISPLAY LOGS =====================
st.subheader("Trade Log")
st.dataframe(st.session_state.trade_log.sort_values("Date"))
st.subheader("Dividend Log")
st.dataframe(st.session_state.dividend_log.sort_values("Date"))

# ===================== ANALYSIS REGION FILTERING =====================
trade_log = st.session_state.trade_log.copy()
dividend_log = st.session_state.dividend_log.copy()
if analysis_region in ["US", "Greek"]:
    if analysis_region == "US":
        trade_log = trade_log[trade_log['Stock'].apply(is_us_stock)]
        dividend_log = dividend_log[dividend_log['Stock'].apply(is_us_stock)]
    else:
        trade_log = trade_log[~trade_log['Stock'].apply(is_us_stock)]
        dividend_log = dividend_log[~dividend_log['Stock'].apply(is_us_stock)]

# ===================== FETCH HISTORICAL PRICE DATA =====================
trade_log['Date'] = pd.to_datetime(trade_log['Date']).dt.normalize()
dividend_log['Date'] = pd.to_datetime(dividend_log['Date']).dt.normalize()
if not trade_log.empty or not dividend_log.empty:
    earliest_date = min(
        trade_log['Date'].min() if not trade_log.empty else datetime.now(),
        dividend_log['Date'].min() if not dividend_log.empty else datetime.now()
    )
else:
    earliest_date = datetime.now() - relativedelta(years=1)
latest_date = datetime.now().date()

# Create a full daily date range
date_list = pd.date_range(start=earliest_date, end=latest_date, freq='D')

# Use user-entered tickers directly
stocks = trade_log['Stock'].unique()
tickers = stocks
start_date = earliest_date.strftime('%Y-%m-%d')
end_date = latest_date.strftime('%Y-%m-%d')

successful_tickers = []
failed_tickers = []
price_data = pd.DataFrame()

st.write("### Downloading Historical Price Data...")
def download_data(ticker, start_date, end_date, retries=3, delay=5):
    for i in range(retries):
        try:
            st.write(f"Downloading data for {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date)['Close']
            if data.empty:
                st.write(f"No data found for {ticker}")
                return None
            else:
                st.write(f"Downloaded data for {ticker}")
                return data
        except Exception as e:
            st.write(f"Attempt {i+1}: Failed to download data for {ticker}: {e}")
            time.sleep(delay)
    return None

for ticker in tickers:
    data = download_data(ticker, start_date, end_date)
    if data is not None:
        price_data[ticker] = data
        successful_tickers.append(ticker)
    else:
        failed_tickers.append(ticker)

if failed_tickers:
    st.write(f"Failed to download data for: {failed_tickers}")

stocks = [stock for stock in stocks if stock in successful_tickers]
trade_log = trade_log[trade_log['Stock'].isin(stocks)]
dividend_log = dividend_log[dividend_log['Stock'].isin(stocks)]

# ===================== PORTFOLIO CALCULATIONS =====================
def calculate_individual_pnl():
    all_individual_pnls = pd.DataFrame(columns=['Date', 'Stock', 'PnL', 'Investment'])
    for stock in trade_log['Stock'].unique():
        stock_trades = trade_log[trade_log['Stock'] == stock].sort_values(by='Date')
        buys = stock_trades[stock_trades['Action'] == 'Buy'].copy()
        sells = stock_trades[stock_trades['Action'] == 'Sell']
        individual_pnls = []
        for _, sell in sells.iterrows():
            sell_quantity = sell['Quantity']
            while sell_quantity > 0 and not buys.empty:
                buy = buys.iloc[0]
                matched_quantity = min(buy['Quantity'], sell_quantity)
                sell_quantity -= matched_quantity
                buys.at[buys.index[0], 'Quantity'] -= matched_quantity
                buy_price_per_share = buy['Price per Share'] + buy['Expenses'] / buy['Quantity']
                sell_price_per_share = sell['Price per Share'] - sell['Expenses'] / sell['Quantity']
                investment = matched_quantity * buy_price_per_share
                sell_proceeds = matched_quantity * sell_price_per_share
                pnl = sell_proceeds - investment
                investment_conv = maybe_convert(investment, stock)
                pnl_conv = maybe_convert(pnl, stock)
                individual_pnls.append({
                    'Date': sell['Date'],
                    'Stock': stock,
                    'PnL': round(pnl_conv, 2),
                    'Investment': round(investment_conv, 2)
                })
                if buys.iloc[0]['Quantity'] == 0:
                    buys = buys.iloc[1:]
        stock_individual_pnls = pd.DataFrame(individual_pnls)
        all_individual_pnls = pd.concat([all_individual_pnls, stock_individual_pnls], ignore_index=True)
    return all_individual_pnls.sort_values(by='Date')

def calculate_dividend_profit():
    dividend_profit = dividend_log.copy()
    if currency_convert:
        dividend_profit['Dividend Received'] = dividend_profit.apply(
            lambda row: maybe_convert(row['Dividend Received'], row['Stock']),
            axis=1
        )
    dividend_profit = dividend_profit.groupby('Stock').agg({'Dividend Received': sum})
    return dividend_profit.round(2)

def calculate_current_invested_capital_per_stock():
    invested_capital_per_stock = {}
    for stock in trade_log['Stock'].unique():
        stock_trades = trade_log[trade_log['Stock'] == stock].sort_values(by='Date')
        remaining_quantity = 0
        invested_capital = 0.0
        for _, trade in stock_trades.iterrows():
            if trade['Action'] == 'Buy':
                total_cost = trade['Quantity'] * (trade['Price per Share'] + trade['Expenses'] / trade['Quantity'])
                total_cost = maybe_convert(total_cost, stock)
                invested_capital += total_cost
                remaining_quantity += trade['Quantity']
            elif trade['Action'] == 'Sell':
                if remaining_quantity > 0:
                    cost_per_share = invested_capital / remaining_quantity
                    invested_capital -= cost_per_share * trade['Quantity']
                    remaining_quantity -= trade['Quantity']
        if stock in successful_tickers:
            try:
                current_price = yf.Ticker(stock).history(period="1d")['Close'].iloc[-1]
                current_price = maybe_convert(current_price, stock)
            except Exception:
                current_price = 0.0
        else:
            current_price = 0.0
        if remaining_quantity > 0:
            average_invested_price = invested_capital / remaining_quantity
            current_value = remaining_quantity * current_price
        else:
            average_invested_price = 0.0
            current_value = 0.0
        invested_capital_per_stock[stock] = {
            'No of Stocks': remaining_quantity,
            'Invested Capital': round(invested_capital, 2),
            'Average Invested Price': round(average_invested_price, 2),
            'Current Price': round(current_price, 2),
            'Current Value': round(current_value, 2)
        }
    df_invest = pd.DataFrame.from_dict(invested_capital_per_stock, orient='index').reset_index()
    df_invest.rename(columns={'index': 'Stock'}, inplace=True)
    if 'Invested Capital' in df_invest.columns:
        return df_invest[df_invest['Invested Capital'] > 0]
    else:
        return df_invest

def process_trades_and_dividends_up_to_date(date, trades, dividends):
    holdings = defaultdict(int)
    invested_cap = defaultdict(float)
    cumulative_realized = 0.0
    div_data = dividends[dividends['Date'] <= date].copy()
    if currency_convert:
        div_data.loc[:, 'Dividend Received'] = div_data.apply(
            lambda row: maybe_convert(row['Dividend Received'], row['Stock']),
            axis=1
        )
    cumulative_div = round(div_data['Dividend Received'].sum(), 2)
    trades_up_to_date = trades[trades['Date'] <= date].sort_values(by='Date')
    for _, trade in trades_up_to_date.iterrows():
        stock = trade['Stock']
        action = trade['Action']
        qty = trade['Quantity']
        price = trade['Price per Share']
        expenses = trade['Expenses']
        if action == 'Buy':
            total_cost = qty * (price + expenses / qty)
            total_cost = maybe_convert(total_cost, stock)
            invested_cap[stock] += total_cost
            holdings[stock] += qty
        elif action == 'Sell':
            if holdings[stock] > 0:
                cost_per_share = invested_cap[stock] / holdings[stock]
                sell_proceeds = qty * (price - expenses / qty)
                sell_proceeds = maybe_convert(sell_proceeds, stock)
                realized = sell_proceeds - (cost_per_share * qty)
                cumulative_realized += realized
                invested_cap[stock] -= cost_per_share * qty
                holdings[stock] -= qty
                if holdings[stock] == 0:
                    invested_cap[stock] = 0.0
    return holdings, invested_cap, cumulative_realized, cumulative_div

# Build time series of portfolio performance
dates, invested_caps, realized_profits, unrealized_profits, total_profits, dividends_list, pnl_results = [], [], [], [], [], [], []
for date in date_list:
    holdings, invested_cap, cum_realized, cum_div = process_trades_and_dividends_up_to_date(date, trade_log, dividend_log)
    date_str = date.strftime('%Y-%m-%d')
    try:
        prices_on_date = price_data.loc[date_str]
    except KeyError:
        try:
            prices_on_date = price_data.loc[:date_str].iloc[-1]
        except IndexError:
            prices_on_date = pd.Series()
    current_value = 0.0
    for stock in holdings:
        price = maybe_convert(prices_on_date.get(stock, 0.0), stock)
        current_value += holdings[stock] * price
    invested_total = round(sum(invested_cap.values()), 2)
    unrealized = round(current_value - invested_total, 2)
    tot_profit = round(cum_realized + unrealized, 2)
    pnl = round(tot_profit + cum_div, 2)
    dates.append(date)
    invested_caps.append(invested_total)
    realized_profits.append(cum_realized)
    unrealized_profits.append(unrealized)
    total_profits.append(tot_profit)
    dividends_list.append(cum_div)
    pnl_results.append(pnl)

df = pd.DataFrame({
    'Date': dates,
    'Invested Capital': invested_caps,
    'Realized Profit': realized_profits,
    'Unrealized Profit': unrealized_profits,
    'Total Profit': total_profits,
    'Dividends': dividends_list,
    'Resulting PNL': pnl_results
})
df['Date'] = df['Date'].dt.date
df.set_index('Date', inplace=True)
filtered_data = df[df.index >= pd.to_datetime("2023-01-01").date()]
st.write("### Time Series Data (from 2023-01-01)")
st.dataframe(filtered_data)

# Add option to export the final time series data
to_write = io.BytesIO()
filtered_data.to_excel(to_write, index=True, sheet_name="Portfolio Data")
to_write.seek(0)
st.download_button(
    label="Download Time Series Data as Excel",
    data=to_write,
    file_name="portfolio_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

filtered_data['Portfolio Value'] = (filtered_data['Invested Capital'] +
                                    filtered_data['Realized Profit'] +
                                    filtered_data['Dividends'] +
                                    filtered_data['Unrealized Profit'])
filtered_data.index = pd.to_datetime(filtered_data.index)
filtered_data['Daily Return'] = filtered_data['Portfolio Value'].pct_change().fillna(0)

# Resample time series based on the selected frequency
if time_series_frequency == "Weekly":
    freq_data = filtered_data.resample('W').last()
elif time_series_frequency == "Monthly":
    freq_data = filtered_data.resample('M').last()
elif time_series_frequency == "Yearly":
    freq_data = filtered_data.resample('A').last()
else:
    freq_data = filtered_data.copy()

# ===================== BENCHMARK DATA =====================
BENCHMARK_TICKER = "^GSPC"
benchmark_data = yf.download(BENCHMARK_TICKER, start=start_date, end=end_date)['Close']
if benchmark_data.empty:
    st.write(f"Warning: No benchmark data for {BENCHMARK_TICKER}.")
else:
    benchmark_data = benchmark_data.sort_index().ffill()
df_benchmark = benchmark_data.reindex(freq_data.index, method='ffill').ffill().fillna(method='bfill')
freq_data['Benchmark'] = df_benchmark
freq_data['Benchmark Return'] = freq_data['Benchmark'].pct_change().fillna(0)

# ===================== VISUALIZATIONS =====================
# Stacked bar chart for portfolio components
if include_capital:
    if separate_profits:
        stacked_data = freq_data[['Invested Capital', 'Dividends', 'Realized Profit', 'Unrealized Profit']]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        title = 'Invested Capital, Realized & Unrealized Profit, and Dividends Over Time'
    else:
        stacked_data = freq_data[['Dividends', 'Total Profit', 'Invested Capital']]
        colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
        title = 'Invested Capital, Total Profit, and Dividends Over Time'
else:
    if separate_profits:
        stacked_data = freq_data[['Dividends', 'Realized Profit', 'Unrealized Profit']]
        colors = ['#ff7f0e', '#2ca02c', '#d62728']
        title = 'Realized & Unrealized Profit and Dividends Over Time'
    else:
        stacked_data = freq_data[['Dividends', 'Total Profit']]
        colors = ['#ff7f0e', '#2ca02c']
        title = 'Total Profit and Dividends Over Time'

fig, ax = plt.subplots(figsize=(12, 6))
stacked_data.plot(kind='bar', stacked=True, color=colors, ax=ax)
plt.title(title)
plt.xlabel('Date')
plt.ylabel('Amount (€)')
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45, ha='right')
plt.legend(title='Components')
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()

# Realized profit & dividend per stock
all_individual_pnls = calculate_individual_pnl()
total_realized_pnl_by_stock = all_individual_pnls.groupby('Stock')['PnL'].sum().round(2)
dividend_profit = calculate_dividend_profit()
dividend_profit_by_stock = dividend_profit['Dividend Received']
combined_df = pd.DataFrame({
    'Realized Profit': total_realized_pnl_by_stock,
    'Dividends': dividend_profit_by_stock
}).fillna(0)
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.4
x_positions = range(len(combined_df.index))
ax.bar([x - width / 2 for x in x_positions],
       combined_df['Realized Profit'], width=width, label='Realized Profit')
ax.bar([x + width / 2 for x in x_positions],
       combined_df['Dividends'], width=width, label='Dividends')
ax.set_xticks(x_positions)
ax.set_xticklabels(combined_df.index, rotation=45, ha='right')
ax.set_ylabel('Amount (€)')
ax.set_title('Total Realized Profit & Dividends per Stock')
ax.legend()
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()

# Combined realized, dividends, and unrealized profit per stock
realized_pnl_by_stock = all_individual_pnls.groupby('Stock')['PnL'].sum().round(2)
current_invested_capital_per_stock = calculate_current_invested_capital_per_stock()
unrealized_profit_series = (current_invested_capital_per_stock['Current Value'] -
                            current_invested_capital_per_stock['Invested Capital']).round(2)
unrealized_profit_series.index = current_invested_capital_per_stock['Stock']
combined_df = pd.DataFrame({
    'Realized Profit': realized_pnl_by_stock,
    'Dividends': dividend_profit_by_stock,
    'Unrealized Profit': unrealized_profit_series
}).fillna(0)
combined_df['Total Profit'] = combined_df['Realized Profit'] + combined_df['Dividends']
total_profit_sum = combined_df['Total Profit'].sum()
combined_df['% of Total Profit'] = ((combined_df['Total Profit'] / total_profit_sum) * 100).round(2)
combined_df.sort_values('Total Profit', ascending=False, inplace=True)
combined_df = combined_df.round(2)
st.write("### Total Realized, Dividends, Unrealized, and % of Total Profit per Stock")
st.dataframe(combined_df)
fig, ax = plt.subplots(figsize=(10, 6))
x_positions = range(len(combined_df.index))
bar_width = 0.25
ax.bar([x - bar_width for x in x_positions], combined_df['Realized Profit'], width=bar_width, label='Realized Profit')
ax.bar(x_positions, combined_df['Dividends'], width=bar_width, label='Dividends')
ax.bar([x + bar_width for x in x_positions], combined_df['Unrealized Profit'], width=bar_width,
       label='Unrealized Profit')
ax.set_xticks(x_positions)
ax.set_xticklabels(combined_df.index, rotation=45, ha='right')
ax.set_ylabel('Amount (€)')
ax.set_title('Realized, Dividends & Unrealized Profit per Stock')
ax.legend()
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()

# Yearly totals for realized profit and dividends
all_individual_pnls = calculate_individual_pnl().copy()
all_individual_pnls['Year'] = all_individual_pnls['Date'].dt.year
realized_pnl_by_year = all_individual_pnls.groupby('Year')['PnL'].sum().round(2)
dividend_data = dividend_log.copy()
dividend_data['Year'] = dividend_data['Date'].dt.year
dividends_by_year = dividend_data.groupby('Year')['Dividend Received'].sum().round(2)
combined_yearly = pd.DataFrame({
    'Realized Profit': realized_pnl_by_year,
    'Dividends': dividends_by_year
}).fillna(0)
st.write("### Yearly Totals")
st.dataframe(combined_yearly)
fig, ax = plt.subplots(figsize=(8, 5))
years = combined_yearly.index
x_positions = range(len(years))
bar_width = 0.4
ax.bar([x - bar_width / 2 for x in x_positions],
       combined_yearly['Realized Profit'], width=bar_width, label='Realized Profit')
ax.bar([x + bar_width / 2 for x in x_positions],
       combined_yearly['Dividends'], width=bar_width, label='Dividends')
ax.set_xticks(x_positions)
ax.set_xticklabels(years, rotation=0)
ax.set_ylabel('Amount (€)')
ax.set_title('Realized Profit & Dividends per Year')
ax.legend()
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()

# Sector allocation pie chart (using current invested capital)
try:
    current_cap_df = calculate_current_invested_capital_per_stock()
    allocation_df = current_cap_df.copy()
    allocation_df['Sector'] = 'Unknown'
    allocation_df = allocation_df.groupby('Sector')['Invested Capital'].sum().reset_index()
    st.write("### Sector Allocation")
    st.dataframe(allocation_df)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(allocation_df['Invested Capital'], labels=allocation_df['Sector'], autopct='%1.1f%%',
           startangle=140)
    ax.set_title("Portfolio Allocation by Sector")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()
except Exception as e:
    st.write("Sector allocation could not be computed:", e)

# Portfolio Value vs. Benchmark
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(freq_data.index, freq_data['Portfolio Value'], label='Portfolio Value', linewidth=2)
ax.plot(freq_data.index, freq_data['Benchmark'], label='Benchmark', linewidth=2, linestyle='--')
ax.set_xlabel('Date')
ax.set_ylabel('Value (€)')
ax.set_title('Portfolio Value vs. Benchmark')
ax.legend()
ax.grid(True)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()

# Portfolio Drawdown Over Time
running_max = freq_data['Portfolio Value'].cummax()
drawdown = freq_data['Portfolio Value'] / running_max - 1
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(freq_data.index, drawdown, color='red', label='Drawdown')
ax.set_xlabel('Date')
ax.set_ylabel('Drawdown (%)')
ax.set_title('Portfolio Drawdown Over Time')
ax.legend()
ax.grid(True)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()

# Distribution of Daily Returns
fig, ax = plt.subplots(figsize=(12,6))
ax.hist(freq_data['Daily Return'], bins=50, edgecolor='black', alpha=0.7)
ax.set_xlabel('Daily Return')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Daily Returns')
ax.grid(True)
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()

# Portfolio vs. Benchmark Daily Returns (Scatter Plot)
fig, ax = plt.subplots(figsize=(12,6))
ax.scatter(freq_data['Benchmark Return'], freq_data['Daily Return'], alpha=0.6, color='purple')
ax.set_xlabel('Benchmark Daily Return')
ax.set_ylabel('Portfolio Daily Return')
ax.set_title('Portfolio vs. Benchmark Daily Returns')
ax.grid(True)
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()

# Cumulative Returns: Portfolio vs. Benchmark
log_portfolio_returns = np.log(freq_data['Portfolio Value'] / freq_data['Portfolio Value'].iloc[0])
log_benchmark_returns = np.log(freq_data['Benchmark'] / freq_data['Benchmark'].iloc[0])
cum_portfolio_return = np.exp(log_portfolio_returns) - 1
cum_benchmark_return = np.exp(log_benchmark_returns) - 1
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(freq_data.index, cum_portfolio_return, label='Portfolio Cumulative Return', linewidth=2)
ax.plot(freq_data.index, cum_benchmark_return, label='Benchmark Cumulative Return', linewidth=2, linestyle='--')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Return')
ax.set_title('Cumulative Returns: Portfolio vs. Benchmark')
ax.legend()
ax.grid(True)
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.tight_layout()
st.pyplot(plt.gcf())
plt.clf()

st.write("### End of Analysis")
st.write("NEW SECTION")
