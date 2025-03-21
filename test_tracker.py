import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.title("Investment Portfolio Tracker")

# ------------------ SESSION STATE INITIALIZATION ------------------
if 'trades' not in st.session_state:
    st.session_state.trades = pd.DataFrame(columns=["Date", "Stock", "Action", "Quantity", "Price"])
if 'dividends' not in st.session_state:
    st.session_state.dividends = pd.DataFrame(columns=["Date", "Stock", "Dividend"])

# ------------------ USER INPUT: ADD A TRADE ------------------
st.header("Add a Trade")
with st.form("trade_form", clear_on_submit=True):
    trade_date = st.date_input("Trade Date", datetime.today())
    trade_stock = st.text_input("Stock Ticker", "AAPL")
    trade_action = st.selectbox("Action", ["Buy", "Sell"])
    trade_quantity = st.number_input("Quantity", min_value=1, step=1, value=1)
    trade_price = st.number_input("Price per Share", min_value=0.0, value=0.0, format="%.2f")
    submitted_trade = st.form_submit_button("Add Trade")
    if submitted_trade:
        new_trade = pd.DataFrame({
            "Date": [trade_date],
            "Stock": [trade_stock.upper()],
            "Action": [trade_action],
            "Quantity": [trade_quantity],
            "Price": [trade_price]
        })
        st.session_state.trades = pd.concat([st.session_state.trades, new_trade], ignore_index=True)
        st.success("Trade added!")

# ------------------ USER INPUT: ADD A DIVIDEND ------------------
st.header("Add a Dividend")
with st.form("dividend_form", clear_on_submit=True):
    dividend_date = st.date_input("Dividend Date", datetime.today(), key="div_date")
    dividend_stock = st.text_input("Stock Ticker for Dividend", "AAPL", key="div_stock")
    dividend_amount = st.number_input("Dividend Received", min_value=0.0, value=0.0, format="%.2f", key="div_amount")
    submitted_dividend = st.form_submit_button("Add Dividend")
    if submitted_dividend:
        new_div = pd.DataFrame({
            "Date": [dividend_date],
            "Stock": [dividend_stock.upper()],
            "Dividend": [dividend_amount]
        })
        st.session_state.dividends = pd.concat([st.session_state.dividends, new_div], ignore_index=True)
        st.success("Dividend added!")

# ------------------ DISPLAY LOGS ------------------
st.subheader("Trade Log")
st.dataframe(st.session_state.trades.sort_values("Date"))
st.subheader("Dividend Log")
st.dataframe(st.session_state.dividends.sort_values("Date"))

# ------------------ PORTFOLIO SUMMARY ------------------
st.header("Portfolio Summary")
if st.session_state.trades.empty:
    st.info("No trades added yet. Please add some trades to see your portfolio summary.")
else:
    trades = st.session_state.trades.copy()
    trades["Date"] = pd.to_datetime(trades["Date"])
    # Group trades by stock to compute net quantity and net cost.
    summary = trades.groupby("Stock").apply(lambda df: pd.Series({
        "Net Quantity": df.apply(lambda row: row["Quantity"] if row["Action"]=="Buy" else -row["Quantity"], axis=1).sum(),
        "Net Cost": df.apply(lambda row: row["Quantity"] * row["Price"] if row["Action"]=="Buy" else -row["Quantity"] * row["Price"], axis=1).sum()
    })).reset_index()

    # Fetch current prices for each stock via yfinance.
    current_prices = {}
    current_values = []
    for stock in summary["Stock"]:
        try:
            ticker = yf.Ticker(stock)
            current_price = ticker.history(period="1d")["Close"].iloc[-1]
        except Exception:
            current_price = np.nan
        current_prices[stock] = current_price
        net_qty = summary.loc[summary["Stock"]==stock, "Net Quantity"].iloc[0]
        if not np.isnan(current_price):
            current_values.append(net_qty * current_price)
        else:
            current_values.append(np.nan)

    summary["Current Price"] = summary["Stock"].map(current_prices)
    summary["Current Value"] = current_values
    # Profit/Loss for current open positions (unrealized profit)
    summary["Unrealized Profit"] = summary["Current Value"] - summary["Net Cost"]

    st.dataframe(summary)

    total_portfolio_value = summary["Current Value"].sum()
    st.write(f"**Total Portfolio Value:** {total_portfolio_value:,.2f}")

# ------------------ REALIZED PNL CALCULATION ------------------
# This function uses FIFO matching to compute realized profit from sell trades.
def calculate_individual_pnl(trades_df):
    all_individual_pnls = pd.DataFrame(columns=["Date", "Stock", "PnL"])
    for stock in trades_df["Stock"].unique():
        stock_trades = trades_df[trades_df["Stock"]==stock].sort_values("Date")
        buys = stock_trades[stock_trades["Action"]=="Buy"].copy()
        sells = stock_trades[stock_trades["Action"]=="Sell"].copy()
        for _, sell in sells.iterrows():
            sell_qty = sell["Quantity"]
            while sell_qty > 0 and not buys.empty:
                buy = buys.iloc[0]
                matched_qty = min(buy["Quantity"], sell_qty)
                sell_qty -= matched_qty
                pnl = (sell["Price"] - buy["Price"]) * matched_qty
                new_row = pd.DataFrame({"Date": [sell["Date"]], "Stock": [stock], "PnL": [pnl]})
                all_individual_pnls = pd.concat([all_individual_pnls, new_row], ignore_index=True)
                # Update or drop the buy record
                buys.at[buy.name, "Quantity"] -= matched_qty
                if buys.at[buy.name, "Quantity"] == 0:
                    buys = buys.drop(buy.name)
    return all_individual_pnls.sort_values("Date")

if not st.session_state.trades.empty:
    realized_df = calculate_individual_pnl(st.session_state.trades)
    # Compute cumulative realized profit by date
    if not realized_df.empty:
        realized_time = realized_df.groupby("Date")["PnL"].sum().sort_index().cumsum().reset_index()
    else:
        realized_time = pd.DataFrame(columns=["Date", "PnL"])
else:
    realized_time = pd.DataFrame(columns=["Date", "PnL"])

# ------------------ DIVIDENDS TIME SERIES ------------------
if not st.session_state.dividends.empty:
    div_df = st.session_state.dividends.copy()
    div_df["Date"] = pd.to_datetime(div_df["Date"])
    dividends_time = div_df.groupby("Date")["Dividend"].sum().sort_index().cumsum().reset_index()
else:
    dividends_time = pd.DataFrame(columns=["Date", "Dividend"])

# ------------------ PLOTS ------------------
st.header("Performance Plots")

# Plot cumulative realized profit over time
st.subheader("Cumulative Realized Profit Over Time")
if not realized_time.empty:
    # Ensure Date is index for plotting
    rt = realized_time.set_index("Date")
    st.line_chart(rt)
else:
    st.info("No sell trades available to compute realized profit.")

# Plot cumulative dividends over time
st.subheader("Cumulative Dividends Over Time")
if not dividends_time.empty:
    dt = dividends_time.set_index("Date")
    st.line_chart(dt)
else:
    st.info("No dividend data available.")

# Bar chart for unrealized profit by stock (from portfolio summary)
st.subheader("Unrealized Profit by Stock")
if not st.session_state.trades.empty:
    unrealized = summary.set_index("Stock")["Unrealized Profit"]
    st.bar_chart(unrealized)
else:
    st.info("No trades available to compute unrealized profit.")
