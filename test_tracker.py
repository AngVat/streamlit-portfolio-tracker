import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

st.title("Investment Portfolio Tracker")

# ------------------ SESSION STATE INITIALIZATION ------------------
# Initialize an empty DataFrame to store trade logs
if 'trades' not in st.session_state:
    st.session_state.trades = pd.DataFrame(columns=["Date", "Stock", "Action", "Quantity", "Price"])

# ------------------ USER INPUT: ADD A TRADE ------------------
st.header("Add a Trade")

with st.form("trade_form", clear_on_submit=True):
    trade_date = st.date_input("Date", datetime.today())
    trade_stock = st.text_input("Stock Ticker", "AAPL")
    trade_action = st.selectbox("Action", ["Buy", "Sell"])
    trade_quantity = st.number_input("Quantity", min_value=1, step=1, value=1)
    trade_price = st.number_input("Price per Share", min_value=0.0, value=0.0, format="%.2f")
    submitted = st.form_submit_button("Add Trade")
    if submitted:
        new_trade = pd.DataFrame({
            "Date": [trade_date],
            "Stock": [trade_stock.upper()],
            "Action": [trade_action],
            "Quantity": [trade_quantity],
            "Price": [trade_price]
        })
        st.session_state.trades = pd.concat([st.session_state.trades, new_trade], ignore_index=True)
        st.success("Trade added!")

# ------------------ DISPLAY TRADE LOG ------------------
st.header("Trade Log")
st.dataframe(st.session_state.trades)

# ------------------ PORTFOLIO SUMMARY CALCULATION ------------------
if not st.session_state.trades.empty:
    trades = st.session_state.trades.copy()
    trades["Date"] = pd.to_datetime(trades["Date"])  # Ensure dates are datetime

    # Group trades by stock and compute net quantity and net cost:
    # For each row: if 'Buy', add quantity and cost; if 'Sell', subtract quantity and cost.
    summary = trades.groupby("Stock").apply(lambda df: pd.Series({
        "Net Quantity": df.apply(lambda row: row["Quantity"] if row["Action"]=="Buy" else -row["Quantity"], axis=1).sum(),
        "Net Cost": df.apply(lambda row: row["Quantity"] * row["Price"] if row["Action"]=="Buy" else -row["Quantity"] * row["Price"], axis=1).sum()
    })).reset_index()

    # Fetch current prices and compute current value for each stock.
    current_prices = {}
    current_values = []
    for stock in summary["Stock"]:
        try:
            ticker = yf.Ticker(stock)
            current_price = ticker.history(period="1d")["Close"].iloc[-1]
        except Exception:
            current_price = None
        current_prices[stock] = current_price
        net_qty = summary.loc[summary["Stock"]==stock, "Net Quantity"].iloc[0]
        if current_price is not None:
            current_values.append(net_qty * current_price)
        else:
            current_values.append(None)

    summary["Current Price"] = summary["Stock"].map(current_prices)
    summary["Current Value"] = current_values

    st.header("Portfolio Summary")
    st.dataframe(summary)

    # Calculate total portfolio value
    total_portfolio_value = summary["Current Value"].sum()
    st.write(f"**Total Portfolio Value:** {total_portfolio_value:.2f}")

    # Optionally, display profit/loss for each stock
    summary["Profit/Loss"] = summary["Current Value"] - summary["Net Cost"]
    st.write("### Profit / Loss by Stock")
    st.dataframe(summary[["Stock", "Net Quantity", "Net Cost", "Current Price", "Current Value", "Profit/Loss"]])
else:
    st.info("No trades added yet. Please add some trades to see your portfolio summary.")
