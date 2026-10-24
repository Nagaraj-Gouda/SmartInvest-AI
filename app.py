import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# --------------------------------------------------
# DASHBOARD UI IMPROVEMENTS
# --------------------------------------------------

st.set_page_config(
    page_title="SmartInvest AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("SmartInvest AI")
    st.caption("AI-powered investment learning")


    st.divider()
    st.subheader("Navigation")

    st.markdown(
        """
        - [Dashboard](#investment-dashboard)
        - [Stock Analysis](#historical-stock-analysis)
        - [Compare Stocks](#compare-historical-stock-performance)
        - [AI Portfolio Classification](#ai-portfolio-risk-classification)
        - [AI Stock Classification](#ai-stock-risk-classification)
        - [Portfolio Simulator](#portfolio-comparison-simulator)
        - [AI Assistant](#ai-investment-assistant)
        - [Risk Profiler](#investor-risk-profiler)
        - [Asset Allocation](#risk-based-portfolio-allocation)
        - [Download Reports](#downloadable-portfolio-report)
        """
    )


    st.divider()

    st.subheader("Project Information")
    st.write("**Project:** SmartInvest AI")
    st.write("**Domain:** FinTech and Machine Learning")
    st.write("**Framework:** Streamlit")
    st.write("**Data Source:** Yahoo Finance")

    st.divider()

    st.info(
        "This application is designed for educational "
        "investment analysis. It does not guarantee returns."
    )


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SmartInvest AI",
    page_icon="📊",
    layout="wide"
)

st.title("SmartInvest AI")
st.subheader("Smart Investment Risk and Portfolio Learning Assistant")

st.write(
    "Explore portfolio allocation, historical returns, volatility, "
    "maximum drawdown, and AI-based stock classification."
)

st.info(
    "Educational project only. Historical performance does not "
    "guarantee future results. This application does not provide "
    "personalized financial advice."
)

# --------------------------------------------------
# 1. INVESTMENT LEARNING DASHBOARD
# --------------------------------------------------

st.header("Your Investment Learning Dashboard")

investment = st.number_input(
    "Total investment amount (₹)",
    min_value=1000,
    max_value=10000000,
    value=10000,
    step=1000
)

st.subheader("Sample Portfolio Allocation")

col1, col2, col3 = st.columns(3)

with col1:
    stock1 = st.slider("Stock A allocation (%)", 0, 100, 40)

with col2:
    stock2 = st.slider("Stock B allocation (%)", 0, 100, 35)

with col3:
    stock3 = st.slider("Stock C allocation (%)", 0, 100, 25)

total = stock1 + stock2 + stock3

col1, col2, col3 = st.columns(3)
col1.metric("Total Investment", f"₹{investment:,.0f}")
col2.metric("Total Allocation", f"{total}%")
col3.metric("Remaining Allocation", f"{100 - total}%")

if total == 100:
    weights = np.array([stock1, stock2, stock3], dtype=float) / 100
    concentration = float(np.sum(weights ** 2))
    diversification = (1 - concentration) * 100

    if concentration <= 0.36:
        concentration_status = "Lower concentration"
    elif concentration <= 0.50:
        concentration_status = "Moderate concentration"
    else:
        concentration_status = "Higher concentration"

    st.subheader("Portfolio Analysis")

    col1, col2 = st.columns(2)
    col1.metric("Diversification Score", f"{diversification:.1f}/100")
    col2.metric("Concentration Status", concentration_status)

    st.subheader("Investment Distribution")

    allocations = {
        "Stock A": stock1,
        "Stock B": stock2,
        "Stock C": stock3
    }

    for name, allocation in allocations.items():
        st.write(f"**{name}: {allocation}%**")
        st.progress(allocation / 100)
        st.write(
            f"Investment amount: ₹{investment * allocation / 100:,.2f}"
        )

else:
    st.warning(
        "Adjust the three allocations until their total equals 100%."
    )

st.caption(
    "The diversification score measures how evenly money is allocated "
    "across three sample holdings. It does not measure actual market risk "
    "or whether the holdings behave differently."
)

# --------------------------------------------------
# HELPER FUNCTION FOR HISTORICAL DATA
# --------------------------------------------------

def download_close_prices(symbol, period):
    """Download adjusted historical closing prices for one symbol."""
    data = yf.download(
        symbol,
        period=period,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if data.empty or "Close" not in data.columns:
        return pd.Series(dtype=float)

    close = data["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    return close.dropna()


def calculate_stock_metrics(close):
    """Calculate historical return, volatility and maximum drawdown."""
    close = close.dropna()
    daily_returns = close.pct_change().dropna()

    if len(close) < 2 or daily_returns.empty:
        return None

    historical_return = (
        float(close.iloc[-1]) / float(close.iloc[0]) - 1
    ) * 100

    annualized_volatility = (
        float(daily_returns.std()) * np.sqrt(252) * 100
    )

    running_peak = close.cummax()
    drawdowns = (close / running_peak - 1) * 100
    maximum_drawdown = float(drawdowns.min())

    metrics = [
        historical_return,
        annualized_volatility,
        maximum_drawdown
    ]

    if not np.all(np.isfinite(metrics)):
        return None

    return {
        "Historical Return (%)": historical_return,
        "Annualized Volatility (%)": annualized_volatility,
        "Maximum Drawdown (%)": maximum_drawdown
    }


# --------------------------------------------------
# 2. HISTORICAL STOCK ANALYSIS
# --------------------------------------------------

st.divider()
st.header("Historical Stock Analysis")

st.write(
    "Analyze one stock using historical market data. "
    "NSE symbols commonly end in .NS."
)

symbol = st.text_input(
    "Stock symbol",
    value="RELIANCE.NS",
    key="single_stock_symbol"
).strip().upper()

period = st.selectbox(
    "Historical period",
    ["6mo", "1y", "2y", "5y"],
    index=1,
    key="single_stock_period"
)

if st.button("Analyze Stock", key="analyze_single_stock"):

    if not symbol:
        st.error("Enter a stock symbol.")

    else:
        try:
            with st.spinner("Downloading historical prices..."):
                close = download_close_prices(symbol, period)

            metrics = calculate_stock_metrics(close)

            if metrics is None:
                st.error(
                    "Insufficient historical data. Check the symbol "
                    "or select a longer period."
                )

            else:
                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Latest Adjusted Close",
                    f"₹{float(close.iloc[-1]):,.2f}"
                )

                col2.metric(
                    "Historical Return",
                    f"{metrics['Historical Return (%)']:.2f}%"
                )

                col3.metric(
                    "Annualized Volatility",
                    f"{metrics['Annualized Volatility (%)']:.2f}%"
                )

                st.metric(
                    "Maximum Drawdown",
                    f"{metrics['Maximum Drawdown (%)']:.2f}%"
                )

                st.subheader("Historical Adjusted Closing Price")
                st.line_chart(close)

                st.caption(
                    "Maximum drawdown measures the largest historical "
                    "peak-to-trough decline in the selected price series."
                )

        except Exception as error:
            st.error("Unable to analyze this stock.")
            st.caption(f"Technical details: {error}")

# --------------------------------------------------
# 3. COMPARE HISTORICAL STOCK PERFORMANCE
# --------------------------------------------------

st.divider()
st.header("Compare Historical Stock Performance")

st.write(
    "Compare historical returns and annualized volatility "
    "for three stocks."
)

comparison_period = st.selectbox(
    "Comparison period",
    ["6mo", "1y", "2y", "5y"],
    index=1,
    key="comparison_period"
)

col1, col2, col3 = st.columns(3)

with col1:
    compare1 = st.text_input(
        "Stock 1",
        value="RELIANCE.NS",
        key="compare_stock1"
    ).strip().upper()

with col2:
    compare2 = st.text_input(
        "Stock 2",
        value="TCS.NS",
        key="compare_stock2"
    ).strip().upper()

with col3:
    compare3 = st.text_input(
        "Stock 3",
        value="INFY.NS",
        key="compare_stock3"
    ).strip().upper()

if st.button("Compare Stocks", key="compare_stocks_button"):

    comparison_results = []
    price_series = {}

    comparison_symbols = list(
        dict.fromkeys([compare1, compare2, compare3])
    )

    for stock in comparison_symbols:
        if not stock:
            continue

        try:
            close = download_close_prices(stock, comparison_period)
            metrics = calculate_stock_metrics(close)

            if metrics is None:
                st.warning(f"Insufficient data for {stock}.")
                continue

            comparison_results.append({
                "Stock": stock,
                "Historical Return (%)": round(
                    metrics["Historical Return (%)"], 2
                ),
                "Annualized Volatility (%)": round(
                    metrics["Annualized Volatility (%)"], 2
                )
            })

            price_series[stock] = close / close.iloc[0] * 100

        except Exception:
            st.warning(f"Could not retrieve data for {stock}.")

    if comparison_results:
        st.subheader("Comparison Results")
        st.dataframe(
            pd.DataFrame(comparison_results),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Normalized Historical Performance")
        st.line_chart(pd.DataFrame(price_series))

    else:
        st.error("No valid stock data could be retrieved.")

# --------------------------------------------------
# 4. AI PORTFOLIO RISK CLASSIFICATION - SAMPLE DATA
# --------------------------------------------------

st.divider()
st.header("AI Portfolio Risk Classification")

st.write(
    "Demonstration of K-Means clustering using synthetic sample portfolios."
)

if st.button("Run AI Risk Classification", key="sample_ai_button"):

    sample_portfolios = pd.DataFrame({
        "Annualized Volatility (%)": [
            10, 12, 14, 16, 18,
            22, 25, 28, 30, 32,
            35, 38, 40, 43, 46
        ],
        "Historical Return (%)": [
            5, 7, 6, 9, 8,
            10, 12, 8, 14, 11,
            15, 13, 18, 16, 20
        ],
        "Concentration Index": [
            0.25, 0.28, 0.30, 0.32, 0.35,
            0.38, 0.40, 0.42, 0.45, 0.48,
            0.50, 0.55, 0.58, 0.62, 0.65
        ]
    })

    features = sample_portfolios[
        [
            "Annualized Volatility (%)",
            "Historical Return (%)",
            "Concentration Index"
        ]
    ]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    sample_portfolios["Cluster"] = model.fit_predict(scaled_features)

    cluster_volatility = sample_portfolios.groupby("Cluster")[
        "Annualized Volatility (%)"
    ].mean()

    ordered_clusters = cluster_volatility.sort_values().index.tolist()

    cluster_names = {
        ordered_clusters[0]: "Lower historical volatility",
        ordered_clusters[1]: "Moderate historical volatility",
        ordered_clusters[2]: "Higher historical volatility"
    }

    sample_portfolios["AI Classification"] = (
        sample_portfolios["Cluster"].map(cluster_names)
    )

    summary = (
        sample_portfolios.groupby("AI Classification")[
            [
                "Annualized Volatility (%)",
                "Historical Return (%)",
                "Concentration Index"
            ]
        ]
        .mean()
        .round(2)
        .reset_index()
    )

    st.success("Sample AI clustering completed.")

    st.subheader("AI Cluster Summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Classified Sample Portfolios")
    st.dataframe(
        sample_portfolios.drop(columns=["Cluster"]),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "These are synthetic examples, not actual investor portfolios."
    )

# --------------------------------------------------
# 5. AI CLASSIFICATION OF INDIVIDUAL STOCKS
# --------------------------------------------------

st.divider()
st.header("AI Stock Risk Classification")

st.write(
    "K-Means groups stocks using historical return, annualized volatility, "
    "and maximum drawdown calculated from real historical prices."
)

ai_period = st.selectbox(
    "AI analysis period",
    ["6mo", "1y", "2y", "5y"],
    index=1,
    key="ai_stock_period"
)

ai_symbols_text = st.text_input(
    "Stock symbols (comma-separated)",
    value="RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS",
    key="ai_stock_symbols"
)

if st.button(
    "Run AI Stock Classification",
    key="run_ai_stock_classification"
):

    symbols = list(dict.fromkeys(
        item.strip().upper()
        for item in ai_symbols_text.split(",")
        if item.strip()
    ))

    stock_metrics = []
    failed_symbols = []

    if len(symbols) < 3:
        st.error("Enter at least three different stock symbols.")

    else:
        with st.spinner("Downloading historical prices..."):

            for stock in symbols:
                try:
                    close = download_close_prices(stock, ai_period)

                    if len(close) < 30:
                        failed_symbols.append(stock)
                        continue

                    metrics = calculate_stock_metrics(close)

                    if metrics is None:
                        failed_symbols.append(stock)
                        continue

                    stock_metrics.append({
                        "Stock": stock,
                        **metrics
                    })

                except Exception:
                    failed_symbols.append(stock)

        if failed_symbols:
            st.warning(
                "Could not analyze: " + ", ".join(failed_symbols)
            )

        if len(stock_metrics) < 3:
            st.error(
                "At least three stocks with sufficient valid data are "
                "required. Check the symbols, internet connection, "
                "or selected period."
            )

        else:
            results = pd.DataFrame(stock_metrics)

            feature_columns = [
                "Historical Return (%)",
                "Annualized Volatility (%)",
                "Maximum Drawdown (%)"
            ]

            features = results[feature_columns]

            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            cluster_count = min(3, len(results))

            model = KMeans(
                n_clusters=cluster_count,
                random_state=42,
                n_init=10
            )

            results["Cluster"] = model.fit_predict(scaled_features)

            average_volatility = results.groupby("Cluster")[
                "Annualized Volatility (%)"
            ].mean()

            ordered_clusters = average_volatility.sort_values().index.tolist()

            if cluster_count == 3:
                cluster_labels = {
                    ordered_clusters[0]: "Lower historical volatility",
                    ordered_clusters[1]: "Moderate historical volatility",
                    ordered_clusters[2]: "Higher historical volatility"
                }
            else:
                cluster_labels = {
                    cluster: f"Relative volatility group {index + 1}"
                    for index, cluster in enumerate(ordered_clusters)
                }

            results["AI Classification"] = results["Cluster"].map(
                cluster_labels
            )

            results = results.drop(columns=["Cluster"])

            results[feature_columns] = results[feature_columns].round(2)

            st.success(
                f"AI classification completed for {len(results)} stocks."
            )

            st.subheader("AI Classification Results")
            st.dataframe(
                results.sort_values("Annualized Volatility (%)"),
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Historical Volatility by Stock")
            volatility_chart = results.set_index("Stock")[
                "Annualized Volatility (%)"
            ]
            st.bar_chart(volatility_chart)

            st.subheader("Maximum Drawdown by Stock")
            drawdown_chart = results.set_index("Stock")[
                "Maximum Drawdown (%)"
            ]
            st.bar_chart(drawdown_chart)

            # ------------------------------------------
            # 6. AI STOCK RISK EXPLANATIONS
            # ------------------------------------------

            st.subheader("AI-Generated Risk Explanation")

            for _, row in results.iterrows():
                stock = row["Stock"]
                ret = row["Historical Return (%)"]
                volatility = row["Annualized Volatility (%)"]
                drawdown = row["Maximum Drawdown (%)"]
                classification = row["AI Classification"]

                if volatility < 20:
                    volatility_text = "relatively lower"
                elif volatility < 35:
                    volatility_text = "moderate"
                else:
                    volatility_text = "relatively higher"

                if drawdown <= -30:
                    drawdown_text = "a large historical peak-to-trough decline"
                elif drawdown <= -15:
                    drawdown_text = (
                        "a noticeable historical peak-to-trough decline"
                    )
                else:
                    drawdown_text = (
                        "a smaller historical peak-to-trough decline"
                    )

                with st.expander(f"{stock} — {classification}"):

                    st.write(
                        f"**Historical return:** {ret:.2f}% "
                        "over the selected period."
                    )

                    st.write(
                        f"**Annualized volatility:** {volatility:.2f}%, "
                        f"indicating {volatility_text} price variability "
                        "under this project's illustrative thresholds."
                    )

                    st.write(
                        f"**Maximum drawdown:** {drawdown:.2f}%, "
                        f"indicating {drawdown_text} during the selected period."
                    )

                    st.write(
                        f"**AI classification:** {classification}. "
                        "This is a relative grouping of the selected stocks, "
                        "not a prediction of future prices or a recommendation."
                    )

            st.caption(
                "The explanations use predefined rules applied to historical "
                "metrics. K-Means performs the grouping; a generative AI model "
                "is not used to create the explanations."
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "SmartInvest AI is an educational project, not financial advice. "
    "Market data availability depends on the external data provider."
)


# --------------------------------------------------
# 7. PORTFOLIO COMPARISON SIMULATOR
# --------------------------------------------------

st.divider()
st.header("Portfolio Comparison Simulator")

st.write(
    "Compare two hypothetical portfolios using historical stock prices. "
    "Results are historical simulations, not forecasts."
)

portfolio_period = st.selectbox(
    "Portfolio simulation period",
    ["6mo", "1y", "2y", "5y"],
    index=1,
    key="portfolio_sim_period"
)

st.subheader("Portfolio A")

a_stock1 = st.text_input(
    "Portfolio A - Stock 1",
    "RELIANCE.NS",
    key="a_stock1"
).strip().upper()

a_stock2 = st.text_input(
    "Portfolio A - Stock 2",
    "TCS.NS",
    key="a_stock2"
).strip().upper()

a_weight1 = st.slider(
    "Portfolio A - Stock 1 allocation (%)",
    0, 100, 60,
    key="a_weight1"
)

a_weight2 = 100 - a_weight1

st.write(f"Portfolio A - Stock 2 allocation: **{a_weight2}%**")

st.subheader("Portfolio B")

b_stock1 = st.text_input(
    "Portfolio B - Stock 1",
    "INFY.NS",
    key="b_stock1"
).strip().upper()

b_stock2 = st.text_input(
    "Portfolio B - Stock 2",
    "HDFCBANK.NS",
    key="b_stock2"
).strip().upper()

b_weight1 = st.slider(
    "Portfolio B - Stock 1 allocation (%)",
    0, 100, 50,
    key="b_weight1"
)

b_weight2 = 100 - b_weight1

st.write(f"Portfolio B - Stock 2 allocation: **{b_weight2}%**")

if st.button("Compare Portfolios", key="compare_portfolios_button"):

    try:
        symbols = list(dict.fromkeys([
            a_stock1, a_stock2, b_stock1, b_stock2
        ]))

        if any(not symbol for symbol in symbols):
            st.error("Enter a stock symbol in every field.")

        else:
            prices = {}

            with st.spinner("Downloading historical prices..."):
                for symbol in symbols:
                    close = download_close_prices(
                        symbol, portfolio_period
                    )

                    if close.empty:
                        st.error(
                            f"No historical data found for {symbol}. "
                            "Check the symbol and try again."
                        )
                        break

                    prices[symbol] = close

            if len(prices) == len(symbols):

                price_df = pd.concat(
                    prices,
                    axis=1,
                    join="inner"
                ).dropna()

                if len(price_df) < 30:
                    st.error(
                        "Not enough overlapping historical data. "
                        "Try a longer period or different stocks."
                    )

                else:
                    daily_returns = price_df.pct_change().dropna()

                    weights_a = pd.Series(
                        {
                            a_stock1: a_weight1 / 100,
                            a_stock2: a_weight2 / 100
                        }
                    ).groupby(level=0).sum()

                    weights_b = pd.Series(
                        {
                            b_stock1: b_weight1 / 100,
                            b_stock2: b_weight2 / 100
                        }
                    ).groupby(level=0).sum()

                    # Align weights to downloaded stock columns
                    weights_a = weights_a.reindex(
                        daily_returns.columns, fill_value=0
                    )
                    weights_b = weights_b.reindex(
                        daily_returns.columns, fill_value=0
                    )

                    # Daily portfolio returns; weights are held constant
                    returns_a = daily_returns.dot(weights_a)
                    returns_b = daily_returns.dot(weights_b)

                    growth_a = (1 + returns_a).cumprod()
                    growth_b = (1 + returns_b).cumprod()

                    # Normalize both portfolios to a starting value of 100
                    growth_a = growth_a * 100
                    growth_b = growth_b * 100

                    def portfolio_metrics(returns, growth):
                        total_return = (float(growth.iloc[-1]) / 100 - 1) * 100

                        volatility = (
                            float(returns.std()) * np.sqrt(252) * 100
                        )

                        drawdown = (
                            growth / growth.cummax() - 1
                        ) * 100

                        maximum_drawdown = float(drawdown.min())

                        return {
                            "Historical Return (%)": round(total_return, 2),
                            "Annualized Volatility (%)": round(volatility, 2),
                            "Maximum Drawdown (%)": round(
                                maximum_drawdown, 2
                            )
                        }

                    metrics_a = portfolio_metrics(returns_a, growth_a)
                    metrics_b = portfolio_metrics(returns_b, growth_b)

                    st.subheader("Portfolio Results")

                    results_df = pd.DataFrame(
                        [metrics_a, metrics_b],
                        index=["Portfolio A", "Portfolio B"]
                    )

                    st.dataframe(
                        results_df,
                        use_container_width=True
                    )

                    st.subheader("Historical Portfolio Growth")

                    growth_df = pd.DataFrame({
                        "Portfolio A": growth_a,
                        "Portfolio B": growth_b
                    })

                    st.line_chart(growth_df)

                    st.caption(
                        "Both portfolios start at an index value of 100. "
                        "The simulation uses overlapping adjusted closing "
                        "prices and fixed target weights applied to daily "
                        "returns. It does not include transaction costs, "
                        "taxes, or rebalancing costs. Historical results "
                        "do not predict future performance."
                    )

    except Exception as error:
        st.error("Unable to compare the portfolios.")
        st.caption(f"Technical details: {error}")


# --------------------------------------------------
# 8. AI INVESTMENT ASSISTANT
# --------------------------------------------------

st.divider()
st.header("AI Investment Assistant")

st.write(
    "Get beginner-friendly explanations of historical "
    "investment performance and risk."
)

st.warning(
    "Educational tool only. This assistant does not predict "
    "future returns or provide personalized financial advice."
)

assistant_topic = st.selectbox(
    "What would you like explained?",
    [
        "Understanding investment risk",
        "Understanding historical returns",
        "Understanding maximum drawdown",
        "Understanding diversification"
    ],
    key="assistant_topic"
)

if st.button("Explain", key="assistant_explain_button"):

    st.subheader("Your Explanation")

    if assistant_topic == "Understanding investment risk":
        st.write(
            "**Investment risk** means the uncertainty of "
            "investment results. Annualized volatility measures "
            "how much daily returns fluctuate over time. "
            "Higher volatility means returns have varied more; "
            "it does not guarantee future losses."
        )

    elif assistant_topic == "Understanding historical returns":
        st.write(
            "**Historical return** measures how an investment "
            "performed over a past period. A positive return "
            "means its value increased over that period, while "
            "a negative return means it decreased. Past results "
            "do not guarantee future performance."
        )

    elif assistant_topic == "Understanding maximum drawdown":
        st.write(
            "**Maximum drawdown** measures the largest "
            "percentage decline from a previous peak to a "
            "subsequent low during the selected period. "
            "A larger negative drawdown indicates a deeper "
            "historical decline."
        )

    elif assistant_topic == "Understanding diversification":
        st.write(
            "**Diversification** means spreading investments "
            "across different assets rather than relying on "
            "one stock. It can help reduce concentration risk, "
            "but it cannot eliminate investment risk."
        )

    st.caption(
        "This explanation uses predefined Python rules and "
        "financial concepts; it does not use a generative AI model."
    )


# --------------------------------------------------
# 9. PERSONALIZED PORTFOLIO METRIC EXPLAINER
# --------------------------------------------------

st.divider()
st.subheader("Portfolio Metric Explainer")

st.write(
    "Enter your portfolio metrics to receive a simple "
    "educational interpretation."
)

col1, col2 = st.columns(2)

with col1:
    user_return = st.number_input(
        "Historical Return (%)",
        min_value=-100.0,
        max_value=1000.0,
        value=-15.0,
        step=1.0,
        key="explainer_return"
    )

    user_volatility = st.number_input(
        "Annualized Volatility (%)",
        min_value=0.0,
        max_value=500.0,
        value=18.87,
        step=1.0,
        key="explainer_volatility"
    )

with col2:
    user_drawdown = st.number_input(
        "Maximum Drawdown (%)",
        min_value=-100.0,
        max_value=0.0,
        value=-28.83,
        step=1.0,
        key="explainer_drawdown"
    )

if st.button("Analyze My Metrics", key="analyze_metrics_button"):

    st.subheader("Portfolio Analysis")

    # Historical return explanation
    if user_return > 0:
        st.success(
            f"The portfolio gained {user_return:.2f}% "
            "during the selected historical period."
        )
    elif user_return < 0:
        st.warning(
            f"The portfolio lost {abs(user_return):.2f}% "
            "during the selected historical period."
        )
    else:
        st.info(
            "The portfolio had approximately zero historical return."
        )

    # Volatility explanation
    if user_volatility < 10:
        volatility_message = "Relatively low historical volatility."
    elif user_volatility < 25:
        volatility_message = "Moderate historical volatility."
    else:
        volatility_message = "High historical volatility."

    st.write(
        f"**Volatility:** {volatility_message} "
        f"Measured volatility: {user_volatility:.2f}% annually."
    )

    # Maximum drawdown explanation
    st.write(
        f"**Maximum drawdown:** {user_drawdown:.2f}%. "
        "This represents the largest historical decline from "
        "a previous peak to a subsequent low."
    )

    if user_drawdown <= -30:
        st.warning(
            "The portfolio experienced a substantial historical "
            "decline. Consider how comfortable you would be with "
            "similar losses when learning about investment risk."
        )
    else:
        st.info(
            "Review the drawdown alongside returns and volatility "
            "to understand the portfolio's historical risk."
        )

    st.caption(
        "These are educational interpretations, not predictions "
        "or personalized investment recommendations. Volatility "
        "categories are simplified guidelines."
    )


# --------------------------------------------------
# 10. INVESTOR RISK PROFILER
# --------------------------------------------------

st.divider()
st.header("Investor Risk Profiler")

st.write(
    "Answer these questions to estimate your risk tolerance "
    "for educational purposes."
)

investment_goal = st.selectbox(
    "1. What is your main investment goal?",
    [
        "Protect my money",
        "Balance growth and safety",
        "Achieve long-term growth"
    ],
    key="risk_goal"
)

investment_period = st.selectbox(
    "2. How long do you plan to invest?",
    [
        "Less than 1 year",
        "1–5 years",
        "More than 5 years"
    ],
    key="risk_period"
)

loss_reaction = st.selectbox(
    "3. How would you react if your investment fell by 20%?",
    [
        "I would feel uncomfortable and sell",
        "I would wait and review",
        "I could tolerate the decline"
    ],
    key="risk_reaction"
)

risk_knowledge = st.selectbox(
    "4. How familiar are you with investment risks?",
    [
        "Beginner",
        "Some knowledge",
        "Experienced"
    ],
    key="risk_knowledge"
)

if st.button("Calculate Risk Profile", key="calculate_risk_button"):

    score = 0

    # Investment goal
    if investment_goal == "Balance growth and safety":
        score += 1
    elif investment_goal == "Achieve long-term growth":
        score += 2

    # Investment period
    if investment_period == "1–5 years":
        score += 1
    elif investment_period == "More than 5 years":
        score += 2

    # Reaction to losses
    if loss_reaction == "I would wait and review":
        score += 1
    elif loss_reaction == "I could tolerate the decline":
        score += 2

    # Familiarity with risk
    if risk_knowledge == "Some knowledge":
        score += 1
    elif risk_knowledge == "Experienced":
        score += 2

    st.subheader("Your Estimated Risk Profile")

    if score <= 2:
        st.info("Low Risk Tolerance")
        st.write(
            "You appear to prefer stability and may be less "
            "comfortable with investment losses. Learn about "
            "capital preservation and lower-risk options."
        )

    elif score <= 5:
        st.info("Moderate Risk Tolerance")
        st.write(
            "Your answers suggest a balance between stability "
            "and growth. Learn how diversification and asset "
            "allocation can affect portfolio risk."
        )

    else:
        st.info("High Risk Tolerance")
        st.write(
            "Your answers suggest greater willingness to accept "
            "investment fluctuations for potential long-term "
            "growth. Higher risk does not guarantee higher returns."
        )

    st.caption(
        f"Educational score: {score}/8. This simplified quiz "
        "is not a validated financial assessment. Risk tolerance "
        "does not measure your financial ability to absorb losses."
    )


# --------------------------------------------------
# 11. RISK-BASED EDUCATIONAL ALLOCATION
# --------------------------------------------------

st.divider()
st.header("Risk-Based Portfolio Allocation")

st.write(
    "Explore an example allocation based on your estimated "
    "risk profile. These are illustrative percentages, not "
    "personalized investment recommendations."
)

allocation_profile = st.selectbox(
    "Choose a risk profile",
    ["Low Risk Tolerance", "Moderate Risk Tolerance",
     "High Risk Tolerance"],
    key="allocation_profile"
)

allocation_options = {
    "Low Risk Tolerance": {
        "Stocks": 20,
        "Bonds": 30,
        "Cash": 50
    },
    "Moderate Risk Tolerance": {
        "Stocks": 50,
        "Bonds": 30,
        "Cash": 20
    },
    "High Risk Tolerance": {
        "Stocks": 75,
        "Bonds": 15,
        "Cash": 10
    }
}

allocation = allocation_options[allocation_profile]

allocation_df = pd.DataFrame({
    "Asset Class": list(allocation.keys()),
    "Allocation (%)": list(allocation.values())
})

st.subheader(f"Example for {allocation_profile}")

st.dataframe(allocation_df, use_container_width=True)

st.bar_chart(
    allocation_df.set_index("Asset Class")["Allocation (%)"]
)

st.write(
    f"Total allocation: **{sum(allocation.values())}%**"
)

st.caption(
    "Stocks can fluctuate significantly. Bonds also carry risks, "
    "and cash may lose purchasing power due to inflation. Actual "
    "asset suitability depends on financial circumstances, goals, "
    "time horizon, and ability to absorb losses."
)


# --------------------------------------------------
# 12. DOWNLOADABLE PORTFOLIO REPORT
# --------------------------------------------------

st.divider()
st.header("Downloadable Portfolio Report")

st.write(
    "Create a downloadable report containing an example "
    "asset allocation and its educational risk profile."
)

report_profile = st.selectbox(
    "Select the risk profile for your report",
    [
        "Low Risk Tolerance",
        "Moderate Risk Tolerance",
        "High Risk Tolerance"
    ],
    key="report_profile"
)

report_allocations = {
    "Low Risk Tolerance": {
        "Stocks": 20,
        "Bonds": 30,
        "Cash": 50
    },
    "Moderate Risk Tolerance": {
        "Stocks": 50,
        "Bonds": 30,
        "Cash": 20
    },
    "High Risk Tolerance": {
        "Stocks": 75,
        "Bonds": 15,
        "Cash": 10
    }
}

selected_allocation = report_allocations[report_profile]

report_df = pd.DataFrame({
    "Risk Profile": [report_profile] * len(selected_allocation),
    "Asset Class": list(selected_allocation.keys()),
    "Allocation (%)": list(selected_allocation.values())
})

csv_data = report_df.to_csv(index=False).encode("utf-8")

st.dataframe(report_df, use_container_width=True)

st.download_button(
    label="Download Portfolio Report (CSV)",
    data=csv_data,
    file_name="smartinvest_portfolio_report.csv",
    mime="text/csv",
    key="download_portfolio_report"
)

st.caption(
    "This report contains illustrative allocations only. "
    "It does not contain verified personal financial data "
    "or personalized investment advice."
)


# --------------------------------------------------
# 13. PDF PORTFOLIO REPORT
# --------------------------------------------------

from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

st.divider()
st.subheader("PDF Portfolio Report")

pdf_profile = st.selectbox(
    "Choose a risk profile for the PDF",
    [
        "Low Risk Tolerance",
        "Moderate Risk Tolerance",
        "High Risk Tolerance"
    ],
    key="pdf_report_profile"
)

pdf_allocations = {
    "Low Risk Tolerance": {
        "Stocks": 20,
        "Bonds": 30,
        "Cash": 50
    },
    "Moderate Risk Tolerance": {
        "Stocks": 50,
        "Bonds": 30,
        "Cash": 20
    },
    "High Risk Tolerance": {
        "Stocks": 75,
        "Bonds": 15,
        "Cash": 10
    }
}

if st.button("Generate PDF Report", key="generate_pdf_button"):

    try:
        buffer = BytesIO()

        document = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        styles = getSampleStyleSheet()
        story = []

        story.append(
            Paragraph("SmartInvest AI", styles["Title"])
        )
        story.append(
            Paragraph(
                "Educational Portfolio Allocation Report",
                styles["Heading2"]
            )
        )
        story.append(Spacer(1, 16))

        story.append(
            Paragraph(
                f"Selected Risk Profile: {pdf_profile}",
                styles["Normal"]
            )
        )
        story.append(Spacer(1, 12))

        table_data = [["Asset Class", "Allocation (%)"]]

        for asset, percentage in pdf_allocations[pdf_profile].items():
            table_data.append([asset, f"{percentage}%"])

        table = Table(
            table_data,
            colWidths=[220, 150]
        )

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))

        story.append(table)
        story.append(Spacer(1, 20))

        story.append(
            Paragraph(
                "Important Disclaimer",
                styles["Heading3"]
            )
        )
        story.append(
            Paragraph(
                "This report contains illustrative asset allocations "
                "for educational purposes only. It is not personalized "
                "financial advice, a prediction of returns, or a "
                "guarantee of investment performance. Stocks, bonds, "
                "and cash each have different risks.",
                styles["BodyText"]
            )
        )

        document.build(story)
        buffer.seek(0)

        st.download_button(
            label="Download PDF Report",
            data=buffer.getvalue(),
            file_name="SmartInvest_AI_Portfolio_Report.pdf",
            mime="application/pdf",
            key="download_pdf_report"
        )

        st.success("PDF report generated successfully.")

    except Exception as error:
        st.error("Unable to generate the PDF report.")
        st.caption(f"Technical details: {error}")
