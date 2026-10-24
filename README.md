# SmartInvest AI — Smart Investment Risk and Portfolio Learning Assistant

## 1. Project Overview

SmartInvest AI is a Python-based web application that helps users understand stock market performance, investment risk, and portfolio management through interactive analysis, machine learning demonstrations, and visualizations.

The project is designed for educational purposes and demonstrates how AI and data analytics can be applied to financial problems.

## 2. Problem Statement

Many beginner investors find it difficult to understand stock performance, investment risk, diversification, and portfolio allocation. SmartInvest AI provides an interactive platform to analyze historical stock data and learn these concepts using understandable metrics and visualizations.

## 3. Objectives

- Analyze historical stock performance.
- Calculate historical returns, annualized volatility, and maximum drawdown.
- Compare the performance and risk of multiple stocks.
- Demonstrate K-Means clustering for stock and portfolio classification.
- Simulate and compare hypothetical investment portfolios.
- Provide an educational AI investment assistant.
- Assess investor risk profiles and demonstrate risk-based asset allocation.
- Generate downloadable CSV and PDF reports.

## 4. Features

1. Investment dashboard with allocation and diversification analysis.
2. Historical stock analysis using market data.
3. Comparison of multiple stocks.
4. K-Means clustering demonstration using sample portfolios.
5. Machine-learning-based classification of stocks using historical metrics.
6. Portfolio comparison simulator.
7. AI Investment Assistant.
8. Portfolio Metric Explainer.
9. Investor Risk Profiler.
10. Risk-based educational asset allocation.
11. CSV report export.
12. PDF report generation.
13. Sidebar navigation.
14. Dark theme.

## 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| Pandas | Data processing |
| NumPy | Numerical calculations |
| yfinance | Historical stock market data |
| Scikit-learn | K-Means clustering |
| Plotly | Interactive charts |
| ReportLab | PDF report generation |
| CSV | Downloadable report format |

## 6. Installation and Setup

### Prerequisites

- Python installed on your computer.
- Visual Studio Code or another code editor.
- Internet connection for retrieving stock market data.

### Install dependencies

Open the terminal inside the project folder and run:

```bash
python -m pip install streamlit pandas numpy scikit-learn plotly yfinance reportlab
```

### Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## 7. How the System Works

1. The user selects stocks or enters portfolio parameters.
2. Historical stock data is retrieved using yfinance.
3. Python processes the data and calculates financial metrics.
4. Scikit-learn performs K-Means clustering where applicable.
5. Streamlit displays analysis results and interactive charts.
6. The user can explore investment concepts and download reports.

## 8. Financial Metrics

- **Historical Return:** Percentage change between the first and last available adjusted closing prices during the selected period.
- **Annualized Volatility:** A measure of historical return variability, estimated from daily returns and annualized using 252 trading days.
- **Maximum Drawdown:** The largest percentage decline from a previous running price peak during the observed period.
- **Diversification Score:** An application-specific indicator used to illustrate how an investment allocation is distributed.

## 9. Machine Learning

The project demonstrates K-Means clustering, an unsupervised machine-learning algorithm. Historical return, annualized volatility, and maximum drawdown are used as features for stock-risk classification.

The resulting clusters illustrate similarities among the analyzed stocks. Cluster labels are exploratory and should not be interpreted as definitive predictions of future risk or returns.

## 10. Limitations

- Analysis depends on the availability and quality of historical market data.
- Historical performance does not guarantee future results.
- Machine-learning clusters depend on the selected data, features, and parameters.
- Educational allocation suggestions are simplified examples.
- The application is not a substitute for professional financial advice.

## 11. Future Enhancements

- Add additional financial datasets and asset classes.
- Improve model evaluation and explainability.
- Introduce portfolio optimization techniques.
- Add user accounts and saved analysis.
- Support additional downloadable report formats.
- Evaluate predictive models using appropriate historical testing.

## 12. Conclusion

SmartInvest AI demonstrates the practical use of Python, machine learning, financial data analysis, and interactive web development to make investment concepts easier to explore. It combines analytical tools and educational features in one application.

## Disclaimer

This project is for educational purposes only. It does not guarantee investment returns and should not be treated as personalized financial advice.
