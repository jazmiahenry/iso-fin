# ISO Financial MVP

A comprehensive financial portfolio analysis system with Monte Carlo simulation, scenario analysis, risk assessment, and interactive visualization.

## Features

- **Financial Educator**:
  - AI-powered financial assistant using GPT models
  - Answers financial questions with accurate information
  - Performs complex financial calculations
  - Access to historical data and market insights
  - Ability to explain complex financial concepts

- **Simulation Engines**:
  - Monte Carlo simulation for portfolio projections
  - Scenario analysis for testing portfolio resilience
  - Risk Parity optimization for portfolio allocation
  - Conditional Value at Risk (CVaR) for tail risk analysis
  - Bayesian scenario ranking for risk prioritization

- **Interactive Portfolio Analysis**:
  - Real-time portfolio simulation
  - Adjustable time horizons and risk profiles
  - Risk contribution analysis
  - Scenario comparison
  - Historical data integration

- **Report Generation**:
  - Free report tier with basic analysis
  - Premium report tier with comprehensive analysis
  - PDF generation with charts and visualizations
  - Standardized disclaimers and legal protection

- **Data Integration**:
  - Polygon.io API for reliable market data
  - Asset correlation analysis
  - Ticker symbol identification from natural language

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- Polygon.io API key (get one at https://polygon.io/)
- OpenAI API key (for the Financial Educator)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/iso-financial-mvp.git
   cd iso-financial-mvp
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root with:
   ```
   POLYGON=your_polygon_api_key
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_ORG_ID=your_openai_org_id  # If applicable
   ```

### Running the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

This launches the interactive Streamlit interface with:
- Portfolio simulation and analysis
- Financial Educator powered by OpenAI GPT
- Risk assessment and scenario analysis
- Report generation

## Project Structure

- `iso_financial_mvp/`: Main package
  - `data_sources/`: Market data integration modules
  - `gpt_agent/`: Financial Educator powered by GPT models
  - `llm_interface/`: Natural language processing modules
  - `report_generator/`: PDF report generation
  - `simulation_engines/`: Financial analysis algorithms
  - `streamlit/`: Streamlit page components and utilities
  - `toolkit/`: Financial calculation tools

## Financial Educator Features

The Financial Educator component is an AI-powered assistant that can:

- Answer financial questions with accurate information
- Perform financial calculations using specialized tools
- Access real-time and historical market data
- Run Monte Carlo simulations for portfolio analysis
- Analyze risk metrics (VaR, CVaR, Maximum Drawdown)
- Optimize portfolio weights for balanced risk
- Test portfolios against various economic scenarios

The system tries to use a finetuned GPT model first, then falls back to a standard GPT model if the finetuned model is unavailable.

## Disclaimer

This software provides financial analysis for informational purposes only. It is not financial advice. 
Always consult with a qualified financial advisor before making investment decisions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.