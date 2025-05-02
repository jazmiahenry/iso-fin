# ISO Financial MVP

A comprehensive portfolio analysis and simulation tool for crypto, NFTs, and alternative assets.

## Overview

ISO Financial is a financial analysis platform designed to help investors understand and manage their portfolios across traditional and alternative asset classes, with special focus on:

- Cryptocurrency (tokens, DeFi, staking)
- NFTs
- Collectibles
- Real-world alternative assets

The system uses natural language queries, Monte Carlo simulations, and scenario analysis to provide insights into portfolio risk, returns, and potential outcomes.

## Key Features

- **Natural Language Interface**: Ask questions like "What if ETH drops 40%?" or "How risky is my NFT-heavy portfolio?"
- **Monte Carlo Simulation**: Project portfolio performance across thousands of possible market scenarios
- **Scenario Analysis**: Test portfolio resilience against specific events like crypto crashes or regulatory changes
- **Risk Parity Optimization**: Balance risk across different asset classes for optimal allocation
- **Conditional Value at Risk (CVaR)**: Accurately measure downside risk beyond standard metrics
- **Bayesian Scenario Ranking**: Prioritize risk scenarios based on probability and portfolio impact
- **Dual Report System**: Free basic reports and premium comprehensive reports
- **Interactive Dashboard**: Visualize and explore portfolio data with adjustable parameters
- **Real Market Data**: Integration with Yahoo Finance for historical asset data

## Report Types

### Free Reports
- Basic portfolio overview
- Simple Monte Carlo projection
- Basic risk metrics
- Single static scenario analysis
- Standard disclaimer
- Premium upsell section

### Premium Reports
- Comprehensive portfolio analysis
- Detailed Monte Carlo simulation with statistics
- Advanced risk analysis (CVaR, drawdown, etc.)
- Multiple scenario analysis with recovery metrics
- Portfolio optimization recommendations
- Interactive dashboard access
- Comprehensive disclaimer

## Architecture

```
            ┌──────────────────────┐
            │   Frontend UI (Web)  │◄────────┐
            └──────────────────────┘         │
                     ▲                       │
                     │ Natural Language Query│
                     ▼                       │
        ┌──────────────────────────┐         │
        │     LLM Orchestrator     │─────────┘
        └──────────────────────────┘
                     │
     ┌───────────────┼────────────────────────────────────┐
     ▼               ▼                 ▼                  ▼
┌────────────┐ ┌──────────────┐ ┌────────────┐ ┌────────────────────┐
│ Monte Carlo│ │ Scenario Sim │ │ Risk Parity│ │ CVaR & Bayesian    │
│ Simulator  │ │ Engine        │ │ Allocator  │ │ Tail Risk Analyzer │
└────────────┘ └──────────────┘ └────────────┘ └────────────────────┘
     ▼               ▼                 ▼                  ▼
         ─────────────┬───────────────────────────────────
                      ▼
            ┌─────────────────────┐
            │   Report Generator  │
            └─────────────────────┘
                      │
       ┌─────────────┴─────────────┐
       ▼                           ▼
┌─────────────┐           ┌───────────────────┐
│ Free Report │           │  Premium Report   │
└─────────────┘           └───────────────────┘
```

## Interactive Dashboard

The included Plotly Dash dashboard provides:

- Interactive portfolio visualization
- Risk contribution analysis
- Scenario testing with real-time updates
- Historical data comparison from Yahoo Finance
- Optimization recommendations
- Mobile-responsive design
- Collapsible financial disclaimer

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/iso_financial_mvp.git
   cd iso_financial_mvp
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the API server:
   ```
   python -m iso_financial_mvp.app.main
   ```

4. Run just the dashboard (optional):
   ```
   python -m iso_financial_mvp.dashboard.dashboard
   ```

## Usage

### API Endpoints

- `POST /analyze`: Submit a natural language query with optional portfolio details
- `POST /api/generate-report`: Generate either a free or premium report
- `POST /api/compare-reports`: Generate both report types for comparison
- `GET /api/download-report/<filename>`: Download a generated report
- `GET /api/scenarios`: List available prebuilt scenarios
- `GET /api/assets`: List available assets from metadata

### Example Query

```json
{
  "query": "What if ETH drops 40% next year?",
  "user_profile": "intermediate",
  "report_type": "premium",
  "portfolio": {
    "assets": [
      {
        "type": "crypto",
        "name": "ethereum",
        "weight": 0.4
      },
      {
        "type": "crypto",
        "name": "bitcoin",
        "weight": 0.3
      },
      {
        "type": "traditional",
        "name": "us_stocks",
        "weight": 0.3
      }
    ],
    "initial_investment": 100000
  }
}
```

## Project Structure

```
iso_financial_mvp/
├── llm_interface/
│   ├── query_parser.py
│   └── ticker_identifier.py
├── simulation_engines/
│   ├── monte_carlo.py
│   ├── scenario_analysis.py
│   ├── risk_parity.py
│   ├── cvar.py
│   └── bayesian_ranker.py
├── report_generator/
│   ├── formatter.py
│   ├── free_report.py
│   ├── premium_report.py
│   ├── report_service.py
│   ├── disclaimer_template.py
│   └── assets/
├── data_sources/
│   └── yahoo_finance.py
├── dashboard/
│   ├── dashboard.py
│   ├── disclaimer_component.py
│   └── assets/
├── templates/
│   ├── query_prompt.j2
│   └── response_prompt.j2
├── data/
│   ├── assets_metadata.csv
│   └── ticker_mapping.json
├── app/
│   ├── main.py
│   ├── routes.py
│   └── report_api.py
├── requirements.txt
└── README.md
```

## Disclaimer

This software is for informational purposes only and does not provide financial advice. All investing involves risk, including the loss of principal. The projections generated by this tool are hypothetical and do not guarantee future results. Always consult with qualified financial professionals before making investment decisions.

## License

[License information]

## Contact

[Contact information]