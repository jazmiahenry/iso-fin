# ISO Financial GPT Agent

This module provides a finetuned GPT-4.1 agent specialized in financial analysis and calculations.

## Overview

The ISO Financial GPT Agent combines a finetuned GPT-4.1 model with the toolkit of financial calculation functions
to provide sophisticated financial advice, analysis, and calculations. The model has been finetuned on:

1. Comprehensive financial concepts including monetary policy, market efficiency, capital budgeting,
   portfolio theory, financial instruments, and more.
2. Tool usage to correctly invoke the appropriate financial calculations when needed.

## Components

### 1. Finetuning Script (`finetune.py`)

This script handles the finetuning process for the GPT model using OpenAI's API.

```bash
# Run finetuning
python finetune.py --api-key your_openai_api_key

# Check status of a finetuning job
python finetune.py --check-job job_id_here --api-key your_openai_api_key
```

Key features:
- Processes training examples from `finetuning.json`
- Integrates tool knowledge from `tool_tuning.json`
- Creates and manages finetuning jobs with OpenAI
- Saves model information for later use

### 2. Financial Agent (`financial_agent.py`)

The main class for interacting with the finetuned model. It provides:
- Registration of all financial tools from the toolkit
- Tool execution handling
- Query processing with context
- Error handling and logging

## Usage

### Finetuning the Model

```bash
cd /Users/jazmiahenry/reports/iso_financial_mvp/gpt_agent
python finetune.py
```

### Using the Agent in Python

```python
from iso_financial_mvp.gpt_agent import FinancialGPTAgent

# Initialize the agent
agent = FinancialGPTAgent()

# Ask a financial question
response = agent.answer_question(
    "What would be a suitable asset allocation for a retirement portfolio with a 20-year horizon?",
    context={"risk_tolerance": "moderate"}
)

# Print the response
print(response["response"])

# View any calculations performed
for calc in response.get("calculations", []):
    print(f"Tool: {calc['tool']}")
    print(f"Inputs: {calc['inputs']}")
    print(f"Result: {calc['result']}")
```

### API Integration

The agent is integrated into the main FastAPI application and can be accessed via the `/gpt-advice` endpoint.

Example request:
```json
{
  "query": "How should I allocate my portfolio during high inflation?",
  "context": {
    "risk_tolerance": "moderate",
    "investment_horizon": "long-term",
    "portfolio": {
      "SPY": 0.4,
      "QQQ": 0.3,
      "AGG": 0.2,
      "GLD": 0.1
    }
  }
}
```

### Streamlit Integration

The agent is also accessible via the Streamlit interface at the "GPT Advisor" page, allowing users to:
- Ask financial questions in natural language
- Provide context about their portfolio, risk tolerance, etc.
- See detailed responses including any calculations performed

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `API_URL`: URL of the API for Streamlit integration (defaults to http://localhost:8000)

## Dependencies

- openai >= 1.0.0
- numpy
- pandas
- logging
- requests (for Streamlit integration)