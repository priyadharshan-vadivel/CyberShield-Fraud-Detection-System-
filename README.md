# MILITARY-GRADE AI SYSTEM TO DETECT AND DISMANTLE MONEY LAUNDERING FLOWS

CyberShield is an AI platform that detects money laundering using an ensemble of machine learning models and a rule engine. Unlike traditional "black box" systems, it uses Explainable AI (SHAP) to show exactly *why* a transaction is flagged as suspicious, ensuring transparency for investigators and regulators.

The system provides an end-to-end workflow, from detection and risk scoring to case management and network analysis, all within an interactive dashboard.

## Key Features

- **Hybrid Detection Engine**: Combines a rule-based engine for known fraud patterns with machine learning (Random Forest & Isolation Forest) for supervised and unsupervised anomaly detection.
- **Explainable AI (XAI)**: Integrates SHAP to provide clear, human-readable explanations for every flagged transaction, showing the contribution of each feature to the risk score.
- **Investigation Queue**: A built-in case management system allows analysts to triage, track status (NEW, IN_PROGRESS, RESOLVED), and document investigations in a persistent SQLite database.
- **Interactive Network Visualization**: Generates network graphs to help analysts uncover complex money-laundering rings and visualize connections between suspicious accounts.
- **Comprehensive Dashboard**: Built with Streamlit and Plotly, the dashboard provides key metrics, high-risk alert details, and access to all system features in a user-friendly interface.


## Technology Stack

| Category      | Technology                               | Purpose                                                    |
| :------------ | :--------------------------------------- | :--------------------------------------------------------- |
| **Frontend** | Streamlit, Plotly                        | Web dashboard creation and interactive charts.          |
| **Backend** | Python 3.8+, Pandas, NumPy               | Main programming language and data manipulation.  |
| **AI / ML** | Scikit-learn, SHAP                       | Machine learning and AI explanations.               |
| **Data Storage**| SQLite, CSV, Pickle                      | Investigation queue, data exchange, and model storage.|
| **Graph Analysis**| NetworkX                                 | Analyzing connections between accounts.                 |

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites
- Python 3.8 or newer 
- 4GB RAM (8GB recommended)

### Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/priyadharshan-vadivel/CyberShield-Fraud-Detection-System.git
    cd CyberShield-Fraud-Detection-System
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate on Windows
    venv\Scripts\activate

    # Activate on Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    The project includes a `requirements.txt` file to install all necessary packages.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    Create a file named `.env` in the root directory of the project. Copy the content below into the file and adjust the values if needed.
    ```ini
    DATA_PATH=data/transactions.csv
    MODEL_PATH=models/aml_model.pkl
    QUEUE_PATH=investigation_queue.db
    HIGH_RISK_THRESHOLD=8.0
    MEDIUM_RISK_THRESHOLD=6.0
    MAX_NETWORK_NODES=100
    LOG_LEVEL=INFO
    ```

## How to Run

Once the setup is complete, run the application from your terminal:
```bash
streamlit run app.py
