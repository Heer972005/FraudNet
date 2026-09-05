🎯 The Problem

Most fraud detection systems evaluate each transaction in isolation. That works fine for lone bad actors, but organized fraud rings easily slip through by spreading their activity across multiple accounts.

The numbers tell the story:
- Global e-commerce fraud losses hit $48 billion in 2025, with cumulative losses projected to exceed $362 billion over the next five years
(https://www.ringly.io/blog/ecommerce-fraud-statistics-2026)

- Refund abuse alone costs merchants approximately **$100 billion annually** worldwide (https://stripe.com/blog/analyzing-first-party-fraud-trends-account-free-trial-and-refund-abuse)

- 57% of merchants reported increasing rates of refund and policy abuse in 2025, with some seeing surges of 50% or more
(https://thepaypers.com/fraud-and-fincrime/expert-views/2025-payment-fraud-retrospective-what-did-we-learn)

- Friendly fraud (customers disputing valid transactions) now accounts for 44% of all chargebacks, and nearly 79% of disputes are classified as friendly fraud (https://www.digitaltransactions.net/nearly-44-of-chargebacks-are-labeled-friendly-fraud-finds-a-chargebacks911-report/)

- For every $1 lost to fraud, U.S. merchants incur $4.61 in total costs once fees, labor, and lost goods are counted (https://redo.com/resources/articles/chargebacks/chargeback-statistics)

- The average enterprise now loses $11.4 million annually to fraud—a 7.5% year-over-year increase (https://www.ravelin.com/blog/ravelin-fraud-survey-2026-press-release)

💼 Business Impact

FraudNet helps merchants fight back against a growing threat:

- Detect coordinated fraud rings earlier — Traditional systems miss organized abuse because they analyze transactions individually. FraudNet spots the patterns humans can't see.
- Reduce refund abuse — With refund abuse displacing payment fraud as the #1 merchant threat in 2026, catching these rings early is critical
(https://nhimg.org/articles/refund-abuse-is-now-the-top-digital-commerce-fraud-threat/)
- Prioritize investigations with confidence — Instead of reviewing hundreds of alerts, merchants focus on the highest-risk communities first
- Understand AI decisions through clear explanations — No more black-box predictions; every flag comes with reasoning your team can act on
- Save operational time with automated risk assessment — The average merchant now loses $5.13 for every confirmed dollar of fraud when all costs are included. Automation reduces that burden. 
(https://nhimg.org/articles/refund-abuse-is-now-the-top-digital-commerce-fraud-threat/)

Real-world context:
- 83% of enterprise merchants reported an increase in friendly fraud over the past three years (https://www.fraudbeat.com/friendly-fraud-employee-collusion/)
- 67% of merchants say refund abuse is a "moderate" or "significant" concern, up from 55% in 2024 (https://www.fraudbeat.com/friendly-fraud-employee-collusion/)
- 38% of merchants pass chargeback costs to consumers through higher prices—fraud detection directly impacts customer pricing 
(https://www.fraudbeat.com/friendly-fraud-employee-collusion/)

✨ What Makes It Different
- Real-time AI payment monitoring with live transaction simulation

- XGBoost-based fraud probability prediction for every account

- Graph Analytics to detect coordinated fraud rings sharing devices, payment methods, or fingerprints

- Automatic community risk scoring to prioritize investigations

- Interactive fraud network visualization showing how suspicious accounts connect

- Merchant Action Center with clear, non-technical recommendations

- AI-powered Copilot to guide fraud investigations

= PDF investigation reports you can share with your team

- Business insights dashboard with interactive charts

- Community Deep Dive to explore suspicious clusters in detail

- Auto-focus on highest-risk community so you always know where to start

🏗️ How It Works
- Load merchant accounts, transactions, and refunds

- Extract graph features (shared devices, payment fingerprints, etc.)

- XGBoost predicts fraud probability for every account

- Group connected suspicious accounts into communities

- Score each community based on ML risk

- Dashboard explains the reasoning behind every alert and recommends actions

🛠️ Tech Stack
| Category         | Technology    |
| ---------------- | ------------- |
| Language         | Python        |
| Dashboard        | Streamlit     |
| Machine Learning | XGBoost       |
| Data Processing  | Pandas, NumPy |
| Graph Analytics  | NetworkX      |
| Visualization    | Plotly        |
| Explainability   | SHAP          |
| Reports          | ReportLab     |

📁 Project Structure
FraudNet/
│
├── data/
│   └── raw/
│       ├── accounts.csv
│       ├── transactions.csv
│       └── refunds.csv
│
├── outputs/
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   ├── shap_waterfall.png
│   └── threshold_savings.png
│
├── src/
│   ├── app.py
│   ├── train_model.py
│   ├── graph_detector.py
│   ├── data_generator.py
│   ├── live_simulator.py
│   ├── report_generator.py
│   └── config.py
│
├── main.py
├── requirements.txt
└── README.md

🖥️ Dashboard Walkthrough

Home Dashboard
The landing page surfaces the most critical fraud community based on the trained ML model—no digging required.

Merchant Action Center
Instead of cryptic risk scores, merchants get plain-English actions like:
- Pause automatic refunds
- Verify linked accounts
- Monitor shared payment fingerprints
- Keep tracking connected devices

Live AI Payment Monitor
Every new payment gets analyzed before approval. For each transaction, you'll see:
- Merchant name
- Transaction amount
- Fraud probability
- AI decision (Approved / Fraud Alert)
- Reasons behind the decision

Example:
| Merchant | Amount | AI Decision |
|----------|--------|-------------|
| HomeNest | ₹573 | Approved |
| TechKart | ₹24,500 | Fraud Alert |

Business Insights
Interactive charts help merchants spot fraud patterns at a glance:
- Account distribution
- Refund behavior
- Fraud probability distribution
- Highest-risk communities

Community Deep Dive
Investigate suspicious clusters in detail. Each community view shows:
- ML risk score
- Member accounts
- Shared devices
- Average account age

Fraud Relationship Network
Visualize how suspicious accounts connect to each other—perfect for spotting organized rings.

Most Suspicious Accounts
A prioritized investigation list where each account displays:
- Fraud probability
- Number of connections
- Refund rate

Explainable AI
FraudNet doesn't just predict fraud—it explains it.

Example explanation:
> Community 4 was flagged because multiple accounts share payment fingerprints, devices, and recent account creation patterns, forming a dense connected network.
 
🧠 Model Features

The XGBoost model uses engineered features that blend behavioral and relationship signals:
- Account Age
- Community Risk
- Graph Degree
- Weighted Graph Degree
- Refund Rate


🚀 Installation

1. Clone the repository

git clone https://github.com/yourusername/FraudNet.git
cd FraudNet

2. Create and activate a virtual environment

python -m venv venv

Windows:
venv\Scripts\activate

macOS/Linux:
source venv/bin/activate


3. Install dependencies
pip install -r requirements.txt


4. Run the dashboard
streamlit run src/app.py



📊 Machine Learning Pipeline
Train the model with:

python src/train_model.py

This generates:
- Trained XGBoost model
- ROC Curve
- Precision-Recall Curve
- SHAP explanations
- Feature importance plots
- Threshold savings analysis


💼 Business Impact

FraudNet helps merchants:
- Detect coordinated fraud rings earlier
- Reduce refund abuse
- Prioritize investigations with confidence
- Understand AI decisions through clear explanations
- Save operational time with automated risk assessment


🔮 Future Improvements

- Real payment gateway integration
- Multi-merchant fraud intelligence sharing
- Email and Slack notifications
- Real-time streaming pipeline using Kafka
- Graph Neural Networks (GNNs) for deeper relationship learning

 
 👨‍💻 Author

**Heer Mehta**  
AI-Powered Fraud Detection Dashboard built using Python, Streamlit, XGBoost, Graph Analytics, and Explainable AI.


📄 License

This project is created for educational, research, and hackathon purposes.
