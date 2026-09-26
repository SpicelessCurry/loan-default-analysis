# 🛡️ CreditScope — Credit Risk Assessment Platform

A web-based chatbot application for assessing loan default risk using deep learning models (RNN, ANN, MLP). Admins can upload trained Keras models and applicant datasets, then use an intelligent chatbot to look up individual applicants and receive AI-powered risk assessments.

## ✨ Features

- **Admin Authentication** — Secure login with configurable credentials
- **Model Management** — Upload, select, and manage multiple Keras models (RNN, ANN, MLP)
- **Dataset Upload** — Import applicant datasets (CSV) with automatic validation
- **Risk Assessment Chatbot** — Enter an applicant ID to get:
  - Default probability from all selected models
  - Color-coded risk level (Low / Moderate / High / Very High)
  - Detailed risk factor analysis across 11 categories
  - AI-generated risk assessment report (via OpenAI (GPT-4o) or rule-based fallback)
  - Full applicant data view
- **Dark Theme UI** — Modern, responsive dashboard with real-time interactions

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

# Install packages
pip install -r requirements.txt
```

> **Note:** `tensorflow` is a large package (~500MB+). For CPU-only inference, you can use `tensorflow-cpu` instead.

### 2. Configure Environment (Optional)

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | `creditscope-secret-key-change-me` | Flask session secret key |
| `ADMIN_USERNAME` | `admin` | Login username |
| `ADMIN_PASSWORD` | `admin123` | Login password |
| `OPENAI_API_KEY` | *(empty)* | OpenAI (GPT-4o) API key for AI-powered assessments |

### 3. Run the Application

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) and log in with the default credentials (`admin` / `admin123`).

## 📋 Usage Guide

### Step 1: Upload Models

1. Navigate to the **Models** tab
2. Enter a model name (e.g., "Credit Risk ANN v1")
3. Select the model type: **RNN**, **ANN**, or **MLP**
4. Upload a Keras model file (`.h5` or `.keras`)
5. Check the box to select models for prediction
6. Click **Save Selection**

> Models must be full Keras models saved with `model.save()`. For RNN models, features will be automatically reshaped to `(1, num_features, 1)`.

### Step 2: Upload Dataset

1. Navigate to the **Dataset** tab
2. Upload a CSV file containing applicant data
3. The CSV **must** include an `SK_ID_CURR` column (applicant ID)
4. Sample applicant IDs will be displayed for quick access

### Step 3: Chat with the Risk Assessor

1. Navigate to the **Chat** tab
2. Enter an applicant ID (e.g., `100002`)
3. The system will:
   - Look up the applicant in the dataset
   - Run predictions through all selected models
   - Analyze 30+ risk rules across financial, bureau, payment, and demographic categories
   - Generate a comprehensive risk assessment report

## 📊 Expected Dataset Columns

The dataset should contain `SK_ID_CURR` (applicant ID) plus 107 feature columns. Key feature categories include:

| Category | Example Features |
|----------|-----------------|
| Financial | `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `CREDIT_INCOME_RATIO` |
| Bureau History | `BUREAU_OVERDUE_COUNT`, `BUREAU_DEBT_INCOME_RATIO` |
| Previous Applications | `PREVIOUS_APPROVAL_RATE`, `PREVIOUS_REFUSAL_RATE` |
| Payment Behavior | `LATE_PAYMENT_RATE`, `AVG_PAYMENT_DELAY` |
| External Scores | `EXT_SOURCE_2`, `EXT_SOURCE_3` |
| Demographics | `AGE_YEARS`, `CNT_CHILDREN` |
| Employment | `DAYS_EMPLOYED`, `OCCUPATION_TYPE` |

## 🏗️ Project Structure

```
CreditScope/
├── app.py                 # Main Flask application with all routes
├── config.py              # Configuration (env vars, paths, limits)
├── ml_engine.py           # ModelManager — load, predict, manage Keras models
├── risk_analyzer.py       # RiskAnalyzer — rule-based + LLM risk assessment
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── templates/
│   ├── login.html         # Login page (self-contained HTML/CSS/JS)
│   └── dashboard.html     # Main dashboard (models, dataset, chat)
└── uploads/
    ├── models/            # Stored model files + registry.json
    └── datasets/          # Stored CSV files
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/login` | Authenticate admin user |
| `POST` | `/api/logout` | End session |
| `GET` | `/api/auth/status` | Check authentication status |
| `GET` | `/api/models` | List all uploaded models |
| `POST` | `/api/models/upload` | Upload a new model file |
| `POST` | `/api/models/select` | Select models for prediction |
| `DELETE` | `/api/models/<id>` | Delete a model |
| `POST` | `/api/dataset/upload` | Upload applicant dataset |
| `GET` | `/api/dataset/info` | Get dataset metadata |
| `GET` | `/api/dataset/applicant/<id>` | Get applicant data by ID |
| `POST` | `/api/chat` | Main chatbot endpoint |

## 🤖 AI-Powered Assessments

When a `OPENAI_API_KEY` is configured in `.env`, the chatbot uses OpenAI (GPT-4o) to generate professional risk assessment reports. Without it, a comprehensive rule-based assessment engine provides detailed analysis covering:

- Overall risk level with recommendations
- Model prediction summaries
- Risk factors grouped by category with severity ratings
- Actionable recommendations

## ⚠️ Notes

- This application is designed for **demonstration and educational purposes**
- Model architectures must match the uploaded weight files
- The dataset is stored **in-memory** — it will reset when the server restarts
- For production use, add proper database storage, HTTPS, and rate limiting
