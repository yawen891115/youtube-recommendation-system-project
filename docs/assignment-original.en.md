# YouTube Shorts Video Recommendation System - Complete Project

## Project Background and Objectives

In today's short video platforms, recommendation systems play a crucial role. A good recommendation system can recommend content that users might like based on their interests and behaviors, thereby enhancing user experience and platform engagement.

This project will guide you through a complete machine learning recommendation system development process, covering data processing, feature engineering, model training, API service development, frontend visualization, and containerized deployment. Through this project, you will master the complete skill stack for building production-grade recommendation systems.

---

## 1. Project Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend Layer                             │
│                   Streamlit Web Application                      │
│              (Real-time Prediction / Batch Prediction / Dashboard)│
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/REST
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        API Service Layer                          │
│                    FastAPI Backend                               │
│              (Prediction API / Model Management / Monitoring)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Model Inference Layer                      │
│              (Feature Engineering / Model Loading / Batch Predict)│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Data Processing Layer                      │
│              (Feature Engineering / Data Validation / Embeddings)│
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Dataset Description

This project provides three core data files stored in the `data/` directory.

### 2.1 User Data Table (users.csv)

This table records basic information about platform users, with each row representing a unique user.

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| user_id | string | Unique user identifier | user_0, user_1, ... |
| age | int | User age | 28, 35, 42 |
| gender | string | User gender | M (Male), F (Female), Other |
| preferred_category | string | User's preferred content category | Comedy, Music, Gaming, Education, etc. |
| join_date | datetime | User registration date | 2020-01-01 00:00:00 |

**Data Scale**: Approximately 10,000 users

### 2.2 Video Data Table (videos.csv)

This table records basic information about platform videos, with each row representing a unique video.

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| video_id | string | Unique video identifier | video_0, video_1, ... |
| creator_id | int | Video creator ID | 485, 977, 110 |
| category | string | Video category | Music, Fashion, Gaming, Education, etc. |
| duration_sec | float | Video duration (seconds) | 45.2, 120.5, 108.6 |
| upload_date | datetime | Video upload date | 2020-01-01 00:00:00 |
| view_count | int | Historical view count | 27956, 354970, 9682 |

**Data Scale**: Approximately 10,000 videos

### 2.3 Interaction Data Table (interactions_cleaned.csv)

This table records interaction behaviors between users and videos, with each row representing a user viewing behavior. This is the core data source for model training.

| Field Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| user_id | string | User ID who viewed the video | user_5763, user_5424 |
| video_id | string | Video ID that was viewed | video_6765, video_887 |
| watch_time | float | Duration watched (seconds) | 1.79, 75.23, 30.29 |
| liked | int | Whether liked (0=No, 1=Yes) | 0, 1 |
| shared | int | Whether shared (0=No, 1=Yes) | 0, 1 |
| timestamp | datetime | Interaction timestamp | 2024-09-05 07:53:23 |

**Data Scale**: Approximately 100,000 interaction records

---

## 3. Task Requirements

### 3.1 Task Objectives

Based on the provided user data, video data, and interaction data, build a production-grade short video recommendation system, including:

1. Complete data processing and feature engineering pipeline
2. Train multiple machine learning models and perform comparative optimization
3. Develop RESTful API services
4. Build visualization frontend
5. Implement containerized deployment

### 3.2 Detailed Task Description

#### Part 1: Data Processing and Feature Engineering (25 points)

**Objective**: Extract meaningful features from raw data to prepare for model training.

**Requirements**:

1. **Data Loading and Validation** (5 points)
   - Implement data loading module to read users.csv, videos.csv, interactions_cleaned.csv
   - Verify data integrity, check for missing values and anomalies
   - Calculate data scale and basic distributions

2. **Temporal Feature Extraction** (5 points)
   - Extract hour from timestamp (hour_of_day)
   - Extract day of week (day_of_week)
   - Determine if weekend (is_weekend)
   - Extract month (month)
   - Calculate recency (recency_days)
   - Time period classification (day_period: morning/afternoon/evening/night)

3. **User Aggregation Features** (5 points)
   - Calculate user's historical average watch time and standard deviation
   - Calculate user activity level (watch count)
   - Calculate user like rate and total likes
   - Calculate user share rate and total shares
   - User session features (session_id, session_length, session_duration_minutes)

4. **Video Aggregation Features** (5 points)
   - Calculate video average watch time
   - Calculate video popularity (view count)
   - Calculate video total likes and shares
   - Calculate engagement rate (engagement_rate)
   - Video freshness (video_age_days)

5. **Interaction Features and Advanced Features** (5 points)
   - Watch completion rate (completion_rate = watch_time / duration_sec)
   - Watch time normalization
   - Categorical feature encoding (One-Hot encoding)
   - User-video embedding features (SVD dimensionality reduction, 50 dimensions)
   - Advanced interaction features (hour_weekend_interaction, popularity_watch_ratio, etc.)

**Output**: Generate processed_interactions.csv and feature engineering artifacts in models/ directory (scalers.pkl, encoders.pkl, embeddings.pkl)

---

#### Part 2: Model Training and Optimization (25 points)

**Objective**: Train multiple machine learning models, select the best model, and perform hyperparameter optimization.

**Requirements**:

1. **Data Splitting** (3 points)
   - Split data chronologically into training (64%), validation (16%), and test (20%) sets
   - Use stratified sampling to ensure consistent positive/negative sample ratios
   - Set random seed (random_state=42)

2. **Baseline Model Training** (8 points)
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - Record accuracy and AUC for each model

3. **LightGBM Model Training** (7 points)
   - Configure LightGBM parameters
   - Implement early stopping mechanism
   - Handle class imbalance (is_unbalance=True)

4. **Hyperparameter Optimization** (7 points)
   - Use Optuna for automated hyperparameter search
   - Search key parameters: num_leaves, learning_rate, feature_fraction, etc.
   - Record optimal parameter combinations and best AUC

**Output**:
- models/best_model_*.pkl (trained model files)
- models/feature_columns.pkl (feature columns)
- MLflow experiment tracking records

---

#### Part 3: API Service Development (25 points)

**Objective**: Use FastAPI to develop RESTful API services supporting model prediction and system monitoring.

**Requirements**:

1. **Basic API Endpoints** (8 points)
   - `GET /health`: Health check endpoint, returns service status
   - `GET /model/info`: Model information query
   - `POST /predict`: Single prediction endpoint
   - `POST /predict/batch`: Batch prediction endpoint

2. **Request Validation** (5 points)
   - Define request/response models using Pydantic
   - Implement input validation (user_id, video_id format validation)
   - Proper error handling and exception returns

3. **Performance Optimization** (6 points)
   - Implement request caching (Redis)
   - Implement rate limiting
   - Async processing support

4. **Monitoring and Logging** (6 points)
   - Integrate Prometheus monitoring
   - Record prediction logs
   - Calculate response times

**API Specification**:

```python
# Single Prediction Request
POST / predict
{"user_id": "user_123", "video_id": "video_456", "watch_time": 45.0, "hour_of_day": 14}

# Single Prediction Response
{
    "user_id": "user_123",
    "video_id": "video_456",
    "probability": 0.75,
    "confidence": "high",
    "model_version": "1.0.0",
    "response_time_ms": 25.3,
    "timestamp": "2024-01-15T10:30:00",
}
```

---

#### Part 4: Frontend Visualization Development (15 points)

**Objective**: Use Streamlit to build visualization interface for prediction features and data analysis dashboard.

**Requirements**:

1. **Single Prediction Page** (5 points)
   - User input area (user_id, video_id, watch_time, hour_of_day)
   - Prediction results display (probability gauge, confidence level)
   - Response time display

2. **Batch Prediction Page** (5 points)
   - CSV file upload
   - Batch prediction results display
   - Results download functionality
   - Prediction probability distribution chart

3. **Data Analysis Dashboard** (5 points)
   - Request volume trend chart
   - Performance metrics monitoring (response time, error rate)
   - Feature importance display

---

#### Part 5: Containerized Deployment (10 points)

**Objective**: Use Docker and Docker Compose to implement service containerized deployment.

**Requirements**:

1. **Dockerfile Writing** (4 points)
   - Based on NVIDIA CUDA image (for GPU training scenarios)
   - Install Python dependencies
   - Copy project files
   - Set environment variables

2. **docker-compose.yml Configuration** (3 points)
   - FastAPI service configuration
   - Redis service configuration
   - MLflow service configuration
   - Volume mount configuration

3. **Deployment Verification** (3 points)
   - Local Docker deployment test
   - API endpoint availability verification
   - Streamlit frontend access verification

---

## 4. Project Structure

```
MLE_youtube/
├── data/                          # Data directory
│   ├── users.csv                  # User data
│   ├── videos.csv                 # Video data
│   ├── interactions_cleaned.csv   # Interaction data
│   └── processed_interactions.csv # Processed data
├── src/                           # Source code directory
│   ├── data_validation.py         # Data validation module
│   ├── feature_engineering.py     # Feature engineering module
│   ├── train.py                   # Model training module
│   ├── api.py                     # FastAPI service
│   └── streamlit_app.py           # Streamlit frontend
├── models/                        # Model storage directory
│   ├── best_model_*.pkl          # Trained models
│   ├── feature_columns.pkl        # Feature columns
│   ├── scalers.pkl               # Scalers
│   ├── encoders.pkl              # Encoders
│   └── embeddings.pkl            # Embedding vectors
├── requirements.txt               # Python dependencies
├── docker-compose.yml             # Docker Compose configuration
├── Dockerfile.train.txt          # Training image Dockerfile
└── README.md                     # Project documentation
```

---

## 5. Grading Criteria

| Part | Points | Grading Standards |
|------|--------|-------------------|
| Data Processing & Feature Engineering | 25 pts | Data validation (5), Temporal features (5), User features (5), Video features (5), Interaction & Advanced features (5) |
| Model Training & Optimization | 25 pts | Data splitting (3), Baseline models (8), LightGBM (7), Hyperparameter optimization (7) |
| API Service Development | 25 pts | Basic endpoints (8), Request validation (5), Performance optimization (6), Monitoring & logging (6) |
| Frontend Visualization | 15 pts | Single prediction (5), Batch prediction (5), Data dashboard (5) |
| Containerized Deployment | 10 pts | Dockerfile (4), Compose configuration (3), Deployment verification (3) |

**Total Score**: 100 points

---

## 6. Environment Setup and Dependencies

### 6.1 Python Environment Requirements

```txt
# Core Machine Learning Libraries
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.25.0
joblib>=1.3.0

# Advanced Machine Learning Libraries
lightgbm>=4.0.0
optuna>=3.0.0

# API & Deployment
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0
redis>=4.6.0

# Frontend
streamlit>=1.26.0
plotly>=5.18.0

# MLOps
mlflow>=2.8.0

# Data Validation
pandera>=0.20.0
```

### 6.2 Quick Installation

```bash
pip install -r requirements.txt
```

### 6.3 Service Startup

```bash
# 1. Start Redis (optional)
redis-server

# 2. Start MLflow (optional)
mlflow ui --port 5000

# 3. Start FastAPI service
uvicorn api:app --reload --port 8000

# 4. Start Streamlit frontend (new terminal)
streamlit run src/streamlit_app.py
```

---

## 7. Implementation References

### 7.1 Feature Engineering Key Code

```python
class FeatureEngineer:
    def create_temporal_features(self, df):
        """Temporal feature extraction"""
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["month"] = df["timestamp"].dt.month
        df["recency_days"] = (df["timestamp"].max() - df["timestamp"]).dt.days
        return df

    def create_interaction_features(self, df, videos_df):
        """Interaction feature extraction"""
        df["duration_sec"] = df["video_id"].map(videos_df.set_index("video_id")["duration_sec"])
        df["completion_rate"] = np.clip(df["watch_time"] / df["duration_sec"], 0, 1)
        return df
```

### 7.2 FastAPI Endpoint Example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="YouTube Shorts Recommendation API")


class PredictionRequest(BaseModel):
    user_id: str = Field(..., pattern=r"^user_\d+$")
    video_id: str = Field(..., pattern=r"^video_\d+$")
    watch_time: float = Field(..., ge=0, le=300)


@app.post("/predict")
async def predict(request: PredictionRequest):
    # Model prediction logic
    return {"probability": 0.75, "confidence": "high"}
```

### 7.3 Streamlit Page Example

```python
import streamlit as st

st.title("Shorts Recommendation System")

user_id = st.text_input("User ID")
video_id = st.text_input("Video ID")
watch_time = st.slider("Watch Time", 0, 180, 45)

if st.button("Predict"):
    result = call_api("predict", {...})
    st.success(f"Probability: {result['probability']}")
```

---

## 8. Important Notes

### 8.1 Data Leakage Issues

⚠️ When constructing features, you must avoid using "future" information. For example, when calculating user like rate, you should not include the current interaction record itself.

### 8.2 Class Imbalance

In recommendation data, most videos are not liked (liked=0), which is normal. During training, you can use `class_weight='balanced'` or `is_unbalance=True` parameters.

### 8.3 Code Standards

- Keep code clean with appropriate comments
- Write docstrings for functions and classes
- Use meaningful variable names
- Print key logs for easier debugging

---

## 9. Submission Requirements

1. **Complete Code**: All code including data processing, feature engineering, model training, API service, and frontend development
2. **Data Exploration Report**: Key statistical indicators and visualization charts
3. **Experiment Report**:
   - Feature engineering approach
   - Model training results comparison
   - Hyperparameter optimization process
   - Problems encountered and solutions
4. **Deployment Instructions**: How to run and deploy locally

---

## 10. Reference Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Docker Documentation](https://docs.docker.com/)

---

