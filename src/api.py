from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI # Flask / Django
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

# ============================================
# 1. 全局变量：模型和特征列
# ============================================
model = None
feature_columns = None

# ============================================
# 2. 请求和响应的数据模型（Pydantic）
# ============================================
class PredictionRequest(BaseModel):
    """预测请求体：API 收到这个 JSON 后会自动校验字段"""
    user_id: str = Field(..., description="用户ID，如 user_123")
    video_id: str = Field(..., description="视频ID，如 video_456")
    watch_time: float = Field(..., ge=0, description="观看时长（秒）")
    hour_of_day: Optional[int] = Field(None, ge=0, le=23, description="小时（0-23）")


class PredictionResponse(BaseModel):
    """预测响应体：返回给前端的字段"""
    user_id: str
    video_id: str
    probability: float
    confidence: str
    model_version: str
    response_time_ms: float
    timestamp: datetime


class BatchPredictionRequest(BaseModel):
    """批量预测请求体：包含多条 prediction 请求"""
    requests: List[PredictionRequest]


# ============================================
# 3. 启动时加载模型
# ============================================
def load_models():
    """加载训练好的模型文件，失败则用 DummyClassifier 兜底"""
    global model, feature_columns
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # 加载模型文件（best_model_xxx.pkl）
    model_files = list(models_dir.glob("best_model_*.pkl"))
    if model_files:
        model = joblib.load(model_files[0])
    else:
        # 没有训练好的模型，用随机猜测的虚拟模型
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy="uniform", random_state=42)

    # 加载特征列名（与训练时一致）
    feature_columns_path = models_dir / "feature_columns.pkl"
    if feature_columns_path.exists():
        import pickle
        with open(feature_columns_path, "rb") as f:
            feature_columns = pickle.load(f)
    else:
        feature_columns = ["watch_time", "hour_of_day", "day_of_week"]


load_models()

# ============================================
# 4. 创建 FastAPI 应用
# ============================================
app = FastAPI(title="YouTube Shorts Recommendation API")

# CORS：允许前端跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# 5. 辅助函数
# ============================================
def preprocess_request(request: PredictionRequest) -> pd.DataFrame:
    """把请求转成模型需要的 DataFrame，缺失的特征列填 0"""
    now = datetime.now()
    data = {
        "watch_time": [request.watch_time],
        "hour_of_day": [request.hour_of_day or now.hour],
        "day_of_week": [now.weekday()],
        "duration_sec": [60],
        "age": [25],
        "completion_rate": [min(request.watch_time / 60, 1.0)],
    }
    df = pd.DataFrame(data)
    # 补齐所有训练时的特征列
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df.reindex(columns=feature_columns, fill_value=0)


def get_confidence(prob: float) -> str:
    """根据概率值判断置信度：离 0.5 越远越自信"""
    if prob > 0.8 or prob < 0.2:
        return "high"
    elif prob > 0.6 or prob < 0.4:
        return "medium"
    return "low"


# ============================================
# 6. API 路由（4 个接口）
# ============================================

# 接口 1：健康检查
# 看模型是否正常加载了
@app.get("/health")
def health():
    """检查服务是否正常运行"""
    return {
        "status": "healthy" if model else "degraded",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
    }


# 接口 2：单条预测
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest): 
    """根据用户ID、视频ID、观看时长预测是否点赞"""
    import time
    start = time.time()

    # 把请求转成 DataFrame
    df = preprocess_request(request)

    # 模型推理：取点赞（正类）的概率
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(df)[0][1])
    else:
        prob = float(model.predict(df)[0])
    prob = float(np.clip(prob, 0, 1))  # 限制在 [0, 1] 范围

# 展示给前端的东西
    return PredictionResponse(
        user_id=request.user_id,
        video_id=request.video_id,
        probability=prob,
        confidence=get_confidence(prob),
        model_version="1.0.0",
        response_time_ms=(time.time() - start) * 1000, #响应时间
        timestamp=datetime.now(),
    )


# 接口 3：批量预测
# 在单条预测的基础上加上一层for循环
@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    """一次性预测多条，单条失败不影响其他条"""
    import time
    start = time.time()
    results = []

    for req in request.requests:
        try:
            df = preprocess_request(req)
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(df)[0][1])
            else:
                prob = 0.5
            prob = float(np.clip(prob, 0, 1))
            results.append({
                "user_id": req.user_id,
                "video_id": req.video_id,
                "probability": prob,
                "confidence": get_confidence(prob),
            })
        except Exception as e:
            # 错误隔离：单条失败时返回错误信息，其他条继续
            results.append({
                "user_id": req.user_id,
                "video_id": req.video_id,
                "error": str(e),
            })

    return {
        "results": results,
        "batch_size": len(request.requests),
        "response_time_ms": (time.time() - start) * 1000,
        "timestamp": datetime.now().isoformat(),
    }


# 接口 4：模型信息
@app.get("/model/info")
def model_info():
    """返回当前加载的模型元信息"""
    return {
        "model_type": type(model).__name__ if model else None,
        "feature_count": len(feature_columns) if feature_columns else 0,
        "features": feature_columns[:10] if feature_columns else [],
        "status": "loaded" if model else "not loaded",
    }


# ============================================
# 7. 启动入口
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
