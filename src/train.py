"""
1. 特征选择：排除ID、时间、标签等不能作为输入的列
2. 数据划分：训练/验证/测试集
3. 模型训练：基线模型（逻辑回归、随机森林、梯度提升）+ LightGBM
4. 超参数优化：Optuna自动搜索最佳参数(训练轮次，随机森林的子树数量)
5. 模型评估：AUC、准确率（预测正确的/总数量）、精确率、召回率Recall、F1
6. 实验跟踪：MLflow记录参数，指标和模型
"""

from pathlib import Path
import logging
import joblib #保存模型训练好的参数
import mlflow #记录员
import numpy as np
import optuna #自动优化超参数
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import(
    GradientBoostingClassifier, # 梯度提升分类器
    RandomForestClassifier # 随机森林分类器
)

from sklearn.metrics import(
    accuracy_score,
    classification_report, # 分类报告（精确率、召回率、F1）
    roc_auc_score
)

from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ModelTrainer:
    def __init__(self, tracking_uri=None):
        # 读取Mlflow跟踪URI
        uri = tracking_uri or os.get("MLFLOW_TRACKING_URI", "file:./mlruns")

        #设置MLflow跟踪服务器
        mlflow.set_tracking_uri(uri)

        #设置实验名称（类似项目名）
        mlflow.set_experiment("XXXX")

        #定义训练时所用的特征列表
        self.feature_colums = []

        #存储各模型的验证指标
        self.results = {}

    def train_baselines(self, X_train, y_train, X_val, y_val):
        #定义三个基线模型
        models = {
            "logistic_regression": LogisticRegression(
                random_state=42,
                class_weight="balanced",
                max_iter=1000,
            ),

            "random": RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1 #和gpu个数有关，这是同时处理，同时用上电脑所有的gpu
            ),

            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                learning_rate=0.1
            )
        }

        for name, model in models.items():
            with mlflow.start_run(run_name=f"baseline_{name}"):

                model.fit(X_train, y_train)


def main():
    # 1.读取数据
    processed = pd.read_csv("data/processed_interaction.csv")

    # 2.创建一个训练器

if __name__ == "__main__":
    main()
