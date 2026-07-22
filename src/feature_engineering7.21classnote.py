from pathlib import Path
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LableEncoder, OneHotEncoder, MinMaxScaler

#logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(levelname)s-%(message)s"
)

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        """初始化特征工程"""
        self.scalers = {}  # save MinMaxScaler

        self.encoders = {}  #保存类别编码

        self.embeddings = {}  #SVD -> 找隐藏特征，和PCA类比学习

    @staticmethod
    def _datetime(series):
        # 解析失败返回NaT
        converted = pd.to_datetime(series, errors="coerce") #出现日期错误的时候如2.30他就会转换成NaT
        # 解析之后返回空的话就把现在的时间戳填充进去
        return converted.fillna(pd.Timestamp.now())

    def create_temporal_features(self, df):
        """小时、星期、周末、月份、时间线"""
        # copy()创建深拷贝，不修改原始的DataFrame
        result = df.copy()

        # timestamp从字符串转换为datatime对象 xxxx-xx-xx xx:xx:xx
        result["timestamp"] = self._datetime(result["timestamp"]) #把等号右边处理过的值赋值给等号左边，覆盖掉旧的timestamp列

        # 提取小时（0-23）
        result["hour_of_day"] = result["timestamp"].dt.hour #新增加了新的一列（hour）

        # 提取周几
        result["day_of_week"] = result["timestamp"].dt.dayofweek #新增加了新的一列（day_of_week）

        # 判断是否周末
        result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)

        # 提取月份 (1-12)
        result["month"] = result["timestamp"].dt.month

        # -1~6 night; 6~12 morning; 12~18 afternoon; 18~23 evening
        result["day_period"] = pd.cut(
            result["hour_of_day"],
            bins=[-1, 6, 12, 18, 23]
            labels=["night", "morning", "afternoon", "evening"]
        )

        # 最新时间
        latest = result["timestamp"].max()

        # 传播天数
        result["recency_days"] = (latest - result["timestamp"]).dt.total_seconds() / (60 * 60 * 24)

        return result

    def create_user_features(self, interactions_df, users_df):

        interactions = interactions_df.copy()
        interactions["timestamp"] = self._datetime(interactions["timestemp"])

        #groupby
        stats = interactions.groupby("user_id").agg(
            # 平均观看时长
            watch_time_mean = ("watch_time", "mean"),
            # 观看时长的标准差
            watch_time_std = ("watch_time", "std"),
            # 观看次数
            watch_time_count = ("watch_time", "count"),
            # 点赞率
            liked_mean = ("liked", "mean"),
            # 点赞总数
            liked_sum = ("liked", "sum"),
            # 分享率
            shared_mean = ("shared", "mean"),
            # 分享总数
            shared_sum = ("shared", "sum"),
            # 首次交互
            first_seen = ("timestamp", "min"),
            # 最近交互
            last_seen = ("timestamp", "max"),
        ).reset_index() #把分组索引转化为列

        # 用户活跃天数
        stats["user_tenure_days"] = (
            stats["last_seen"] - stats["first_seen"]
        ).dt.total_seconds() / (60 * 60 * 24)

        # 平均每天的交互次数
        stats["interactions_per_day"] = (
            stats["watch_time"] / stats["user_tenure_days"]
        )

        result = users_df.merge(stats, on = "user_id", how = "left")

        numeric = result.select_detypes(include = np.nan).colums
        result[numeric] = result[numeric].fillna(0)

        return result





def main():  #python中不知道写什么要写pass，可以让他过
    #读取数据
    users = pd.read_csv("data/users.csv") #涉及到中文文本格式的话编码格式encoding就要变成utf-8
    videos = pd.read_csv("data/videos.csv")
    interactions = pd.read_csv("data/interactions.csv")

    #创建特征工程
    FeatureEngineer()

if __name__ == "__main__": #主函数
    main()

#1、数据check
#2、数据处理 -> 特征工程
#3、模型训练 -> 模型save
#4、部署 -> api -> frontnd(Docker)

"""
svd需要了解一下，代码直接掉api就行，需要形成一套自己能够表述清楚的东西
机器学习能不能手动做一下
代码看一下
"""