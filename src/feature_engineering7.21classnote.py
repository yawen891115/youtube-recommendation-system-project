from pathlib import Path
import logging
import pickle #可以保存程序里的变量、列表、字典、模型等，下次直接load就行，不用重新训练模型，对ml尤其好用
import numpy as np
import pandas as pd
from sklearn.preprocessing import LableEncoder, OneHotEncoder, MinMaxScaler
#LabelEncoder：把类别标签转成整数编号，通常用于目标变量y
#OneHotEncoder：把特征X转成0/1，如性别，男性0，女性1
#MinMaxScaler：把数值型特征缩放到0-1之间，通常用于特征X

#logging：创建一个info级别的日志，按照时间、日志级别、日志信息的格式输出到终端
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(levelname)s-%(message)s"
)

logger = logging.getLogger(__name__) #创建一个日志小助手

class FeatureEngineer:
    def __init__(self):
        """初始化特征工程，以下三个都是字典，key是列名，value是对应的处理器"""
        self.scalers = {}  # save MinMaxScaler
        # 存放每一列对应的 MinMaxScaler 实例,比如 {"age": MinMaxScaler(), "income": MinMaxScaler()}

        self.encoders = {}  #保存类别编码
        # 存放每一列对应的类别编码器,比如 {"gender": OneHotEncoder(), "city": LabelEncoder()}

        self.embeddings = {}  #SVD -> 找隐藏特征，和PCA类比学习
        # 存放 SVD 分解出来的隐藏特征,比如 {"user_id": SVD(), "video_id": SVD()}

    @staticmethod #在class中，如果一个函数第一个参数不是self，而是一个普通的参数，那么就加上@staticmethod，表示这个函数是静态方法
    # 静态方法和class创建出的object无关，而其他函数都和object有关，因为都有self参数，self可以理解成class创建出的object
    def _datetime(series):
        # 解析失败返回NaT
        converted = pd.to_datetime(series, errors="coerce") 
        # 把字符串列转成datetime类型;遇到解析不了的(比如2月30日),不报错,而是变成 NaT(Not a Time)
        
        # 解析之后返回NaT的话就把现在的时间戳填充进去
        return converted.fillna(pd.Timestamp.now())

    def create_temporal_features(self, df):
        """小时、星期、周末、月份、时间线"""
        # copy()创建深拷贝，不修改原始的DataFrame
        result = df.copy()

        # timestamp从字符串转换为datatime对象 xxxx-xx-xx xx:xx:xx
        result["timestamp"] = self._datetime(result["timestamp"]) #把等号右边处理过的值赋值给等号左边，覆盖掉旧的timestamp列

        # 提取小时（0-23）
        result["hour_of_day"] = result["timestamp"].dt.hour 
        # 等号右边的dt.hour是把timestamp列的每个元素都取出小时部分，返回一个Series对象
        # 等号左边是把这个Series,作为新的一列塞进result这个DataFrame里
        # .dt 是 pandas 专门给 datetime 类型列准备的"工具包",后面接不同属性,取出时间的不同部分

        # 提取周几
        result["day_of_week"] = result["timestamp"].dt.dayofweek
        # 周一是0，周二是1，···，周日是6

        # 判断是否周末
        result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
        # result["day_of_week"] >= 5是一个布尔型Series
        # astype(int)是把布尔型Series转成整数型Series，True变成1，False变成0

        # 提取月份 (1-12)
        result["month"] = result["timestamp"].dt.month

        # -1~6 night; 6~12 morning; 12~18 afternoon; 18~23 evening
        result["day_period"] = pd.cut(
            result["hour_of_day"],
            bins=[-1, 6, 12, 18, 23], # 分切点
            labels=["night", "morning", "afternoon", "evening"] # 每一段起的名字
        )
        # pd.cut: 把一列连续的数字,按照你指定的分界点,切成几段区间,每段贴一个标签

        # 最新时间
        latest = result["timestamp"].max()

        # 传播天数
        result["recency_days"] = (latest - result["timestamp"]).dt.total_seconds() / (60 * 60 * 24)
        # 算出"每一条记录,距离数据集里最新那条记录,过去了多少天",存成新的一列 recency_days

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





def main():  #python中不知道写什么就写pass，不会中断程序运行
    #读取数据
    users = pd.read_csv("data/users.csv") #read_csv中有一个encoding参数，涉及到中文文本格式的话编码格式encoding就要变成utf-8
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