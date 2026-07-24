from pathlib import Path
import logging
import pickle #可以保存程序里的变量、列表、字典、模型等，下次直接load就行，不用重新训练模型，对ml尤其好用
import numpy as np
import pandas as pd
from sklearn.preprocessing import LableEncoder, OneHotEncoder, MinMaxScaler
#LabelEncoder：把类别标签转成整数编号，通常用于目标变量y
#OneHotEncoder：把特征X转成0/1，如性别，男性0，女性1
#MinMaxScaler：把数值型特征缩放到0-1之间，通常用于特征X

#logging：创建一个info级别及以上的日志，按照时间、日志级别、日志信息的格式输出到终端
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(levelname)s-%(message)s"
)

logger = logging.getLogger(__name__) #创建一个日志小助手

"""关于class和main函数的结构上课时问一下老师"""
class FeatureEngineer:
    def __init__(self):
        '''
        初始化特征工程，以下三个都是字典，key是列名，value是对应的处理器
        创建存储字典，用于保存后续训练阶段需要复用的对象。这些对象在训练时fit（拟合），推理时只load不fit。
        '''
        # scalers字典：保存MinMaxScaler对象
        # 键如"watch_time"表示对watch_time列的缩放器
        # 值是已经fit好的MinMaxScaler实例
        self.scalers = {}

        # encoders字典：保存类别编码器
        # 包含LabelEncoder（ID编码）和OneHotEncoder（类别编码）
        self.encoders = {}

        # embeddings字典：保存向量表示
        # 包含SVD模型和用户/视频的低维向量
        self.embeddings = {}
    
    @staticmethod 
    #静态方法(static method)= 写在 class 里面、但不需要用到 object 自身数据的函数, 是一个放在class里的普通函数
    #先判断:这个函数需不需要用到某个 object 的数据(self.xxx)?不需要 → 它就该设计成静态方法 → 于是不写 self + 加 @staticmethod
    #self可以理解成class创建出的object，静态方法和class创建出的object无关，这里的其他函数和self有关
    #该静态方法只能在这个class内部使用
    def _datetime(series):
        # 解析失败返回NaT
        converted = pd.to_datetime(series, errors="coerce") 
        #series指一维向量，csv里的一列，如join_date; string指的是一列里的每个字段，如2020-01-01 00:00:00.000000000
        #把series里的string列转成series里的datetime类型，converted还是series
        #遇到解析不了的(比如2月30日),不报错,而是变成 NaT(Not a Time)
        
        
        # 把NaT用现在的时间戳填充进去
        return converted.fillna(pd.Timestamp.now())
        # pd.Timestamp是使用pandas这个工具箱里面Timestamp这个class，他的作用是生成具体的时间点
        # .now是给具体的时间点为现在

    def create_temporal_features(self, df):
        """小时、星期、周末、月份、时间线"""
        # copy()创建深拷贝，不修改原始的DataFrame
        result = df.copy()

        # 调用_datetime函数，timestamp从字符串转换为datatime对象 xxxx-xx-xx xx:xx:xx
        result["timestamp"] = self._datetime(result["timestamp"])
         #处理interactioncsv，把等号右边处理过的值赋值给等号左边，覆盖掉旧的timestamp列

        # 提取小时（0-23）
        result["hour_of_day"] = result["timestamp"].dt.hour 
        # 等号右边的dt.hour是取出小时，返回一个Series对象,创建新的列
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
        # 函数里的参数是interactions_df和users_df，到时候调用函数的时候分别传入interactions_clean.csv和users.csv
        interactions = interactions_df.copy()
        interactions["timestamp"] = self._datetime(interactions["timestamp"])

        #groupby
        '''
        原始的 interactions 表是"每一次交互一行"(某用户在某时刻看了某视频、点没点赞、看了多久)。
        我们想把它变成"每个用户一行"的用户画像表(这个用户平均看多久、总共点了多少赞、第一次和最后一次交互是什么时候)
        '''
        stats = interactions.groupby("user_id").agg(
        # .groupby("user_id"): 把整张表按 user_id 的值分组,不会有可见输出,只是表达分组的意图
        # .agg(): 对每个分组,做聚合计算，结构如 新列名 = ("原列名", "聚合函数")

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
        ).reset_index() 
        # agg完了以后，df的索引是user_id，reset_index()把user_id变成一列，索引变成默认的0,1,2,3...

        # 用户活跃天数
        stats["user_tenure_days"] = (
            stats["last_seen"] - stats["first_seen"]
        ).dt.total_seconds() / (60 * 60 * 24)

        # 平均每天的交互次数
        stats["interactions_per_day"] = (
            stats["watch_time"] / stats["user_tenure_days"]
        )

        result = users_df.merge(stats, on = "user_id", how = "left")

        numeric = result.select_dtypes(include = np.nan).columns
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