from pathlib import Path #找文件用的
import sys # 改python的设置
import logging #日志工具

import pandas as pd
import pandera as pa #数据校验专用库

from pandera import Column, Check, DataFrameSchema #DataFrameSchema是 pandera 提供的类,用来定义一整张表的规则

#配置日志
logging.basicConfig( #全局日志，basicConfig — 给日志系统做全局设置
    level = logging.INFO,
    format="%(asctime)s-%(levelname)s-%(message)s"
    #YYYY-MM-DD HH:MM:SS
)

logger = logging.getLogger(__name__) # 拿一个日志小助手来用

USERS_SCHEMA = DataFrameSchema({ #用DataFrameSchema给users写一份规则书
    #{}this is dictionary, key是列名，value是列的规则
    "user_id": Column( #Column(...) 是 pandera 提供的类,专门描述user_id这一列的规则
        str, # user_id这一列的数据类型是str
        checks=[ #[]is a list,里面可以放很多个Check对象,每个Check对象都是一个规则
            Check( #规则1
                lambda s: s.str.startswith("user_"), #是一个匿名函数，s是一个输入的Series对象，s.str.startswith("user_")是返回值，返回一个布尔型Series，表示每个元素是否以"user_"开头
                error="user_id 必须以 user_开头" #error是遇到false的情况报错信息里自己写的解释
            ),
            Check( #规则2
                lambda s: s.nunique() == len(s), #s.nunique()这一列里有多少个不重复的值(number of unique)，和len比较，输出bool
                error="user_id 不能重复"
            )
        ]
    ),
    "age": Column(
        int,
        checks=[
            Check(
                lambda s: s>= 13,
                error="age不能小于13"
            ),
            Check(
                lambda s: s<100,
                error="age不能大于100"
            )
        ]

    ),
    "gender": Column(
        str,
        checks=Check(lambda s: s.isin(["M", "F", "Other"]))
        #checks=Check.isin(["M", "F", "Other"])自己继续写
    ),
    "preferred_category": 

})

VIDEOS_SCHEMA = DataFrameSchema({

INTERACTIONS_SCHEMA = DataFrameSchema({





#主校验函数
def validate_data(data_dir="data"):

    data_path = Path(data_dir)

    try:
        #开始记录日志
        logger.info("开始加载数据") # info级别的日志输出

        #读取三张表格
        users_df = pd.read_csv(data_path / "users.csv"). # /在这里是拼接路径的意思
        videos_df = pd.read_csv(data_path / "videos.csv")
        interactions_df = pd.read_csv(data_path / "interactions.csv")

        interactions_df["timestamp"] = pd.to_datetime( #CSV 里的时间(比如 "2026-01-15 10:30:00")读进来默认是字符串,不能做时间比较、排序、计算
            interactions_df["timestamp"],
            errors = "raise" #日期校验，时间匹配规则
        )
        #errors="raise"表示:如果有任何一行转换失败(比如格式乱了),直接抛错。其他选项:"coerce"(转不了就变 NaT)、"ignore"(转不了保留原样)。

        USERS_SCHEMA.validate(users_df)
        VIDEOS_SCHEMA.validate(videos_df)
        INTERACTIONS_SCHEMA.validate(interactions_df)
        logger.info("结构校验通过")
        #任何一个不通过 → 抛 SchemaError → 跳到 except 块。三个都通过 → 打印"结构校验通过"

        #检测缺失值和重复行
        for name, frame in [    #tuple写法：列表里每个元素是一个元组,每次循环,name拿到str,frame拿到对应的DataFrame
            ("users", users_df),
            ("videos", videos_df),
            ("interactions", interactions_df)
        ]:
        
            missing = frame.isna().sum().sum() #整张表一共有多少个缺失值
            duplicates = frame.duplicated().sum()#整张表一共有多少行重复行

            logger.info(
                "%s:缺失值=%d,重复行=%d",
                name, missing, duplicates
            )

        logger.info(
            "点赞率=%.4f, 分享率=%.4f, 平均观看=%.2f", # %.4f先保留四位小数再加百分号
            #把下面的三个参数依次传给上面这个字符串里的三个占位符
            interactions_df["liked"].mean(), # 参数1
            interactions_df["shared"].mean(), # 参数2
            interactions_df["watch_time"].mean() # 参数3
        )

    except (
        FileNotFoundError, # 文件不存在
        pd.errors.SchemaError, # 数据不符合Schema要求
        ValueError, 
        TypeError) as exc:
        
        


#python数据类型：str,int,float,bool
#pandas一维数组series，二维DataFrame
#df.dropna() 一行中只要有一个nan就把整行删掉
#df.dropna("all") 一整行都是缺失值把这行删掉
#df.dropna(subset=["col1","col2"]) 只要这两列都是缺失值就删掉
#df.dropna(axis=1) 删除整列
#df.dropna(axis=0) 删除整行

#df.fillna()保留所有行，把缺失的地方补上合理的值，数据量不变

#删除重复行 df.drop_duplicates()