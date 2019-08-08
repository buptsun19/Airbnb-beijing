# -*- coding: utf-8 -*-

"""
    作者：孙建书
    版本：1.0
    日期：2019.7.20
    功能：Airbnb数据分析工具函数集合
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决保存图像是负号'-'显示为方块的问题

filepath = './dataset/listings.csv'
listing_data = pd.read_csv(filepath)

print("listing列表的基本信息如下：")
listing_data.info()
print("\n")


print("预览列表的前三行数据如下：")
print(listing_data.head(3))
print("\n")

print("基本的统计信息如下：")
print(listing_data.describe())
print("\n")

print("每列的缺失信息统计如下：")
print(listing_data.isnull().any())


# 从以下代码可以判断，reviews_per_month字段以及last_review字段为NA的时候，number_of_reviews均为0，说明此条信息无任何评论
null_data = listing_data[listing_data['reviews_per_month'].isnull()]
print("reviews_per_month为0时，number_of_reviews的总和为：", null_data['number_of_reviews'].sum())
print("reviews_per_month为0时，last_review的总和为：", null_data['last_review'].sum())
print(null_data['last_review'].isnull().sum())

# 从以下代码可以判断，id是唯一的值
id_data_only = listing_data.drop_duplicates(subset=['id'])
print("id去重后的总计数为:", id_data_only['id'].count())


# 房屋在各个区县的分布比例
listing_data['区县'] = listing_data['neighbourhood'].str[0:3]
district_info = listing_data.groupby('区县').size()
district_info.sort_values(inplace=True, ascending=False)
district_info.name = '区县分布'
district_info.plot(kind='pie', autopct='%.1f%%', figsize=(8, 8))
plt.tight_layout()
plt.show()

# 北京市不同房型的数量
room_type_info = listing_data.groupby('room_type').size()
room_type_info.plot(kind='bar', figsize=(8, 8), color='r', rot=0)
plt.xlabel('房间型号')
plt.ylabel('房间数量')
plt.title('各种房型分布')
plt.tight_layout()
plt.show()

# 价格分布
sns.stripplot(x='price', data=listing_data)
plt.tight_layout()
plt.show()

# 价格和房型之间的关系
sns.stripplot(x='price', y='room_type', data=listing_data)
plt.tight_layout()
plt.show()

# 不同房型的价格分布
price_below_x = listing_data[listing_data['price'] <= 1000]
sns.boxplot(x='room_type', y='price', data=price_below_x)
plt.tight_layout()
plt.show()

sns.jointplot(x='reviews_per_month', y='number_of_reviews', data=listing_data, kind='scatter')
plt.tight_layout()
plt.show()

pairplot_data = listing_data.dropna(subset=['reviews_per_month'])
sns.pairplot(pairplot_data, hue='room_type', vars=['reviews_per_month', 'number_of_reviews'], kind='reg')
plt.tight_layout()
plt.show()
