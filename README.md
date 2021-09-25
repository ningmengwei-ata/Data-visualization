## 数据可视化 实验报告

本次项目的主题是世界幸福指数以及相关因素的分析。通过从心理健康、自由、安全、经济等不同维度分析各国的现状、国家之间的差异以及它们与幸福指数的相关性。

数据来源kaggle的五个数据集

世界幸福指数：https://www.kaggle.com/unsdsn/world-happiness

自杀率：https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016

全球恐怖袭击：https://www.kaggle.com/START-UMD/gtd

https://www.kesci.com/home/dataset/5a436b2849b4287ef7b358e1

自由指数：https://www.kaggle.com/gsutters/the-human-freedom-index

自杀数：https://www.kaggle.com/szamil/who-suicide-statistics

#### 1.主要任务介绍

项目的主要任务分为数据处理、可视化方法的选择、可视化结果的形成以及分析等。

首先，导入相关常用的库。

```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure, show
```

#### 2.可视化设计

##### 2.1可视化设计原则

对于不同的数据特征采用不同的可视化图像来刻画。尽可能突出数据的特点,描述清晰明了。

##### 2.2颜色设计

在颜色设计方面，主要使用cmap、col = plt.cm.Spectral()自定义颜色和根据数据分布范围自定义颜色三种方式。cmap和根据数据自定义如下图所示：

<img src="/Users/wangwenqing/Library/Application Support/typora-user-images/image-20200729215919551.png" alt="image-20200729215919551" style="zoom:30%;" />

<img src="/Users/wangwenqing/Library/Application Support/typora-user-images/image-20200729215936218.png" alt="image-20200729215936218" style="zoom:30%;" />

<img src="/Users/wangwenqing/Library/Application Support/typora-user-images/image-20200729215946311.png" alt="image-20200729215946311" style="zoom:30%;" />

##### 2.3图的使用

在图的使用方面，基于python的matplotlib库、seaborn库、plotly库、Basemap等

热图(heatmap)初步对数据集各个变量的关系的认识

基于plotly的交互彩色地图，如下图所示

<img src="/Users/wangwenqing/Library/Application Support/typora-user-images/image-20200729220219561.png" alt="image-20200729220219561" style="zoom:50%;" />

柱状图(Barplot) 应用于排序以及不同年龄、性别等的分布情况

散点图(scatter)描绘两个变量之间的相关关系

##### 2.4可视化设计的亮点

使用动画展现多年中全球恐怖袭击事件的变化

#### 3.可视化分析

##### 3.1 全球心理健康角度分析

读取数据


```python
suicide=pd.read_csv('who_suicide_statistics.csv')
suicide.head()
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>year</th>
      <th>sex</th>
      <th>age</th>
      <th>suicides_no</th>
      <th>population</th>
      <th>Unnamed: 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Albania</td>
      <td>1985</td>
      <td>female</td>
      <td>15-24 years</td>
      <td>NaN</td>
      <td>277900.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Albania</td>
      <td>1985</td>
      <td>female</td>
      <td>25-34 years</td>
      <td>NaN</td>
      <td>246800.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Albania</td>
      <td>1985</td>
      <td>female</td>
      <td>35-54 years</td>
      <td>NaN</td>
      <td>267500.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Albania</td>
      <td>1985</td>
      <td>female</td>
      <td>5-14 years</td>
      <td>NaN</td>
      <td>298300.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Albania</td>
      <td>1985</td>
      <td>female</td>
      <td>55-74 years</td>
      <td>NaN</td>
      <td>138700.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

**3.1.1 全球心理健康分析**

整体呈上身趋势在1982-1986年，自杀人数的波动较大。

在2004年以后，自杀人数有小幅下降。

男性自杀人数约为女性的三倍。


```python
plt.figure(figsize=(8, 6))

#选出year列并且去除重复的年份
year = suicide.groupby('year').year.unique()

#各个年份的自杀人数汇总
#使用seaborn进行可视化,输入的数据必须为dataframe
totalpyear = pd.DataFrame(suicide.groupby('year').suicides_no.sum())   

plt.plot(year.index[0:36], totalpyear[0:36], '-o',color=col[18])  #选取范围为[0:31]  1985年到2015年
plt.xlabel('year', fontsize=20)
plt.ylabel('Total number of suicides in the world', fontsize=20)
plt.show()

```


![png](output_5_0.png)

```python
data= suicide.groupby(by=['sex']).agg({"suicides_no": ['sum']})
data.columns = ['total_suicide']
data.reset_index(inplace=True)
wedges, texts, autotexts = ax1.pie(data['total_suicide'], labels=data['sex'], autopct='%1.1f%%', startangle=90, colors=['#FF99CC','#99CCFF'])

plt.setp(autotexts, size=15)

plt.show()
```


![png](output_6_0.png)

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>sex</th>
      <th>age</th>
      <th>suicides_no</th>
      <th>population</th>
      <th>suicides/100k pop</th>
      <th>country-year</th>
      <th>HDI for year</th>
      <th>gdp_for_year ($)</th>
      <th>gdp_per_capita ($)</th>
      <th>generation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Albania</td>
      <td>1987</td>
      <td>1</td>
      <td>1</td>
      <td>21</td>
      <td>312900</td>
      <td>6.71</td>
      <td>Albania1987</td>
      <td>NaN</td>
      <td>2,156,624,900</td>
      <td>796</td>
      <td>Generation X</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Albania</td>
      <td>1987</td>
      <td>1</td>
      <td>3</td>
      <td>16</td>
      <td>308000</td>
      <td>5.19</td>
      <td>Albania1987</td>
      <td>NaN</td>
      <td>2,156,624,900</td>
      <td>796</td>
      <td>Silent</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Albania</td>
      <td>1987</td>
      <td>0</td>
      <td>1</td>
      <td>14</td>
      <td>289700</td>
      <td>4.83</td>
      <td>Albania1987</td>
      <td>NaN</td>
      <td>2,156,624,900</td>
      <td>796</td>
      <td>Generation X</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Albania</td>
      <td>1987</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>21800</td>
      <td>4.59</td>
      <td>Albania1987</td>
      <td>NaN</td>
      <td>2,156,624,900</td>
      <td>796</td>
      <td>G.I. Generation</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Albania</td>
      <td>1987</td>
      <td>1</td>
      <td>2</td>
      <td>9</td>
      <td>274300</td>
      <td>3.28</td>
      <td>Albania1987</td>
      <td>NaN</td>
      <td>2,156,624,900</td>
      <td>796</td>
      <td>Boomers</td>
    </tr>
  </tbody>
</table>
</div>


```python
suicide.head()
```

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>age</th>
      <th>suicides_no</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1979</td>
      <td>35-54 years</td>
      <td>28614.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1979</td>
      <td>55-74 years</td>
      <td>23270.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1979</td>
      <td>25-34 years</td>
      <td>17149.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1979</td>
      <td>15-24 years</td>
      <td>14701.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1979</td>
      <td>75+ years</td>
      <td>8421.0</td>
    </tr>
  </tbody>
</table>
</div>

**全球自杀数最多的十个国家**，代码和结果如下所示：


```python
suicide=pd.read_csv('who_suicide_statistics.csv')
suicide=suicide.groupby(['year','country']).suicides_no.sum().reset_index()
suicide.head()
```


```python
col = plt.cm.Spectral(np.linspace(0, 1, 20))
plt.figure(figsize=(10, 10))

plt.subplot(111)
#自杀人数的平均值最高的前10个国家
suicide.groupby(['country']).suicides_no.mean().nlargest(10).plot(kind='barh', color=col)
plt.xlabel('Average Suicides number', size=20)
plt.ylabel('Country', fontsize=20)
plt.tick_params(labelsize=18) 
plt.title('Top 10  countries with highest suicide number', fontsize=30)

```


![png](output_19_1.png)

从图中可以看到，自杀数最多的国家是俄罗斯、美国、日本。

平均自杀人数较多的国家以发达国家居多。

**各年龄段人群的具体分布**如下图所示：


![png](output_21_0.png)

**自杀人数的各个年龄段的性别分布**


```python
suicide['age'] = suicide.age.astype(pd.api.types.CategoricalDtype(categories = ['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years']))
ax=sns.barplot(x='sex', y='suicides_no', hue='age', data=suicide, palette="coolwarm")
ax.tick_params(labelsize=15) 

#plt.title("")
plt.legend(fontsize=16,loc="best")

```


![png](output_23_1.png)

从图中可以看到，男性自杀人数在整体上远高于女性且35-54岁的男性自杀人数远高于其余年龄段的男性

```python
suicide['age'] = suicide.age.astype(pd.api.types.CategoricalDtype(categories = ['35-54 years','55-74 years','25-34 years','15-24 years','75+ years','5-14 years']))
suicide.head()
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>sex</th>
      <th>age</th>
      <th>suicides_no</th>
      <th>population</th>
      <th>Unnamed: 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Albania</td>
      <td>1985</td>
      <td>female</td>
      <td>15-24 years</td>
      <td>NaN</td>
      <td>277900.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Albania</td>
      <td>1985</td>
      <td>female</td>
      <td>25-34 years</td>
      <td>NaN</td>
      <td>246800.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Albania</td>
      <td>1985</td>
      <td>female</td>
      <td>35-54 years</td>
      <td>NaN</td>
      <td>267500.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Albania</td>
      <td>1985</td>
      <td>female</td>
      <td>5-14 years</td>
      <td>NaN</td>
      <td>298300.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Albania</td>
      <td>1985</td>
      <td>female</td>
      <td>55-74 years</td>
      <td>NaN</td>
      <td>138700.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

**自杀数随时间各年龄层分布**如下；

这里我们设置图例将图例中的年龄和线条的上下顺序对应起来，方便阅读。


```python
from matplotlib.colors import LogNorm


suicide['age'] = suicide.age.astype(pd.api.types.CategoricalDtype(categories = ['35-54 years','55-74 years','25-34 years','15-24 years','75+ years','5-14 years']))
suicide.head()
suicide =suicide.groupby(['year','age']).suicides_no.sum().reset_index()
sns.lineplot('year','suicides_no',hue='age',style='age',data=suicide,hue_norm=LogNorm(),palette="Dark2",lw=3,sort=False)
plt.show()
```


![png](output_26_0.png)

**自杀率分析**


![png](output_28_0.png)

从图中可以看出：自杀率都呈下降趋势。其中75+自杀比率最高 55-74岁自杀率其次。5-14岁以及15-24岁的自杀比率较低。


```python
plt.figure(figsize=(12, 8))
sns.lineplot(x='year', y='suicides/100k pop', hue='sex', data=rate, palette="Set1")  #hue按年龄分组
plt.xticks(ha='right', fontsize=20)
plt.ylabel('suicidesper100k', fontsize=20)
plt.xlabel('year', fontsize=20)
plt.legend(fontsize=14, loc='best')  
plt.title("the relation between sex,year and suicide rate")
plt.show()

```


![png](output_30_0.png)

从图中可以看出：女性自杀率呈缓慢下降趋势。男性自杀率在1985-1995呈上升趋势。在1995-2015下降趋势而在2016年又出现一个陡然上升。

**3.1.2心理健康与经济的关系**


```python
from sklearn import preprocessing
plt.figure(figsize=(15, 10))
plt.plot(suicide_mean_scaled['year'],suicide_mean_scaled['gdp_per_capita ($)'], 'r--',label='gdp_per_capita ($)')
plt.plot(suicide_mean_scaled['year'], suicide_mean_scaled['suicides/100k pop'], 'b--',label='suicides per 100k population')

plt.plot(suicide_mean_scaled['year'],suicide_mean_scaled['gdp_per_capita ($)'],'ro-',suicide_mean_scaled['year'],suicide_mean_scaled['suicides/100k pop'],'go-')
plt.legend(loc='best',fontsize=22)
plt.xticks(fontsize='20')
plt.yticks(fontsize='20')
plt.xlabel('year',fontsize='25')
plt.ylabel('per 100k population',fontsize='25')
#plt.legend(handles=[l1,l2],labels=['gdp_per_capita ($)','suicides per 100k population'],loc='best')
plt.figtext(.5,.91,' suicide per 100k population chart with GDP (1980 - 2016)',fontsize=25, ha='center')
plt.show()
```


![png](output_33_0.png)

全球的GDP不断上涨，1985年到2000年缓慢上涨，2000年以后经济增长的势头猛增；受到金融危机的影响，2008年经济下滑后又开始增长。

平均自杀率在1986-1994年以小幅增长为主，1995年出现猛增后直至2015年呈下降趋势，2016年又一次猛增。


```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(rate['gdp_per_capita ($)'],rate['year'],rate['suicides/100k pop'],  alpha=0.25, c="#339966", edgecolors='black',s=50, label="both sex") 
#plt.title('')
plt.xticks(fontsize='12')
plt.yticks(fontsize='12')
ax.set_xlabel('GDP',fontsize=18,rotation=-5)
ax.set_ylabel('year',fontsize=18,rotation=35)
ax.set_zlabel('suicide per 100k population',fontsize=18)
plt.legend(fontsize=18,loc=2)

plt.show()
```


![png](output_35_0.png)

从图中可以看到GDP逐年增长，每一年平均自杀率在较小区间(0-50 per 100k pop)

的比例最高，经济较不发达的地区的自杀率高于GDP较高的地区。

**3.1.3 心理健康与幸福的关系**

通过合并数据库分析出自杀率和幸福指数的相关性，代码和图如下。


```python
temps=pd.read_csv('who_suicide_statistics.csv')
temps=temps.groupby(['year','Country']).population.sum().reset_index()
temps=temps[temps['year']==2010]
temps.head()
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>Country</th>
      <th>population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2986</td>
      <td>2010</td>
      <td>Albania</td>
      <td>2736025.0</td>
    </tr>
    <tr>
      <td>2987</td>
      <td>2010</td>
      <td>Anguilla</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2988</td>
      <td>2010</td>
      <td>Argentina</td>
      <td>37578454.0</td>
    </tr>
    <tr>
      <td>2989</td>
      <td>2010</td>
      <td>Armenia</td>
      <td>2676225.0</td>
    </tr>
    <tr>
      <td>2990</td>
      <td>2010</td>
      <td>Aruba</td>
      <td>95006.0</td>
    </tr>
  </tbody>
</table>

</div>


```python
#rate=pd.read_csv('master 5.csv')
rate=rate.groupby(['year','Country']).suicides_100k.sum().reset_index()
rate=rate[rate['year']==2010]
rate.head()
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>Country</th>
      <th>suicides_100k</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1830</td>
      <td>2010</td>
      <td>Albania</td>
      <td>3.471667</td>
    </tr>
    <tr>
      <td>1831</td>
      <td>2010</td>
      <td>Argentina</td>
      <td>9.270000</td>
    </tr>
    <tr>
      <td>1832</td>
      <td>2010</td>
      <td>Armenia</td>
      <td>3.367500</td>
    </tr>
    <tr>
      <td>1833</td>
      <td>2010</td>
      <td>Aruba</td>
      <td>4.887500</td>
    </tr>
    <tr>
      <td>1834</td>
      <td>2010</td>
      <td>Australia</td>
      <td>11.053333</td>
    </tr>
  </tbody>
</table>

</div>




```python
suicide=pd.read_csv('who_suicide_statistics.csv')
suicide=suicide.groupby(['year','Country']).suicides_no.sum().reset_index()
suicide=suicide[suicide['year']==2010]
suicide.merge(temps)
suicide.head()
```

```python
temp.head()
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>Region</th>
      <th>Happiness Rank</th>
      <th>Happiness Score</th>
      <th>Standard Error</th>
      <th>Economy (GDP per Capita)</th>
      <th>Family</th>
      <th>Health (Life Expectancy)</th>
      <th>Freedom</th>
      <th>Trust (Government Corruption)</th>
      <th>Generosity</th>
      <th>Dystopia Residual</th>
      <th>year</th>
      <th>suicides_no</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Switzerland</td>
      <td>Western Europe</td>
      <td>1</td>
      <td>7.587</td>
      <td>0.03411</td>
      <td>1.39651</td>
      <td>1.34951</td>
      <td>0.94143</td>
      <td>0.66557</td>
      <td>0.41978</td>
      <td>0.29678</td>
      <td>2.51738</td>
      <td>2015</td>
      <td>1073.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Iceland</td>
      <td>Western Europe</td>
      <td>2</td>
      <td>7.561</td>
      <td>0.04884</td>
      <td>1.30232</td>
      <td>1.40223</td>
      <td>0.94784</td>
      <td>0.62877</td>
      <td>0.14145</td>
      <td>0.43630</td>
      <td>2.70201</td>
      <td>2015</td>
      <td>40.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Denmark</td>
      <td>Western Europe</td>
      <td>3</td>
      <td>7.527</td>
      <td>0.03328</td>
      <td>1.32548</td>
      <td>1.36058</td>
      <td>0.87464</td>
      <td>0.64938</td>
      <td>0.48357</td>
      <td>0.34139</td>
      <td>2.49204</td>
      <td>2015</td>
      <td>564.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Norway</td>
      <td>Western Europe</td>
      <td>4</td>
      <td>7.522</td>
      <td>0.03880</td>
      <td>1.45900</td>
      <td>1.33095</td>
      <td>0.88521</td>
      <td>0.66973</td>
      <td>0.36503</td>
      <td>0.34699</td>
      <td>2.46531</td>
      <td>2015</td>
      <td>590.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Finland</td>
      <td>Western Europe</td>
      <td>6</td>
      <td>7.406</td>
      <td>0.03140</td>
      <td>1.29025</td>
      <td>1.31826</td>
      <td>0.88911</td>
      <td>0.64169</td>
      <td>0.41372</td>
      <td>0.23351</td>
      <td>2.61955</td>
      <td>2015</td>
      <td>731.0</td>
    </tr>
  </tbody>
</table>

</div>


```python
happiness_report = pd.read_csv('2016.csv')
plt.figure(figsize=(15, 15))
#happiness_report = happiness_report[['country', 'Happiness Score']]
temp = happiness_report.merge(rate)
sns.scatterplot( x=temp['Happiness Score'],y=temp['suicides_100k'], hue=temp['Region'],s=300)
```


![png](output_15_1.png)

从上图可以看出心理健康与幸福指数的相关性较低。从西欧地区散点图可以看出大致有自杀率较高 幸福指数较低 。自杀数据特别高的国家的幸福指数不会特别高，如图中100k自杀率为35的点。

##### 3.2 安全角度分析

世界恐怖袭击概况


```python
print(regions)
```

    ['South Asia', 'South America', 'Central America & Caribbean', 'Southeast Asia', 'North America', 'Eastern Europe', 'Central Asia', 'Australasia & Oceania', 'Middle East & North Africa', 'East Asia', 'Sub-Saharan Africa', 'Western Europe']


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import HTML
terror=pd.read_csv('globalterrorism.csv',encoding='ISO-8859-1')
terror.head()
```

    Average number of people killed per attack by Region
    
    South Asia:2.3
    South America:1.67
    Central America & Caribbean:3.58
    Southeast Asia:1.28
    North America:1.46
    Eastern Europe:1.49
    Central Asia:1.79
    Australasia & Oceania:0.54
    Middle East & North Africa:2.86
    East Asia:1.51
    Sub-Saharan Africa:4.92
    Western Europe:0.43

**3.2.1全球恐怖袭击概况**

```python
plt.subplots(figsize=(15,6))
sns.countplot('iyear',data=terror,palette='RdYlGn_r')
plt.xticks(rotation=90)
plt.xlabel('year')
plt.title('Terrorist Activities Each Year')
plt.show()
```


![png](output_51_0.png)

从图中可以看到，全球恐怖袭击活动在1970-1977年上升缓慢。1992-1998年有小幅下降。在2011-2014年猛增，2015-2017年有小幅下降。

```python
terror_region=pd.crosstab(terror.iyear,terror.region_txt)
terror_region.plot(color=sns.color_palette('Dark2'))
fig=plt.gcf()
plt.xlabel('year')
plt.ylabel('terrorist attack in the region')
plt.show()
```


![png](output_52_0.png)

图中可以看到，1984-1992 南美恐怖袭击最多。地区恐怖袭击活动在中东&北非地区最盛、南亚其次。在2005-2010年增长迅猛。恐怖袭击活动较少的地区主要有大洋洲和东亚。


```python
plt.subplots(figsize=(12,6))
sns.countplot('region_txt',data=terror,palette='RdYlBu',order=terror['region_txt'].value_counts().index)
plt.xticks(rotation=60)

plt.xlabel('region')
plt.show()
```


![png](output_55_0.png)

从图中可以看到，全球恐怖袭击活动在1970-1977年上升缓慢。1992-1998年有小幅下降。在2011-2014年猛增。2015-2017年有小幅下降

![image-20200730172748717](/Users/wangwenqing/Library/Application Support/typora-user-images/image-20200730172748717.png)

![image-20200730172839033](/Users/wangwenqing/Library/Application Support/typora-user-images/image-20200730172839033.png)

从动画中我们可以清晰的看到恐怖袭击活动的扩散。

印度、撒哈拉沙漠地区、非洲北部和南部地区、中美洲国家、秘鲁、哥伦比亚、中东地区恐怖袭击活动较密。

**3.2.3从自由角度分析**


```python
free18=pd.read_csv('hfi_cc_2018.csv')
free19=pd.read_csv('hfi_cc_2019.csv')
```


```python
free18.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>ISO_code</th>
      <th>country</th>
      <th>region</th>
      <th>pf_rol_procedural</th>
      <th>pf_rol_civil</th>
      <th>pf_rol_criminal</th>
      <th>pf_rol</th>
      <th>pf_ss_homicide</th>
      <th>pf_ss_disappearances_disap</th>
      <th>...</th>
      <th>ef_regulation_business_bribes</th>
      <th>ef_regulation_business_licensing</th>
      <th>ef_regulation_business_compliance</th>
      <th>ef_regulation_business</th>
      <th>ef_regulation</th>
      <th>ef_score</th>
      <th>ef_rank</th>
      <th>hf_score</th>
      <th>hf_rank</th>
      <th>hf_quartile</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2016</td>
      <td>ALB</td>
      <td>Albania</td>
      <td>Eastern Europe</td>
      <td>6.661503</td>
      <td>4.547244</td>
      <td>4.666508</td>
      <td>5.291752</td>
      <td>8.920429</td>
      <td>10.0</td>
      <td>...</td>
      <td>4.050196</td>
      <td>7.324582</td>
      <td>7.074366</td>
      <td>6.705863</td>
      <td>6.906901</td>
      <td>7.54</td>
      <td>34.0</td>
      <td>7.568140</td>
      <td>48.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2016</td>
      <td>DZA</td>
      <td>Algeria</td>
      <td>Middle East &amp; North Africa</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.819566</td>
      <td>9.456254</td>
      <td>10.0</td>
      <td>...</td>
      <td>3.765515</td>
      <td>8.523503</td>
      <td>7.029528</td>
      <td>5.676956</td>
      <td>5.268992</td>
      <td>4.99</td>
      <td>159.0</td>
      <td>5.135886</td>
      <td>155.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2016</td>
      <td>AGO</td>
      <td>Angola</td>
      <td>Sub-Saharan Africa</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.451814</td>
      <td>8.060260</td>
      <td>5.0</td>
      <td>...</td>
      <td>1.945540</td>
      <td>8.096776</td>
      <td>6.782923</td>
      <td>4.930271</td>
      <td>5.518500</td>
      <td>5.17</td>
      <td>155.0</td>
      <td>5.640662</td>
      <td>142.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2016</td>
      <td>ARG</td>
      <td>Argentina</td>
      <td>Latin America &amp; the Caribbean</td>
      <td>7.098483</td>
      <td>5.791960</td>
      <td>4.343930</td>
      <td>5.744791</td>
      <td>7.622974</td>
      <td>10.0</td>
      <td>...</td>
      <td>3.260044</td>
      <td>5.253411</td>
      <td>6.508295</td>
      <td>5.535831</td>
      <td>5.369019</td>
      <td>4.84</td>
      <td>160.0</td>
      <td>6.469848</td>
      <td>107.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2016</td>
      <td>ARM</td>
      <td>Armenia</td>
      <td>Caucasus &amp; Central Asia</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.003205</td>
      <td>8.808750</td>
      <td>10.0</td>
      <td>...</td>
      <td>4.575152</td>
      <td>9.319612</td>
      <td>6.491481</td>
      <td>6.797530</td>
      <td>7.378069</td>
      <td>7.57</td>
      <td>29.0</td>
      <td>7.241402</td>
      <td>57.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 123 columns</p>
</div>


```python
free18=pd.read_csv('hfi_cc_2018.csv')
free18=free18[free18['year']==2016]
free18=free18[['Country','hf_score','pf_score']]
free18.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>hf_score</th>
      <th>pf_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Albania</td>
      <td>7.568140</td>
      <td>7.596281</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Algeria</td>
      <td>5.135886</td>
      <td>5.281772</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Angola</td>
      <td>5.640662</td>
      <td>6.111324</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Argentina</td>
      <td>6.469848</td>
      <td>8.099696</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Armenia</td>
      <td>7.241402</td>
      <td>6.912804</td>
    </tr>
  </tbody>
</table>

</div>




```python
happiness_report.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Region</th>
      <th>Happiness Rank</th>
      <th>Happiness Score</th>
      <th>Lower Confidence Interval</th>
      <th>Upper Confidence Interval</th>
      <th>Economy (GDP per Capita)</th>
      <th>Family</th>
      <th>Health (Life Expectancy)</th>
      <th>Freedom</th>
      <th>Trust (Government Corruption)</th>
      <th>Generosity</th>
      <th>Dystopia Residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Denmark</td>
      <td>Western Europe</td>
      <td>1</td>
      <td>7.526</td>
      <td>7.460</td>
      <td>7.592</td>
      <td>1.44178</td>
      <td>1.16374</td>
      <td>0.79504</td>
      <td>0.57941</td>
      <td>0.44453</td>
      <td>0.36171</td>
      <td>2.73939</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Switzerland</td>
      <td>Western Europe</td>
      <td>2</td>
      <td>7.509</td>
      <td>7.428</td>
      <td>7.590</td>
      <td>1.52733</td>
      <td>1.14524</td>
      <td>0.86303</td>
      <td>0.58557</td>
      <td>0.41203</td>
      <td>0.28083</td>
      <td>2.69463</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Iceland</td>
      <td>Western Europe</td>
      <td>3</td>
      <td>7.501</td>
      <td>7.333</td>
      <td>7.669</td>
      <td>1.42666</td>
      <td>1.18326</td>
      <td>0.86733</td>
      <td>0.56624</td>
      <td>0.14975</td>
      <td>0.47678</td>
      <td>2.83137</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Norway</td>
      <td>Western Europe</td>
      <td>4</td>
      <td>7.498</td>
      <td>7.421</td>
      <td>7.575</td>
      <td>1.57744</td>
      <td>1.12690</td>
      <td>0.79579</td>
      <td>0.59609</td>
      <td>0.35776</td>
      <td>0.37895</td>
      <td>2.66465</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Finland</td>
      <td>Western Europe</td>
      <td>5</td>
      <td>7.413</td>
      <td>7.351</td>
      <td>7.475</td>
      <td>1.40598</td>
      <td>1.13464</td>
      <td>0.81091</td>
      <td>0.57104</td>
      <td>0.41004</td>
      <td>0.25492</td>
      <td>2.82596</td>
    </tr>
  </tbody>
</table>

</div>


```python
happiness_report = pd.read_csv('2016.csv')
plt.figure(figsize=(18, 18))
#happiness_report = happiness_report[['country', 'Happiness Score']]
temp = happiness_report.merge(free18)

sns.scatterplot( x=temp['Happiness Score'],y=temp['hf_score'], hue=happy2016['Region'],s=300)
plt.ylabel('human_freedom',fontsize=30)
plt.xlabel('happiness score',fontsize=30)
plt.show()
```


![png](output_94_0.png)

可以看出国民自由与幸福指数的有一定相关性.撒哈拉沙漠地区普遍处于幸福指数与人类自由评分都较低的区域.西欧地区普遍处于幸福指数与人类自由评分都较高的区域.

```python
sns.scatterplot( x=temp['Happiness Score'],y=temp['pf_score'], hue=happy2020['Regional indicator'],s=300)

plt.ylabel('personal_freedom',fontsize=30)
plt.xlabel('happiness score',fontsize=30)
plt.show()
```

![png](output_95_0.png)

个人自由与幸福指数相关性比与国民自由相关性略高一点.部分西欧国家的幸福指数与个人自由评分更集中在一个很高的区域.撒哈拉沙漠地区个人自由较国民自由评分有小幅上升.

```python
free19.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>ISO_code</th>
      <th>countries</th>
      <th>region</th>
      <th>hf_score</th>
      <th>hf_rank</th>
      <th>hf_quartile</th>
      <th>pf_rol_procedural</th>
      <th>pf_rol_civil</th>
      <th>pf_rol_criminal</th>
      <th>...</th>
      <th>ef_regulation_business_adm</th>
      <th>ef_regulation_business_bureaucracy</th>
      <th>ef_regulation_business_start</th>
      <th>ef_regulation_business_bribes</th>
      <th>ef_regulation_business_licensing</th>
      <th>ef_regulation_business_compliance</th>
      <th>ef_regulation_business</th>
      <th>ef_regulation</th>
      <th>ef_score</th>
      <th>ef_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2017</td>
      <td>ALB</td>
      <td>Albania</td>
      <td>Eastern Europe</td>
      <td>7.84</td>
      <td>38</td>
      <td>1</td>
      <td>6.7</td>
      <td>4.5</td>
      <td>4.7</td>
      <td>...</td>
      <td>6.3</td>
      <td>6.7</td>
      <td>9.7</td>
      <td>4.1</td>
      <td>6</td>
      <td>7.2</td>
      <td>6.7</td>
      <td>7.8</td>
      <td>7.67</td>
      <td>30</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2017</td>
      <td>DZA</td>
      <td>Algeria</td>
      <td>Middle East &amp; North Africa</td>
      <td>4.99</td>
      <td>155</td>
      <td>4</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>3.7</td>
      <td>1.8</td>
      <td>9.3</td>
      <td>3.8</td>
      <td>8.7</td>
      <td>7</td>
      <td>5.7</td>
      <td>5.4</td>
      <td>4.77</td>
      <td>159</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2017</td>
      <td>AGO</td>
      <td>Angola</td>
      <td>Sub-Saharan Africa</td>
      <td>5.4</td>
      <td>151</td>
      <td>4</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>2.4</td>
      <td>1.3</td>
      <td>8.7</td>
      <td>1.9</td>
      <td>8.1</td>
      <td>6.8</td>
      <td>4.9</td>
      <td>5.7</td>
      <td>4.83</td>
      <td>158</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2017</td>
      <td>ARG</td>
      <td>Argentina</td>
      <td>Latin America &amp; the Caribbean</td>
      <td>6.86</td>
      <td>77</td>
      <td>2</td>
      <td>7.1</td>
      <td>5.8</td>
      <td>4.3</td>
      <td>...</td>
      <td>2.5</td>
      <td>7.1</td>
      <td>9.6</td>
      <td>3.3</td>
      <td>5.4</td>
      <td>6.5</td>
      <td>5.7</td>
      <td>5.6</td>
      <td>5.67</td>
      <td>147</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2017</td>
      <td>ARM</td>
      <td>Armenia</td>
      <td>Caucasus &amp; Central Asia</td>
      <td>7.42</td>
      <td>54</td>
      <td>2</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>...</td>
      <td>4.6</td>
      <td>6.2</td>
      <td>9.9</td>
      <td>4.6</td>
      <td>9.3</td>
      <td>7.1</td>
      <td>6.9</td>
      <td>7.5</td>
      <td>7.7</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 120 columns</p>
</div>


```python
plt.figure(figsize=(20,5))
sns.set(font_scale=1.5)
sns.boxplot(x='region',y='pf_score',data=free18,order=['Western Europe','North America', 'Oceania','Middle East & North Africa','Latin America & the Caribbean','Caucasus & Central Asia','Eastern Europe','East Asia', 'Sub-Saharan Africa','South Asia'],palette='rainbow');
plt.title("Personal Freedom ", fontsize=20)
plt.xlabel('')
plt.xticks(rotation=30)
plt.show()

```


![png](output_98_0.png)

##### 3.4 2020幸福评分分析


```python
happy2015=pd.read_csv('2015.csv')
happy2016=pd.read_csv('2016.csv')
happy2017=pd.read_csv('2017.csv')
happy2018=pd.read_csv('2018.csv')
happy2019=pd.read_csv('2019.csv')
happy2020=pd.read_csv('2020.csv')
```

**数据预处理**


```python
happy2020 = happy2020.drop(['Standard error of ladder score', 'upperwhisker', 'upperwhisker', 'lowerwhisker', 'Explained by: Log GDP per capita', 'Explained by: Social support', 'Explained by: Healthy life expectancy', 'Explained by: Freedom to make life choices', 'Explained by: Generosity', 'Explained by: Perceptions of corruption'], axis=1)
```


```python
import pandas as pd
CountryInfo=pd.read_csv('2020.csv',index_col=0)
RegionInfo=pd.read_excel('match.xlsx')
RegionInfoDict=RegionInfo.T.to_dict()
CountryInfo.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Regional indicator</th>
      <th>Ladder score</th>
      <th>Standard error of ladder score</th>
      <th>upperwhisker</th>
      <th>lowerwhisker</th>
      <th>Logged GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
      <th>Ladder score in Dystopia</th>
      <th>Explained by: Log GDP per capita</th>
      <th>Explained by: Social support</th>
      <th>Explained by: Healthy life expectancy</th>
      <th>Explained by: Freedom to make life choices</th>
      <th>Explained by: Generosity</th>
      <th>Explained by: Perceptions of corruption</th>
      <th>Dystopia + residual</th>
    </tr>
    <tr>
      <th>Country name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Finland</td>
      <td>Western Europe</td>
      <td>7.8087</td>
      <td>0.031156</td>
      <td>7.869766</td>
      <td>7.747634</td>
      <td>10.639267</td>
      <td>0.954330</td>
      <td>71.900825</td>
      <td>0.949172</td>
      <td>-0.059482</td>
      <td>0.195445</td>
      <td>1.972317</td>
      <td>1.285190</td>
      <td>1.499526</td>
      <td>0.961271</td>
      <td>0.662317</td>
      <td>0.159670</td>
      <td>0.477857</td>
      <td>2.762835</td>
    </tr>
    <tr>
      <td>Denmark</td>
      <td>Western Europe</td>
      <td>7.6456</td>
      <td>0.033492</td>
      <td>7.711245</td>
      <td>7.579955</td>
      <td>10.774001</td>
      <td>0.955991</td>
      <td>72.402504</td>
      <td>0.951444</td>
      <td>0.066202</td>
      <td>0.168489</td>
      <td>1.972317</td>
      <td>1.326949</td>
      <td>1.503449</td>
      <td>0.979333</td>
      <td>0.665040</td>
      <td>0.242793</td>
      <td>0.495260</td>
      <td>2.432741</td>
    </tr>
    <tr>
      <td>Switzerland</td>
      <td>Western Europe</td>
      <td>7.5599</td>
      <td>0.035014</td>
      <td>7.628528</td>
      <td>7.491272</td>
      <td>10.979933</td>
      <td>0.942847</td>
      <td>74.102448</td>
      <td>0.921337</td>
      <td>0.105911</td>
      <td>0.303728</td>
      <td>1.972317</td>
      <td>1.390774</td>
      <td>1.472403</td>
      <td>1.040533</td>
      <td>0.628954</td>
      <td>0.269056</td>
      <td>0.407946</td>
      <td>2.350267</td>
    </tr>
    <tr>
      <td>Iceland</td>
      <td>Western Europe</td>
      <td>7.5045</td>
      <td>0.059616</td>
      <td>7.621347</td>
      <td>7.387653</td>
      <td>10.772559</td>
      <td>0.974670</td>
      <td>73.000000</td>
      <td>0.948892</td>
      <td>0.246944</td>
      <td>0.711710</td>
      <td>1.972317</td>
      <td>1.326502</td>
      <td>1.547567</td>
      <td>1.000843</td>
      <td>0.661981</td>
      <td>0.362330</td>
      <td>0.144541</td>
      <td>2.460688</td>
    </tr>
    <tr>
      <td>Norway</td>
      <td>Western Europe</td>
      <td>7.4880</td>
      <td>0.034837</td>
      <td>7.556281</td>
      <td>7.419719</td>
      <td>11.087804</td>
      <td>0.952487</td>
      <td>73.200783</td>
      <td>0.955750</td>
      <td>0.134533</td>
      <td>0.263218</td>
      <td>1.972317</td>
      <td>1.424207</td>
      <td>1.495173</td>
      <td>1.008072</td>
      <td>0.670201</td>
      <td>0.287985</td>
      <td>0.434101</td>
      <td>2.168266</td>
    </tr>
  </tbody>
</table>
</div>


```python
#print(RegionInfoDict)
```


```python
CitysInfo = {}
for key, value in RegionInfoDict.items():
    CitysInfo[key] = value['Region']
```


```python
import numpy as np
import pandas as pd
import seaborn as sns
import random
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stt
import matplotlib.pyplot as plt
import math
maps=pd.read_csv('concap.csv')
```


```python
happy2020=pd.read_csv('2020.csv')
t1=happy2020[happy2020['Country name']=='Russia']
t1.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country name</th>
      <th>Regional indicator</th>
      <th>Ladder score</th>
      <th>Standard error of ladder score</th>
      <th>upperwhisker</th>
      <th>lowerwhisker</th>
      <th>Logged GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
      <th>Ladder score in Dystopia</th>
      <th>Explained by: Log GDP per capita</th>
      <th>Explained by: Social support</th>
      <th>Explained by: Healthy life expectancy</th>
      <th>Explained by: Freedom to make life choices</th>
      <th>Explained by: Generosity</th>
      <th>Explained by: Perceptions of corruption</th>
      <th>Dystopia + residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>72</td>
      <td>Russia</td>
      <td>Commonwealth of Independent States</td>
      <td>5.546</td>
      <td>0.03961</td>
      <td>5.623635</td>
      <td>5.468365</td>
      <td>10.128872</td>
      <td>0.903151</td>
      <td>64.100456</td>
      <td>0.729893</td>
      <td>-0.151154</td>
      <td>0.864803</td>
      <td>1.972317</td>
      <td>1.127</td>
      <td>1.378644</td>
      <td>0.680446</td>
      <td>0.3995</td>
      <td>0.099042</td>
      <td>0.045699</td>
      <td>1.815717</td>
    </tr>
  </tbody>
</table>
</div>

![image-20200730172613938](/Users/wangwenqing/Library/Application Support/typora-user-images/image-20200730172613938.png)

我国处于中等水平，北美，澳洲，西欧国家幸福评分较高，非洲国家幸福评分较低。


```python
contr_list = list(happy2020[happy2020['Regional indicator'].isin(['East Asia','Southeast Asia','South Asia','Commonwealth of Independent States'])]['Country name'].unique())
as_gps = maps[maps['CountryName'].isin(contr_list)]
as_data = happy2020[happy2020['Regional indicator'].isin(['East Asia','Southeast Asia','South Asia','Commonwealth of Independent States'])]
asi =pd.merge(as_gps[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],\
         as_data,left_on='CountryName',right_on='Country name')
```


```python
def mapAsia(column_color, column_size,colbar=True):
    m = Basemap(projection='mill',llcrnrlon=21,llcrnrlat=-12,urcrnrlon=150,urcrnrlat=60,resolution='l',lat_0=35,lon_0=120)
    m.drawcountries()
    m.drawstates()
    #m.drawmapboundary()
    #s=1000*a_2
    m.scatter(lon, lat, latlon=True,c=100*a_1,s=0.03*pow(2.8,a_2),linewidth=1,edgecolors='black',cmap='gist_heat', alpha=1)
    m.fillcontinents(color='#072B57',lake_color='#FFFFFF',alpha=0.3)
    if colbar:
            m.colorbar(label='Happiness Score*100')
    else:pass
plt.figure(figsize=(15,15))
plt.title('Asia - Happiness&GDP', fontsize=30)
mapAsia('Ladder score','Logged GDP per capita')
```

![png](output_66_1.png)

图中圆圈大小代表0.03*GDP

我国的幸福指数和GDP处于亚洲中上水平

可以明显看出GDP高的国家普遍幸福评分也较高

新加坡的幸福指数和GDP都是亚洲最高的

阿富汗这两个指标是亚洲最低的

针对不同的数据采用不同的衡量维度

```python
 m.scatter(lon, lat, latlon=True,c=100*a_1,s=1000*a_2,linewidth=1,edgecolors='black',cmap='gist_heat', alpha=1)
    m.fillcontinents(color='#072B57',lake_color='#FFFFFF',alpha=0.3)
```


```python
plt.figure(figsize=(15,15))
plt.title('Asia - Happiness&Freedom', fontsize=30)
mapAsia('Ladder score','Freedom to make life choices')
```

![png](output_68_0.png)

圆圈大小为1000*自由导向程度。

亚洲各国的自由导向程度差异不大。

阿富汗自由导向程度很低，幸福指数也很低。

```python
 m.scatter(lon, lat, latlon=True,c=100*a_1,s=40*(a_2-50),linewidth=1,edgecolors='black',cmap='gist_heat', alpha=1)
    m.fillcontinents(color='#072B57',lake_color='#FFFFFF',alpha=0.3)
```


```python
plt.figure(figsize=(15,15))
plt.title('Asia - Happiness&Health', fontsize=30)
mapAsia('Ladder score','Healthy life expectancy')
```

![png](output_70_1.png)

图中圆圈大小代表(预期寿命-50)*40大小。

可以看出预期寿命高的国家普遍幸福评分也较高。

我国的幸福指数和预期寿命处于亚洲中上水平。

新加坡的幸福指数和健康程度都是亚洲最高的。

阿富汗这两个指标是亚洲最低的。

```python
from mpl_toolkits.basemap import Basemap
happy2020=pd.read_csv('2020.csv')
contr_list = list(happy2020[happy2020['Regional indicator'].isin(['Sub-Saharan Africa','Middle East and North Africa'])]['Country name'].unique())
#print(contr_list)
afr_gps = maps[maps['CountryName'].isin(contr_list)]
#print(afr_gps)
afr_data = happy2020[happy2020['Regional indicator'].isin(['Sub-Saharan Africa','Middle East and North Africa'])]
#print(afr_data)
afr =pd.merge(afr_gps[['CountryName', 'CapitalName', 'CapitalLatitude', 'CapitalLongitude']],\
         afr_data,left_on='CountryName',right_on='Country name')
afr.head()
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CountryName</th>
      <th>CapitalName</th>
      <th>CapitalLatitude</th>
      <th>CapitalLongitude</th>
      <th>Country name</th>
      <th>Regional indicator</th>
      <th>Ladder score</th>
      <th>Standard error of ladder score</th>
      <th>upperwhisker</th>
      <th>lowerwhisker</th>
      <th>...</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
      <th>Ladder score in Dystopia</th>
      <th>Explained by: Log GDP per capita</th>
      <th>Explained by: Social support</th>
      <th>Explained by: Healthy life expectancy</th>
      <th>Explained by: Freedom to make life choices</th>
      <th>Explained by: Generosity</th>
      <th>Explained by: Perceptions of corruption</th>
      <th>Dystopia + residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Algeria</td>
      <td>Algiers</td>
      <td>36.750000</td>
      <td>3.050000</td>
      <td>Algeria</td>
      <td>Middle East and North Africa</td>
      <td>5.0051</td>
      <td>0.044236</td>
      <td>5.091802</td>
      <td>4.918397</td>
      <td>...</td>
      <td>-0.121105</td>
      <td>0.735485</td>
      <td>1.972317</td>
      <td>0.943856</td>
      <td>1.143004</td>
      <td>0.745419</td>
      <td>0.083944</td>
      <td>0.118915</td>
      <td>0.129191</td>
      <td>1.840812</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Bahrain</td>
      <td>Manama</td>
      <td>26.233333</td>
      <td>50.566667</td>
      <td>Bahrain</td>
      <td>Middle East and North Africa</td>
      <td>6.2273</td>
      <td>0.081882</td>
      <td>6.387789</td>
      <td>6.066811</td>
      <td>...</td>
      <td>0.133729</td>
      <td>0.739347</td>
      <td>1.972317</td>
      <td>1.296692</td>
      <td>1.315324</td>
      <td>0.838836</td>
      <td>0.610400</td>
      <td>0.287454</td>
      <td>0.126697</td>
      <td>1.751917</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Benin</td>
      <td>Porto-Novo</td>
      <td>6.483333</td>
      <td>2.616667</td>
      <td>Benin</td>
      <td>Sub-Saharan Africa</td>
      <td>5.2160</td>
      <td>0.077759</td>
      <td>5.368408</td>
      <td>5.063592</td>
      <td>...</td>
      <td>-0.003537</td>
      <td>0.740533</td>
      <td>1.972317</td>
      <td>0.366245</td>
      <td>0.352428</td>
      <td>0.328063</td>
      <td>0.405840</td>
      <td>0.196670</td>
      <td>0.125932</td>
      <td>3.440810</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Botswana</td>
      <td>Gaborone</td>
      <td>-24.633333</td>
      <td>25.900000</td>
      <td>Botswana</td>
      <td>Sub-Saharan Africa</td>
      <td>3.4789</td>
      <td>0.060543</td>
      <td>3.597564</td>
      <td>3.360236</td>
      <td>...</td>
      <td>-0.250394</td>
      <td>0.777931</td>
      <td>1.972317</td>
      <td>0.997549</td>
      <td>1.085695</td>
      <td>0.494102</td>
      <td>0.509089</td>
      <td>0.033407</td>
      <td>0.101786</td>
      <td>0.257241</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Burkina Faso</td>
      <td>Ouagadougou</td>
      <td>12.366667</td>
      <td>-1.516667</td>
      <td>Burkina Faso</td>
      <td>Sub-Saharan Africa</td>
      <td>4.7687</td>
      <td>0.062067</td>
      <td>4.890352</td>
      <td>4.647048</td>
      <td>...</td>
      <td>-0.019081</td>
      <td>0.739795</td>
      <td>1.972317</td>
      <td>0.302468</td>
      <td>0.929386</td>
      <td>0.312834</td>
      <td>0.322398</td>
      <td>0.186391</td>
      <td>0.126408</td>
      <td>2.588826</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
```python
def mapAfrica(column_color, column_size,colbar=True):
    m = Basemap(projection='mill',llcrnrlat=-38,urcrnrlat=45,\
                llcrnrlon=-22,urcrnrlon=66,resolution='l')
    m.drawcountries()
    m.drawstates()
    #m.drawmapboundary()
    m.drawparallels(np.arange(-90,91.,30.))
    m.drawmeridians(np.arange(-90,90.,60.))
    lat = afr['CapitalLatitude'].values
    lon = afr['CapitalLongitude'].values
    a_1 = afr[column_color].values
    a_2 = afr[column_size].values
    print(column_color)
    #s=1000*a_2
    m.scatter(lon, lat, latlon=True,c=100*a_1,s=1000*a_2,linewidth=2,edgecolors='black',cmap='gist_heat', alpha=1)
    m.fillcontinents(color='#072B57',lake_color='#FFFFFF',alpha=0.3)
    if colbar:
            m.colorbar(label='Happiness Score*100')
    else:pass
```


```python
plt.figure(figsize=(10,10))
plt.title('Africa - Happiness&GDP', fontsize=30)
mapAfrica('Ladder score','Logged GDP per capita')
```

![png](output_73_2.png)

```python
plt.figure(figsize=(10,10))
plt.title('Africa - Happiness&Health', fontsize=30)
mapAfrica('Ladder score','Healthy life expectancy')
```

![png](output_74_2.png)

```python
plt.figure(figsize=(12,12))
plt.title('Africa - Happiness&Freedom', fontsize=30)
mapAfrica('Ladder score','Freedom to make life choices')
```


    Ladder score



![png](output_75_2.png)


    [4.88269997 7.29419994 6.86350012 5.67409992 5.10150003 5.50470018
     6.15899992 6.91090012 7.64559984 6.02180004 7.80870008 6.66379976
     7.07579994 5.51499987 6.00040007 7.50449991 7.09369993 6.38740015
     6.32520008 5.94999981 6.21549988 7.23750019 5.15980005 6.77279997
     5.54610014 7.44890022 7.48799992 6.1862998  5.91090012 6.12370014
     5.77820015 6.28060007 6.36339998 6.40089989 7.35349989 7.55989981
     7.16450024]

![png](output_78_2.png)


    [4.88269997 7.29419994 6.86350012 5.67409992 5.10150003 5.50470018
     6.15899992 6.91090012 7.64559984 6.02180004 7.80870008 6.66379976
     7.07579994 5.51499987 6.00040007 7.50449991 7.09369993 6.38740015
     6.32520008 5.94999981 6.21549988 7.23750019 5.15980005 6.77279997
     5.54610014 7.44890022 7.48799992 6.1862998  5.91090012 6.12370014
     5.77820015 6.28060007 6.36339998 6.40089989 7.35349989 7.55989981
     7.16450024]

![png](output_79_2.png)




![png](output_80_0.png)




```python
temp=temp.merge(rate)
temp.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Region</th>
      <th>Happiness Rank</th>
      <th>Happiness Score</th>
      <th>Lower Confidence Interval</th>
      <th>Upper Confidence Interval</th>
      <th>Economy (GDP per Capita)</th>
      <th>Family</th>
      <th>Health (Life Expectancy)</th>
      <th>Freedom</th>
      <th>Trust (Government Corruption)</th>
      <th>Generosity</th>
      <th>Dystopia Residual</th>
      <th>hf_score</th>
      <th>pf_score</th>
      <th>year</th>
      <th>suicides_100k</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Denmark</td>
      <td>Western Europe</td>
      <td>1</td>
      <td>7.526</td>
      <td>7.460</td>
      <td>7.592</td>
      <td>1.44178</td>
      <td>1.16374</td>
      <td>0.79504</td>
      <td>0.57941</td>
      <td>0.44453</td>
      <td>0.36171</td>
      <td>2.73939</td>
      <td>8.547820</td>
      <td>9.325640</td>
      <td>2010</td>
      <td>10.835000</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Switzerland</td>
      <td>Western Europe</td>
      <td>2</td>
      <td>7.509</td>
      <td>7.428</td>
      <td>7.590</td>
      <td>1.52733</td>
      <td>1.14524</td>
      <td>0.86303</td>
      <td>0.58557</td>
      <td>0.41203</td>
      <td>0.28083</td>
      <td>2.69463</td>
      <td>8.787759</td>
      <td>9.185518</td>
      <td>2010</td>
      <td>13.376667</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Iceland</td>
      <td>Western Europe</td>
      <td>3</td>
      <td>7.501</td>
      <td>7.333</td>
      <td>7.669</td>
      <td>1.42666</td>
      <td>1.18326</td>
      <td>0.86733</td>
      <td>0.56624</td>
      <td>0.14975</td>
      <td>0.47678</td>
      <td>2.83137</td>
      <td>8.151753</td>
      <td>9.083506</td>
      <td>2010</td>
      <td>15.355833</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Norway</td>
      <td>Western Europe</td>
      <td>4</td>
      <td>7.498</td>
      <td>7.421</td>
      <td>7.575</td>
      <td>1.57744</td>
      <td>1.12690</td>
      <td>0.79579</td>
      <td>0.59609</td>
      <td>0.35776</td>
      <td>0.37895</td>
      <td>2.66465</td>
      <td>8.471241</td>
      <td>9.342481</td>
      <td>2010</td>
      <td>11.369167</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Finland</td>
      <td>Western Europe</td>
      <td>5</td>
      <td>7.413</td>
      <td>7.351</td>
      <td>7.475</td>
      <td>1.40598</td>
      <td>1.13464</td>
      <td>0.81091</td>
      <td>0.57104</td>
      <td>0.41004</td>
      <td>0.25492</td>
      <td>2.82596</td>
      <td>8.472184</td>
      <td>9.294368</td>
      <td>2010</td>
      <td>17.934167</td>
    </tr>
  </tbody>
</table>
</div>


```python
fig = plt.figure(figsize=(7,5))
sns.set()

sns.distplot(happy2020['Ladder score'],bins=13)
plt.xlabel('Happiness Score')
plt.show()
```


![png](output_83_0.png)

分布接近正态分布 相对分布在幸福评分较高的国家比较低的国家比重大。

```python
plt.rcParams['figure.figsize'] = (28,20)

temp=temp[['Happiness Score','Economy (GDP per Capita)','Health (Life Expectancy)','Freedom','hf_score','pf_score','Generosity','Trust (Government Corruption)','suicides_100k']]
ax=sns.heatmap(temp.corr(),annot = True,cmap="coolwarm")
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.xticks(rotation = 80,fontsize=30)
plt.yticks(fontsize=30)
plt.show()
```


![png](output_84_0.png)

幸福指数与自由导向相关性最强、经济的相关性其次、与慷慨度、信任度、健康相关性较强、与国民自由、个人自由相关性中等、

与自杀率正向相关性很低。


![png](output_85_0.png)



```python

```


```python
top = happy2020.sort_values(['Ladder score'],ascending = 0)[:10]
ax = sns.barplot(x = 'Ladder score' , y = 'Country name' , data = top)
ax.set_xlabel('Happiness score', size = 20)
ax.tick_params(labelsize=10) 
ax.set_ylabel('Country name', size = 20)
ax.set_title("Top 10 Happiest Countries", size = 30)
```




    Text(0.5, 1.0, 'Top 10 Happiest Countries')


![png](output_87_1.png)

```python
happy2020.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country name</th>
      <th>Regional indicator</th>
      <th>Ladder score</th>
      <th>Logged GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
      <th>Ladder score in Dystopia</th>
      <th>Dystopia + residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Finland</td>
      <td>Western Europe</td>
      <td>7.8087</td>
      <td>10.639267</td>
      <td>0.954330</td>
      <td>71.900825</td>
      <td>0.949172</td>
      <td>-0.059482</td>
      <td>0.195445</td>
      <td>1.972317</td>
      <td>2.762835</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Denmark</td>
      <td>Western Europe</td>
      <td>7.6456</td>
      <td>10.774001</td>
      <td>0.955991</td>
      <td>72.402504</td>
      <td>0.951444</td>
      <td>0.066202</td>
      <td>0.168489</td>
      <td>1.972317</td>
      <td>2.432741</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Switzerland</td>
      <td>Western Europe</td>
      <td>7.5599</td>
      <td>10.979933</td>
      <td>0.942847</td>
      <td>74.102448</td>
      <td>0.921337</td>
      <td>0.105911</td>
      <td>0.303728</td>
      <td>1.972317</td>
      <td>2.350267</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Iceland</td>
      <td>Western Europe</td>
      <td>7.5045</td>
      <td>10.772559</td>
      <td>0.974670</td>
      <td>73.000000</td>
      <td>0.948892</td>
      <td>0.246944</td>
      <td>0.711710</td>
      <td>1.972317</td>
      <td>2.460688</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Norway</td>
      <td>Western Europe</td>
      <td>7.4880</td>
      <td>11.087804</td>
      <td>0.952487</td>
      <td>73.200783</td>
      <td>0.955750</td>
      <td>0.134533</td>
      <td>0.263218</td>
      <td>1.972317</td>
      <td>2.168266</td>
    </tr>
  </tbody>
</table>
</div>



```python
free18.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>ISO_code</th>
      <th>countries</th>
      <th>region</th>
      <th>pf_rol_procedural</th>
      <th>pf_rol_civil</th>
      <th>pf_rol_criminal</th>
      <th>pf_rol</th>
      <th>pf_ss_homicide</th>
      <th>pf_ss_disappearances_disap</th>
      <th>...</th>
      <th>ef_regulation_business_bribes</th>
      <th>ef_regulation_business_licensing</th>
      <th>ef_regulation_business_compliance</th>
      <th>ef_regulation_business</th>
      <th>ef_regulation</th>
      <th>ef_score</th>
      <th>ef_rank</th>
      <th>hf_score</th>
      <th>hf_rank</th>
      <th>hf_quartile</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2016</td>
      <td>ALB</td>
      <td>Albania</td>
      <td>Eastern Europe</td>
      <td>6.661503</td>
      <td>4.547244</td>
      <td>4.666508</td>
      <td>5.291752</td>
      <td>8.920429</td>
      <td>10.0</td>
      <td>...</td>
      <td>4.050196</td>
      <td>7.324582</td>
      <td>7.074366</td>
      <td>6.705863</td>
      <td>6.906901</td>
      <td>7.54</td>
      <td>34.0</td>
      <td>7.568140</td>
      <td>48.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2016</td>
      <td>DZA</td>
      <td>Algeria</td>
      <td>Middle East &amp; North Africa</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.819566</td>
      <td>9.456254</td>
      <td>10.0</td>
      <td>...</td>
      <td>3.765515</td>
      <td>8.523503</td>
      <td>7.029528</td>
      <td>5.676956</td>
      <td>5.268992</td>
      <td>4.99</td>
      <td>159.0</td>
      <td>5.135886</td>
      <td>155.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2016</td>
      <td>AGO</td>
      <td>Angola</td>
      <td>Sub-Saharan Africa</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.451814</td>
      <td>8.060260</td>
      <td>5.0</td>
      <td>...</td>
      <td>1.945540</td>
      <td>8.096776</td>
      <td>6.782923</td>
      <td>4.930271</td>
      <td>5.518500</td>
      <td>5.17</td>
      <td>155.0</td>
      <td>5.640662</td>
      <td>142.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2016</td>
      <td>ARG</td>
      <td>Argentina</td>
      <td>Latin America &amp; the Caribbean</td>
      <td>7.098483</td>
      <td>5.791960</td>
      <td>4.343930</td>
      <td>5.744791</td>
      <td>7.622974</td>
      <td>10.0</td>
      <td>...</td>
      <td>3.260044</td>
      <td>5.253411</td>
      <td>6.508295</td>
      <td>5.535831</td>
      <td>5.369019</td>
      <td>4.84</td>
      <td>160.0</td>
      <td>6.469848</td>
      <td>107.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2016</td>
      <td>ARM</td>
      <td>Armenia</td>
      <td>Caucasus &amp; Central Asia</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.003205</td>
      <td>8.808750</td>
      <td>10.0</td>
      <td>...</td>
      <td>4.575152</td>
      <td>9.319612</td>
      <td>6.491481</td>
      <td>6.797530</td>
      <td>7.378069</td>
      <td>7.57</td>
      <td>29.0</td>
      <td>7.241402</td>
      <td>57.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 123 columns</p>
</div>

```python
happy2019.head()
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Overall rank</th>
      <th>Country or region</th>
      <th>Score</th>
      <th>GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>Finland</td>
      <td>7.769</td>
      <td>1.340</td>
      <td>1.587</td>
      <td>0.986</td>
      <td>0.596</td>
      <td>0.153</td>
      <td>0.393</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>Denmark</td>
      <td>7.600</td>
      <td>1.383</td>
      <td>1.573</td>
      <td>0.996</td>
      <td>0.592</td>
      <td>0.252</td>
      <td>0.410</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3</td>
      <td>Norway</td>
      <td>7.554</td>
      <td>1.488</td>
      <td>1.582</td>
      <td>1.028</td>
      <td>0.603</td>
      <td>0.271</td>
      <td>0.341</td>
    </tr>
    <tr>
      <td>3</td>
      <td>4</td>
      <td>Iceland</td>
      <td>7.494</td>
      <td>1.380</td>
      <td>1.624</td>
      <td>1.026</td>
      <td>0.591</td>
      <td>0.354</td>
      <td>0.118</td>
    </tr>
    <tr>
      <td>4</td>
      <td>5</td>
      <td>Netherlands</td>
      <td>7.488</td>
      <td>1.396</td>
      <td>1.522</td>
      <td>0.999</td>
      <td>0.557</td>
      <td>0.322</td>
      <td>0.298</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set(rc={'figure.figsize':(14,15)})
#g = sns.FacetGrid(happiness, col="Region",  margin_titles=True, col_wrap=3)
sns.set(font_scale=1.6)
#(g.map(plt.scatter, "Generosity","Happiness Score", edgecolor="w")).add_legend()
sns.scatterplot(happy2020['Freedom to make life choices'],happy2020['Ladder score'], s=400,hue=happy2020['Regional indicator'],data=happy2020,alpha=0.6)
plt.xlabel('Freedom',fontsize=30)
plt.ylabel('Happiness Score',fontsize=30)
plt.show
```




    <function matplotlib.pyplot.show(*args, **kw)>




![png](output_105_1.png)

从图中看到，自由导向程度和幸福评分的有一定的相关性。

幸福评分最高的西欧国家的自由度也普遍较高。

幸福评分较低的撒哈拉沙漠地区自由评分主要集中在中等偏下的区域。

**各地区幸福评分分析**


```python
plt.figure(figsize=(8,8))
top = happy2020.sort_values(['Ladder score'],ascending = 0)[:10]
ax = sns.barplot(x = 'Ladder score' , y = 'Country name' , data = top)
ax.set_xlabel('Happiness Score', size = 20)
ax.set_ylabel('Country name', size = 20)
ax.set_title("Top 10 happiest Countries", size = 25)
```


    Text(0.5, 1.0, 'Top 10 happiest Countries')




![png](output_111_1.png)



```python
happy2020.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country name</th>
      <th>Regional indicator</th>
      <th>Ladder score</th>
      <th>Logged GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
      <th>Ladder score in Dystopia</th>
      <th>Dystopia + residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Finland</td>
      <td>Western Europe</td>
      <td>7.8087</td>
      <td>10.639267</td>
      <td>0.954330</td>
      <td>71.900825</td>
      <td>0.949172</td>
      <td>-0.059482</td>
      <td>0.195445</td>
      <td>1.972317</td>
      <td>2.762835</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Denmark</td>
      <td>Western Europe</td>
      <td>7.6456</td>
      <td>10.774001</td>
      <td>0.955991</td>
      <td>72.402504</td>
      <td>0.951444</td>
      <td>0.066202</td>
      <td>0.168489</td>
      <td>1.972317</td>
      <td>2.432741</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Switzerland</td>
      <td>Western Europe</td>
      <td>7.5599</td>
      <td>10.979933</td>
      <td>0.942847</td>
      <td>74.102448</td>
      <td>0.921337</td>
      <td>0.105911</td>
      <td>0.303728</td>
      <td>1.972317</td>
      <td>2.350267</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Iceland</td>
      <td>Western Europe</td>
      <td>7.5045</td>
      <td>10.772559</td>
      <td>0.974670</td>
      <td>73.000000</td>
      <td>0.948892</td>
      <td>0.246944</td>
      <td>0.711710</td>
      <td>1.972317</td>
      <td>2.460688</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Norway</td>
      <td>Western Europe</td>
      <td>7.4880</td>
      <td>11.087804</td>
      <td>0.952487</td>
      <td>73.200783</td>
      <td>0.955750</td>
      <td>0.134533</td>
      <td>0.263218</td>
      <td>1.972317</td>
      <td>2.168266</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(20,5))
sns.boxplot( x=happy2020['Regional indicator'],y=happy2020['Ladder score'],order=['Middle East and North Africa','South Asia','Latin America and Caribbean','Sub-Saharan Africa','Western Europe','Southeast Asia','Central and Eastern Europe','North America and ANZ','East Asia','Commonwealth of Independent States' ])
plt.ylabel('Happiness score',fontsize=30.0)
plt.xlabel('Region',fontsize=30.0)
plt.xticks(rotation=30)
plt.show()
```


![png](output_113_0.png)



```python
plt.figure(figsize=(20,5))
sns.boxplot( x=happy2020['Regional indicator'],y=happy2020['Ladder score'])
plt.ylabel('Happiness score',fontsize=30.0)
plt.xlabel('Region',fontsize=30.0)
plt.xticks(rotation=30)
plt.show()
```


![png](output_114_0.png)



```python
plt.figure(figsize=(20,5))
sns.violinplot( x=happy2020['Regional indicator'],y=happy2020['Ladder score'],palette="rainbow" )
plt.ylabel('Happiness score',fontsize=30.0)
plt.xlabel('Region',fontsize=30.0)

plt.xticks(rotation=45)
```




    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), <a list of 10 Text xticklabel objects>)




![png](output_115_1.png)



```python
sns.set(font_scale=1.2)
ax=sns.lmplot(x="Healthy life expectancy",y="Ladder score",data=happy2020,col="Regional indicator",col_wrap=2,palette="coolwarm")

```



**自由以及预期寿命与幸福评分的关系**


```python
sns.set(rc={'figure.figsize':(15,15)})
#g = sns.FacetGrid(happiness, col="Region",  margin_titles=True, col_wrap=3)
sns.set(font_scale=1.6)
#(g.map(plt.scatter, "Generosity","Happiness Score", edgecolor="w")).add_legend()
sns.scatterplot(happy2020['Healthy life expectancy'],happy2020['Ladder score'], hue=happy2020['Regional indicator'],s=400,data=happy2020,alpha=0.6)
plt.xlabel('Healthy life expectancy',fontsize=30)
plt.ylabel('Happiness Score',fontsize=30)
plt.show
```


![png](output_119_1.png)


观察到幸福评分最小值为2.5多 所以用size =(happy2020['Ladder score']-2.5)*300可以更好的刻画圆圈大小的差异


```python
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"/System/Library/Fonts/PingFang.ttc")
plt.rcParams['axes.unicode_minus'] = False
```


```python
from pylab import mpl

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"/System/Library/Fonts/PingFang.ttc")

plt.rcParams['axes.unicode_minus'] = False
freedom=happy2020['Freedom to make life choices']
life=happy2020['Healthy life expectancy']
size =(happy2020['Ladder score']-2.5)*300
colors = np.random.rand(len(freedom))  # 颜色数组
plt.scatter(freedom, life, s=size,c=colors,cmap='coolwarm',alpha=0.5)  # 画散点图, alpha=0.6 表示不透明度为 0.6
plt.ylim([43, 78])  # 纵坐标轴范围
plt.xlim([0.35, 1])   # 横坐标轴范围
plt.xlabel('生活中做决定的自由',fontproperties=font,fontsize=20)  # 横坐标轴标题
plt.ylabel('健康预期寿命',fontproperties=font,fontsize=20)  # 纵坐标轴标题
plt.title("自由以及预期寿命与幸福评分的关系",fontproperties=font,fontsize=25)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
#plt.legend()
#print(len(freedom))
#print(colors)
plt.text(0.37, 76, r'圆圈大小代表幸福评分大小',fontproperties=font,fontsize=20)
plt.show()
```


![png](output_122_0.png)

**全球经济发展程度**

我国处于中等水平，北美，澳洲，西欧国家经济较发达，非洲国家经济欠发达。

![image-20200730171922457](/Users/wangwenqing/Library/Application Support/typora-user-images/image-20200730171922457.png)

国民经济与幸福评分的关系**


```python
sns.set(rc={'figure.figsize':(15,15)})
#g = sns.FacetGrid(happiness, col="Region",  margin_titles=True, col_wrap=3)
sns.set(font_scale=1.6)
#(g.map(plt.scatter, "Generosity","Happiness Score", edgecolor="w")).add_legend()
sns.scatterplot(happy2020['Logged GDP per capita'],happy2020['Ladder score'], hue=happy2020['Regional indicator'],s=400,data=happy2020,alpha=0.6)
plt.xlabel('Logged GDP per capita',fontsize=30)
plt.ylabel('Happiness Score',fontsize=30)
plt.show
```




    <function matplotlib.pyplot.show(*args, **kw)>




![png](output_124_1.png)

随着GDP的增加，幸福指数增加。

撒哈拉沙漠地区GDP和幸福指数都较低。

西欧地区GDP普遍较高，幸福指数分布在中上部分区域。

```python
sns.set(rc={'figure.figsize':(15,15)})
#g = sns.FacetGrid(happiness, col="Region",  margin_titles=True, col_wrap=3)
sns.set(font_scale=1.6)
#(g.map(plt.scatter, "Generosity","Happiness Score", edgecolor="w")).add_legend()
sns.scatterplot(happy2020['Perceptions of corruption'],happy2020['Ladder score'], hue=happy2020['Regional indicator'],s=400,data=happy2020,alpha=0.6)
plt.xlabel('Perceptions of corruption',fontsize=30)
plt.ylabel('Happiness Score',fontsize=30)
plt.show
```


![png](output_125_1.png)

腐败程度较低的国家大部分幸福指数较高。

腐败程度较高的国家的幸福指数分布不一。

```python
happy2020.head()
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country name</th>
      <th>Regional indicator</th>
      <th>Ladder score</th>
      <th>Standard error of ladder score</th>
      <th>upperwhisker</th>
      <th>lowerwhisker</th>
      <th>Logged GDP per capita</th>
      <th>Social support</th>
      <th>Healthy life expectancy</th>
      <th>Freedom to make life choices</th>
      <th>Generosity</th>
      <th>Perceptions of corruption</th>
      <th>Ladder score in Dystopia</th>
      <th>Explained by: Log GDP per capita</th>
      <th>Explained by: Social support</th>
      <th>Explained by: Healthy life expectancy</th>
      <th>Explained by: Freedom to make life choices</th>
      <th>Explained by: Generosity</th>
      <th>Explained by: Perceptions of corruption</th>
      <th>Dystopia + residual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Finland</td>
      <td>Western Europe</td>
      <td>7.8087</td>
      <td>0.031156</td>
      <td>7.869766</td>
      <td>7.747634</td>
      <td>10.639267</td>
      <td>0.954330</td>
      <td>71.900825</td>
      <td>0.949172</td>
      <td>-0.059482</td>
      <td>0.195445</td>
      <td>1.972317</td>
      <td>1.285190</td>
      <td>1.499526</td>
      <td>0.961271</td>
      <td>0.662317</td>
      <td>0.159670</td>
      <td>0.477857</td>
      <td>2.762835</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Denmark</td>
      <td>Western Europe</td>
      <td>7.6456</td>
      <td>0.033492</td>
      <td>7.711245</td>
      <td>7.579955</td>
      <td>10.774001</td>
      <td>0.955991</td>
      <td>72.402504</td>
      <td>0.951444</td>
      <td>0.066202</td>
      <td>0.168489</td>
      <td>1.972317</td>
      <td>1.326949</td>
      <td>1.503449</td>
      <td>0.979333</td>
      <td>0.665040</td>
      <td>0.242793</td>
      <td>0.495260</td>
      <td>2.432741</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Switzerland</td>
      <td>Western Europe</td>
      <td>7.5599</td>
      <td>0.035014</td>
      <td>7.628528</td>
      <td>7.491272</td>
      <td>10.979933</td>
      <td>0.942847</td>
      <td>74.102448</td>
      <td>0.921337</td>
      <td>0.105911</td>
      <td>0.303728</td>
      <td>1.972317</td>
      <td>1.390774</td>
      <td>1.472403</td>
      <td>1.040533</td>
      <td>0.628954</td>
      <td>0.269056</td>
      <td>0.407946</td>
      <td>2.350267</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Iceland</td>
      <td>Western Europe</td>
      <td>7.5045</td>
      <td>0.059616</td>
      <td>7.621347</td>
      <td>7.387653</td>
      <td>10.772559</td>
      <td>0.974670</td>
      <td>73.000000</td>
      <td>0.948892</td>
      <td>0.246944</td>
      <td>0.711710</td>
      <td>1.972317</td>
      <td>1.326502</td>
      <td>1.547567</td>
      <td>1.000843</td>
      <td>0.661981</td>
      <td>0.362330</td>
      <td>0.144541</td>
      <td>2.460688</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Norway</td>
      <td>Western Europe</td>
      <td>7.4880</td>
      <td>0.034837</td>
      <td>7.556281</td>
      <td>7.419719</td>
      <td>11.087804</td>
      <td>0.952487</td>
      <td>73.200783</td>
      <td>0.955750</td>
      <td>0.134533</td>
      <td>0.263218</td>
      <td>1.972317</td>
      <td>1.424207</td>
      <td>1.495173</td>
      <td>1.008072</td>
      <td>0.670201</td>
      <td>0.287985</td>
      <td>0.434101</td>
      <td>2.168266</td>
    </tr>
  </tbody>
</table>
</div>


```python
import plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot
map_plot = dict(type = 'choropleth', 
locations = happy2020['Country name'],
locationmode = 'country names',
z = happy2020['Logged GDP per capita'].astype(float), 
text =happy2020['Regional indicator'],
colorscale='ylgnbu',
autocolorscale = False, reversescale = True,colorbar=dict(
title = "Logged GDP per capita"))
layout = dict(title = 'Most Wealthiest Countries In The World ', 
geo = dict(showframe = False, 
projection = {'type': 'equirectangular'}))
choromap = go.Figure(data = [map_plot], layout=layout)
iplot(choromap)
```

**幸福评分与社会支持以及慷慨度的关系**


```python
from mpl_toolkits.mplot3d import Axes3D
x, y, z = happy2020['Social support'], happy2020['Generosity'], happy2020['Ladder score']
ax = plt.subplot(111,projection='3d') # 创建一个三维的绘图工程
#  将数据点分成三部分画，在颜色上有区分度
l1=ax.scatter(x[:51], y[:51], z[:51], c='blue',s=200)  # 绘制数据点
l2=ax.scatter(x[52:102], y[52:102], z[52:102], c='red',s=200)
l3=ax.scatter(x[103:], y[103:], z[103:], c='green',s=200)

ax.set_zlabel('happiness score',fontsize=25)  # 坐标轴
ax.set_ylabel('Generosity',fontsize=20)
ax.set_xlabel('Social support',fontsize=20)

#ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis._axinfo["grid"]['linewidth'] = 5
ax.xaxis._axinfo["grid"]['linewidth'] = 5
ax.zaxis._axinfo["grid"]['linewidth'] = 5

ax.set_facecolor('white')
plt.legend(handles=[l1,l2,l3],labels=['countries whose happiness score ranks 1-51','countries whose happiness score ranks 52-102','countries whose happiness score ranks 103-153'],loc='best')
plt.show()

```


![png](output_130_0.png)

经济、预期寿命均与幸福指数正相关。

幸福指数排名较高的国家经济和预期寿命都较高。

幸福指数排名较低的国家经济和预期寿命都较低。




```python
from mpl_toolkits.mplot3d import Axes3D
x, y, z = happy2020['Healthy life expectancy'], happy2020['Logged GDP per capita'], happy2020['Ladder score']
ax = plt.subplot(111,projection='3d') # 创建一个三维的绘图工程
#  将数据点分成三部分画，在颜色上有区分度
l1=ax.scatter(x[:51], y[:51], z[:51], c='blue',s=200)  # 绘制数据点
l2=ax.scatter(x[52:102], y[52:102], z[52:102], c='red',s=200)
l3=ax.scatter(x[103:], y[103:], z[103:], c='green',s=200)
ax.set_ylabel('Economy',fontsize=20)
ax.set_xlabel('Healthy life expectancy',fontsize=20)
ax.set_zlabel('Happiness score',fontsize=20)
ax.yaxis._axinfo["grid"]['linewidth'] = 5
ax.xaxis._axinfo["grid"]['linewidth'] = 5
ax.zaxis._axinfo["grid"]['linewidth'] = 5

ax.set_facecolor('white')
plt.legend(handles=[l1,l2,l3],labels=['countries whose happiness score ranks 1-51','countries whose happiness score ranks 52-102','countries whose happiness score ranks 103-153'],loc='best')
plt.show()
```


![png](output_132_0.png)

幸福指数排名较高的国家社会支持和慷慨度都较高。

随着慷慨度增加 幸福指数增加，幸福指数较低的国家， 慷慨度较低 社会支持分布不一。


```python
plt.figure(figsize=(5,5))
top =happy2020.sort_values(['Healthy life expectancy'],ascending = 0)[:10]
ax = sns.barplot(x = 'Healthy life expectancy' , y = 'Country name' , data = top)
ax.set_xlabel('Healthy life expectancy', size = 20)
ax.set_ylabel('Country name', size = 20)
ax.set_title("Top 10 Healthiest Countries", size = 25)
```




    Text(0.5, 1.0, 'Top 10 Healthiest Countries')




![png](output_134_1.png)

亚洲国家占据了几个席位。

#### 4.总结

本次项目从经济、健康、心理健康、安全等不同角度分析了全球分布的差异以及幸福指数与这些因素的相关性，也了解了如果想要国家的幸福指数提高应该从哪些方面入手。