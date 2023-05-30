#data visualization
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dc import cars
import warnings
warnings.filterwarnings('ignore')

#distribution of car price
plt.subplot(1,2,1)
plt.title('Car Price Distribution Plot')
sns.distplot(cars.price)

plt.subplot(1,2,2)
plt.title('Car Price Spread')
sns.boxplot(y=cars.price)

plt.figure(figsize=(25,6))
#bar plot for car brands
plt.subplot(1,3,1)
plt1=cars.car_brand.value_counts().plot.bar()
plt.title('Brands Histogram')
plt1.set(xlabel='Car company',ylabel='Frequency of brands')

#bar plot for fuel types
plt.subplot(1,3,2)
plt1=cars.fueltype.value_counts().plot.bar()
plt1.set_title('Fuel type Histogram')
plt1.set(xlabel='Fuel Type',ylabel='Frequency of fuel type')

#histogram for car type
plt.subplot(1,3,3)
plt1=cars.carbody.value_counts().plot.bar()
plt1.set_title('Car Type')
plt1.set(xlabel='Car Type',ylabel='Frequency of car type')
plt.show()

plt.figure(figsize=(25,6))
#car brand vs price
brand_avg=cars.groupby('car_brand')['price']\
    .mean().sort_values(ascending=False)
plt.subplot(1,4,1)
brand_avg.plot.bar()
plt.title('Brand Name Vs Average Price')

#fuel type vs price
plt.subplot(1,4,2)
fuel_avg=cars.groupby('fueltype')['price']\
    .mean().sort_values(ascending=False)
fuel_avg.plot.bar()
plt.title('Fuel Type Vs Average Price')

#Body type vs price
plt.subplot(1,4,3)
type_avg=cars.groupby('carbody')['price']\
    .mean().sort_values(ascending=False)
type_avg.plot.bar()
plt.title('Car Type vs Average Price')
plt.show()

#correlation between each variable
plt.figure(figsize=(25,6))
sns.heatmap(cars.corr(),annot=True)
plt.show()