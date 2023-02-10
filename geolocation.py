import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/content/food_coded.csv")



df.shape

df.isnull()

df.head()

df.tail()

df.dropna(inplace=True)

df.isna().sum()

df.columns

df[['cook', 'eating_out', 'employment', 'ethnic_food', 'exercise', 'fruit_day', 'income', 'on_off_campus', 'pay_meal_out', 'sports', 'veggies_day']]

np.random.seed(42)
df = pd.DataFrame(data = np.random.random(size=(11,11)), columns = ['cook', 'eating_out', 'employment', 'ethnic_food', 'exercise', 'fruit_day', 'income', 'on_off_campus', 'pay_meal_out', 'sports', 'veggies_day'])

df.T.boxplot(vert=False)
plt.subplots_adjust(left=0.25)
plt.show()

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df['income'] = pd.to_numeric(df['income'].astype(float),errors='coerce')

print (df.dtypes)

x = df.iloc[:, [0,1,2,3]].values

kmeans5 = KMeans(n_clusters=5)
y_kmeans5 = kmeans5.fit_predict(x)
print(y_kmeans5)

kmeans5.cluster_centers_

Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()

kmeans3 = KMeans(n_clusters=3)
y_kmeans3 = kmeans3.fit_predict(x)
print(y_kmeans3)

kmeans3.cluster_centers_

plt.scatter(x[:, 0], x[:, 1], c = y_kmeans3, cmap = 'rainbow')

