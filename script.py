import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


cluster_data = pd.read_csv('data/train.csv')
print('Cluster dataset generated:\n',cluster_data.head())

classes = cluster_data['class'] # Extract the class column
unique_classes = np.unique(classes) # Unique classes

# Replace the string names with the indices of them in unique classes array
cluster_data_digit_cls = cluster_data.copy(deep=True)  
for i, label in enumerate(unique_classes):
    cluster_data_digit_cls = cluster_data_digit_cls.replace(label, i)

print(
    'Cluster dataset with indices as class names generated:\n',
    cluster_data_digit_cls.head()
)

# Generate triples with indices of sourceIP cluster, destIP cluster and class
cluster_triples = [(cluster_data_digit_cls.iloc[i][0], cluster_data_digit_cls.iloc[i][1], cluster_data_digit_cls.iloc[i][2]) for i in cluster_data_digit_cls.index]

# Use Counter method
counter_relation = Counter(cluster_triples)

# Generate the numpy array in shape (n,4) where n denotes all types of triples and the four column contains the number of records of the corresponding triples. This step may cost about 10 seconds
relation = np.concatenate((np.asarray(list(counter_relation.keys())),np.asarray(list(counter_relation.values())).reshape(-1,1)), axis=1)

# Save the dataset with counts
# pd.DataFrame(relation, columns=['sourceIP cluster', 'destIP cluster', 'class', 'counts']).to_csv('relation.csv')

# Generate data for size-encoding scatter plot
x = relation[:,0]   # Source IP cluster indices
y = relation[:,1]   # Destination IP cluster indices
area = (relation[:,3])**2/10000 # Marker size with real number of records
log_area = (np.log(relation[:,3]))**2*15    # Constrained size in logspace
colors = relation[:,2]  # Colours defined by classes

# Create new subplots figure
fig, axes = plt.subplots(1,2,figsize=(20,10))
fig.suptitle('Cluster Connections with Classifications', fontsize=20)
plt.setp(axes.flat, xlabel='sourceIP Clusters',
         ylabel='destIP Clusters')

# Scatter plot: use alpha to increase transparency
scatter = axes[0].scatter(x, y, s=area, c=colors, alpha=0.8, cmap='Paired')
axes[0].set_title('Real size encoding records')

# Legend of classes
handles, _ = scatter.legend_elements(prop='colors', alpha=0.6)
lgd2 = axes[0].legend(handles, unique_classes, loc="best", title="Classes")

# Scatter plot in logspace
scatter = axes[1].scatter(x, y, s=log_area, c=colors, alpha=0.8, cmap='Paired')
axes[1].set_title('Logspace size encoding records')

# Legend of sizes
kw = dict(prop="sizes", num=5, color=scatter.cmap(0.7), fmt="{x:.0f}",
          func=lambda s: s) 
handles, labels = scatter.legend_elements(**kw)
lgd2 = axes[1].legend(handles, labels, loc='best', title='Sizes = \n$(log(num\_records))^2*15$', labelspacing=2.5)

plt.savefig('data/Q4-relation-scatter.pdf')
plt.savefig('data/Q4-relation-scatter.jpg')
plt.show()
