# Wedding challenge

When organizing a wedding, the tricky step is to decide who sits where. This algorithm aims to find a good combination taking affinities into account and without breaking couples.

![Clustering for table attribution](https://github.com/savoga/wedding-challenge/blob/main/clustering-schema.png)

### 0- Data

The data file is used to know people's names as well as their affinities. Weights are used to put more importance on certain level of affinities. In this example, there are 4 affinity groups. We assume that the first affinity group has the highest weight since it represents the couples and those should not be broken.

Example:

Paul is married to Lisa --> affinity 1

Paul is in the same family as Greg --> affinity 2
...

### 1- Preprocessing

When an individual has no specific affinities, we create an affinity group in generating a random string that is different for all other individuals. This is equivalent of saying that this individual has an affinity group with only him/her.

### 2- K-Modes

K-modes allows to perform a first clustering with categorical variables.

### 3- Divide big clusters

The previous step is likely to give clusters that are bigger than the maximum allowed size. To solve this, we break down each big cluster into many small ones (using only couples) that we then combine altogether.

### 4- Combine clusters further

We combine clusters that could be grouped together so that we have the maximum clusters with size=TABLE_SIZE_MAX


Note (1): I am not sure whether k-modes at the beginning really plays a key role. It may be worthwhile to test without this step and starting with clusters made of couples only.
Note (2): few things can be changed to "play" with the results and end up with different combinations:
- random.seed(3): it seems changing the random generated strings can impact the clustering
- n_run: number of runs in the very first clustering
- k in range(1,20): we do only the last k combinations for the sake of speed; increasing this number could lead to different results
- changing step "Sum to TABLE_SIZE_MAX-1" to, for example, "Sum to TABLE_SIZE_MAX-2" could also provide good results
Note (3): there might be another method using the size as a constraint (maybe linear assignment?)
Note (4): for perfect results, it may be good to do final adjustments manually (e.g. swaping couples)