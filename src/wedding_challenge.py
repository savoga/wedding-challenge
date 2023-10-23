import numpy as np
import pandas as pd

import random
import string

import itertools
from itertools import combinations
from sklearn.metrics import adjusted_rand_score

import math

from kmodes.kmodes import KModes

TABLE_SIZE_MAX = 5

df = pd.read_csv('./../data/affinities.csv', header=None)
poids = [int(x) for x in df.loc[0].values[2:-1]]
columns = list(df.loc[1].values[2:-1])
dict_poids = {}
for i, column in enumerate(columns):
    dict_poids[column] = poids[i]
df = df.iloc[2:]
df['Name'] = df[0]+df[1]
df = df.drop(columns=[0,1,6])
df = df.set_index('Name')
df.index.name = None
df.columns = columns

# when there are old and young people, it makes sense to run the algorithm twice:
# once with the old people, once with the young. Then results could be merged.
# Note: in the test dataset, there are only young people
df_filter = df[(df['Affinity_2']=='no')].copy()
df_filter = df_filter.drop(columns=['Affinity_2'])
n_tables_ideal = math.ceil(df_filter.shape[0]/TABLE_SIZE_MAX)
print('Number of tables should be approximately {}'.format(n_tables_ideal))

# --------------------------------- Preprocessing -----------------------------

random.seed(0)
def generate_random_string(length=2):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

# fill cells with no values ('-') with an random and unique string
# --> this will represent one unique affinity group
random_strings_set = set()
for column in df_filter.columns:
    for idx, value in df_filter[column].items():
        if value == "-":
            # Generate a random string that is not in the set
            new_random_string = generate_random_string()
            while new_random_string in random_strings_set:
                new_random_string = generate_random_string()

            # Replace the "-" cell with the new random string
            df_filter.at[idx, column] = new_random_string
            random_strings_set.add(new_random_string)

# if a column has a higher weight, we duplicate it
def duplicate_column(df, s, n):
    if s not in df.columns:
        raise ValueError(f"Column '{s}' does not exist in the DataFrame.")
    new_df = df.copy()
    for i in range(1, n):
        new_column_name = f"{s}_{i+1}"
        new_df[new_column_name] = df[s]
    return new_df
df_poids = df_filter.copy()
for column_name in df_filter.columns:
    df_poids = duplicate_column(df_poids, column_name, dict_poids[column_name])

# ------------------------------ First clustering -----------------------------

def check_couples(df_clusters):
    couple_names = df_clusters['Affinity_1'].unique()
    for couple_name in couple_names:
        df_temp = df_clusters[df_clusters['Affinity_1']==couple_name]
        if len(df_temp)==1:
            continue
        if len(df_temp['cluster'].unique())>1:
            print('Some couples are broken!!')
            return False
    print('Couple check ok')
    return True

n_clusters_start = n_tables_ideal # we start with the ideal number of tables
n_clusters_list = n_clusters_start + np.arange(2)
found = False
df_clusters_best = None
best_sizes = None
number_full_tables_opt = 0
n_run = 5
random_state_list = np.arange(n_run)
dict_simul = {}
std_opt = np.inf
n_big_clusters_opt = np.inf
n_clusters_opt = None
random_state_opt = None
cluster_sizes_opt = None
ratio_opt = -np.inf
for n_clusters in n_clusters_list:
    print('N clusters: {}'.format(n_clusters))
    for random_state in random_state_list:
        df_clusters = df_poids.copy()
        print('Random state: {}'.format(random_state))
        km = KModes(n_clusters=n_clusters, 
                    init='Huang',
                    n_init=5, 
                    verbose=False,
                    random_state=random_state)
        clusters = km.fit_predict(df_clusters)
        df_clusters['cluster'] = km.labels_
        cluster_names = np.array(df_clusters.groupby('cluster').agg({'cluster':'count'})['cluster'].index)
        cluster_sizes = df_clusters.groupby('cluster').agg({'cluster':'count'})['cluster'].values
        print('Initial cluster sizes: {}'.format(cluster_sizes))
        std_iter = np.std(cluster_sizes)
        n_big_clusters = len(np.where(cluster_sizes>TABLE_SIZE_MAX)[0])
        ratio_iter = -n_big_clusters/std_iter
        if not check_couples(df_clusters):
            continue
        if np.max(cluster_sizes)<TABLE_SIZE_MAX+5 and std_iter<std_opt:
        # other test possibilities:
        # if ratio_iter>ratio_opt:
        # if std_iter<std_opt:
        # if n_big_clusters<n_big_clusters_opt:
        # if len(np.where(cluster_sizes>18)[0])==0:
            std_opt = std_iter
            n_big_clusters_opt = n_big_clusters
            n_clusters_opt = n_clusters
            random_state_opt = random_state
            cluster_names_opt = cluster_names
            cluster_sizes_opt = cluster_sizes
            df_clusters_opt = df_clusters
            
print('Optimal number of clusters: {}'.format(n_clusters_opt))
print('Optimal random state: {}'.format(random_state_opt))
print('Optimal cluster sizes: {}'.format(cluster_sizes_opt))

# ---------------------- Reduce big clusters ------------------------

df_clusters_divide = df_clusters_opt.copy()

big_cluster_names = [cluster_names_opt[i] for i in cluster_names_opt if cluster_sizes_opt[i]>TABLE_SIZE_MAX]
df_clusters_small = df_clusters_opt[df_clusters_opt['cluster'].isin(big_cluster_names)].copy()

# we add the couples one by one, until we are above TABLE_SIZE_MAX
for big_cluster_name in big_cluster_names:
    df_big_split_1 = pd.DataFrame(columns=df_clusters_divide.columns)
    df_big_split_2 = pd.DataFrame(columns=df_clusters_divide.columns)
    df_clusters_iter = df_clusters_divide[df_clusters_divide['cluster']==big_cluster_name]
    for couple in df_clusters_iter['Affinity_1'].unique():
        df_couple = df_clusters_opt[df_clusters_opt['Affinity_1']==couple].copy()
        if pd.concat([df_big_split_1, df_couple]).shape[0]>TABLE_SIZE_MAX and \
            pd.concat([df_big_split_2, df_couple]).shape[0]<=TABLE_SIZE_MAX:
            df_big_split_2 = pd.concat([df_big_split_2, df_couple])
            continue
        if pd.concat([df_big_split_1, df_couple]).shape[0]>TABLE_SIZE_MAX and \
                pd.concat([df_big_split_2, df_couple]).shape[0]>TABLE_SIZE_MAX:
            df_clusters_divide.loc[df_big_split_1.index, 'cluster'] = df_clusters_divide['cluster'].max()+1
            df_clusters_divide.loc[df_big_split_2.index, 'cluster'] = df_clusters_divide['cluster'].max()+1
            break
        df_big_split_1 = pd.concat([df_big_split_1, df_couple])
    name_cluster_split_1 = df_clusters_divide['cluster'].max()+1   
    df_clusters_divide.loc[df_big_split_1.index, 'cluster'] = name_cluster_split_1
    name_cluster_split_2 = df_clusters_divide['cluster'].max()+1   
    df_clusters_divide.loc[df_big_split_2.index, 'cluster'] = name_cluster_split_2

cluster_sizes = df_clusters_divide.groupby('cluster').agg({'cluster':'count'})['cluster'].values
cluster_names = np.array(df_clusters_divide.groupby('cluster').agg({'cluster':'count'})['cluster'].index)

print('Cluster sizes after grouping by couples: {}'.format(cluster_sizes))

#%%

# returns indexes of clusters that could be grouped because they sum to 10
def find_combinations(cluster_sizes, target_sum=10):
    result = []
    for r in range(1, len(cluster_sizes) + 1):
        combinations = list(itertools.combinations(enumerate(cluster_sizes), r))
        for combination in combinations:
            combination_indexes = [x[0] for x in combination]
            combination_values = [x[1] for x in combination]
            if sum(combination_values) == target_sum:
                if len(combination_values)==1:
                    continue
                result.append(combination_indexes)
    return result

# returns groups of clusters that have no cluster in common and that could be grouped together
# ideally, we would like to have a group of clusters with the largest number of different clusters
# as possible
def find_unique_combinations(list_of_list_of_clusters):
    result = []
    for r in reversed(range(1, len(list_of_list_of_clusters) + 1)):  
    # for r in range(1, len(list_of_list_of_clusters) + 1):    
        for comb in combinations(list_of_list_of_clusters, r):
            combined = []
            for c in comb:
                combined.extend(c)
            # Check if the combined list has unique elements
            if len(set(combined)) == len(combined):
                result.append(list(comb))
                # return list(comb)
    return result

# returns best combination of clusters i.e the one that groups the most clusters
def get_longest_list(list_of_list_of_clusters):
    return max(list_of_list_of_clusters, key=len)

# we give to all relevant clusters the same name as the first one from the group 
def group_clusters(df_clusters, best_combination):
    df_clusters_res = df_clusters.copy()
    for cluster_group in best_combination:
        for cluster_name in cluster_group[1:]:
            df_clusters_res.loc[df_clusters_res['cluster']==cluster_name,'cluster'] = cluster_group[0]
    return df_clusters_res

number_full_tables_opt = 0
best_sizes = None
df_clusters_best = None

df_res = []

for k in range(1,20):

    # ------------------------- Sum to TABLE_SIZE_MAX -------------------------
    
    cluster_sizes = df_clusters_divide.groupby('cluster').agg({'cluster':'count'})['cluster'].values
    cluster_names = np.array(df_clusters_divide.groupby('cluster').agg({'cluster':'count'})['cluster'].index)
    if len(np.where(cluster_sizes>TABLE_SIZE_MAX)[0])>0:
        raise Exception('One cluster is too big!')
    print('No big clusters anymore...')
    
    print('Finding all combinations that sum to {}...'.format(TABLE_SIZE_MAX))
    all_combinations_1 = find_combinations(cluster_sizes, target_sum=TABLE_SIZE_MAX)
    if all_combinations_1==[]: 
        print()
        continue
    
    print('Finding all distinct combinations...')
    # we do only the last k combinations for the sake of speed
    all_combinations_1_unique = find_unique_combinations(all_combinations_1[-k:])
    
    print('Finding the best combination...')
    best_combination_indexes = get_longest_list(all_combinations_1_unique)
    best_combination_names = [[cluster_names[idx] for idx in combo_idx] for combo_idx in best_combination_indexes]
    df_clusters_grouped_1 = group_clusters(df_clusters_divide, best_combination_names)
    cluster_sizes_grouped_1 = df_clusters_grouped_1.groupby('cluster').agg({'cluster':'count'})['cluster'].values
    print('Cluster sizes after combining clusters that sum to {}: {}'.format(TABLE_SIZE_MAX, cluster_sizes_grouped_1))
    
    # ---------------------------- Sum to TABLE_SIZE_MAX-1 --------------------
    
    cluster_names = list(df_clusters_grouped_1.groupby('cluster').agg({'cluster':'count'})['cluster'].index)
    cluster_sizes = df_clusters_grouped_1.groupby('cluster').agg({'cluster':'count'})['cluster'].values
    print('Finding all combinations that sum to {}...'.format(TABLE_SIZE_MAX-1))
    all_combinations_2 = find_combinations(cluster_sizes, target_sum=TABLE_SIZE_MAX-1)
    if all_combinations_2==[]: 
        print()
        continue
    print('Finding all distinct combinations...')
    all_combinations_2_unique = find_unique_combinations(all_combinations_2[-k:])
    print('Finding the best combination...')
    best_combination_indexes = get_longest_list(all_combinations_2_unique)
    best_combination_names = [[cluster_names[idx] for idx in combo_idx] for combo_idx in best_combination_indexes]
    df_clusters_grouped_2 = group_clusters(df_clusters_grouped_1, best_combination_names)
    cluster_sizes_grouped_2 = df_clusters_grouped_2.groupby('cluster').agg({'cluster':'count'})['cluster'].values
    print('Cluster sizes after combining clusters that sum to {}: {}'.format(TABLE_SIZE_MAX-1, cluster_sizes_grouped_2))
    
    # # ------------------------ Sum to TABLE_SIZE_MAX ------------------------
    
    cluster_sizes = df_clusters_grouped_2.groupby('cluster').agg({'cluster':'count'})['cluster'].values
    cluster_names = np.array(df_clusters_grouped_2.groupby('cluster').agg({'cluster':'count'})['cluster'].index)
    
    print('Finding all combinations that sum to {}...'.format(TABLE_SIZE_MAX))
    all_combinations_3 = find_combinations(cluster_sizes, target_sum=TABLE_SIZE_MAX)
    if all_combinations_3!=[]:
    
        print('Finding all distinct combinations...')
        all_combinations_3_unique = find_unique_combinations(all_combinations_3[-k:])
        
        print('Finding the best combination...')
        best_combination_indexes = get_longest_list(all_combinations_3_unique)
        best_combination_names = [[cluster_names[idx] for idx in combo_idx] for combo_idx in best_combination_indexes]
        df_clusters_grouped_3 = group_clusters(df_clusters_grouped_2, best_combination_names)
        cluster_sizes_grouped_3 = df_clusters_grouped_3.groupby('cluster').agg({'cluster':'count'})['cluster'].values
        print('Cluster sizes after combining AGAIN clusters that sum to {}: {}'.format(TABLE_SIZE_MAX, cluster_sizes_grouped_3))
    else:
        df_clusters_grouped_3 = df_clusters_grouped_2.copy()
        cluster_sizes_grouped_3 = cluster_sizes_grouped_2.copy()
    
    df_clusters_final = df_clusters_grouped_3.sort_values(by='cluster').copy()
    cluster_sizes_final = cluster_sizes_grouped_3.copy()
    
    if not check_couples(df_clusters_final):
        print()
        continue
    
    number_full_tables1 = len(np.where(cluster_sizes_final==TABLE_SIZE_MAX)[0])
    number_full_tables2 = len(np.where(cluster_sizes_final==TABLE_SIZE_MAX-1)[0])
    number_full_tables = number_full_tables1 + number_full_tables2
    ratio_full_tables = round(number_full_tables/len(cluster_sizes_final)*100,2)
    
    print('RATIO OF FULL TABLES: {}%'.format(ratio_full_tables))
    print()
    
    if np.min(cluster_sizes_final)>=TABLE_SIZE_MAX-2 and number_full_tables>number_full_tables_opt:
        number_full_tables_opt = number_full_tables
        best_sizes = cluster_sizes_final.copy()
        df_clusters_best = df_clusters_final[['Affinity_1','Affinity_3','Affinity_4','cluster']].copy()
        already_saved = False
        clusters = df_clusters_best['cluster'].values
        for i in range(len(df_res)):
            clusters_iter = df_res[i]['cluster'].values
            ari = adjusted_rand_score(clusters, clusters_iter)
            if ari==1:
                already_saved = True
                break
        if not already_saved:
            df_res.append(df_clusters_best)

print()
# if not found:
#     print('No good cluster found')
print('best cluster sizes: {}'.format(best_sizes))

df_clusters_best_adjust = df_clusters_best.copy()
# df_clusters_best_adjust = group_clusters(df_clusters_best, [[6,3]]).copy()

for idx, cluster_name in enumerate(df_clusters_best_adjust['cluster'].unique()):
    print('Table {}: {} personnes'.format(idx+1, df_clusters_best_adjust[df_clusters_best_adjust['cluster']==cluster_name].shape[0]))

#%%

excel_writer = pd.ExcelWriter('./../output/table_proposals_test.xlsx', engine='xlsxwriter')
empty_rows = 3
for index, df in enumerate(df_res[:3]):
    start_row = index * (df.shape[0] + empty_rows)
    df.to_excel(excel_writer, sheet_name='Sheet1', startrow=start_row)
workbook  = excel_writer.book
worksheet = excel_writer.sheets['Sheet1']
for i, col in enumerate(df.columns):
    column_len = max(df[col].astype(str).apply(len).max(), len(col)) + 2
    worksheet.set_column(i, i, column_len)  # Set the width for all columns
excel_writer.save()
print("Dataframes exported to Excel successfully.")




