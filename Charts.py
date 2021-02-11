#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json


charts_dir='figures'

def is_valid(string):
    if str(string)=='nan':
        return 0
    elif string:
        return 1
    else:
        return 0
    
    


def string_replacer(string, mapping):
    '''Replace the string by looking it up in the mapping.
    If string is not in mapping, return the string'''
    response=mapping.get(string)
    if response:
        return response
    else:
        return string
    

def ecoapple_fix(string):
    '''Fix for combining the Red Tomato Eco with Eco-Apple as one Cert'''
    return string_replacer(string, {'Red Tomato Eco':"Red Tomato Eco Apple"})


def multi_remove(string, pieces):
    '''Remove an arbitrary number of substrings from a string'''
    for piece in pieces:
        string=string.replace(piece, '')
    return string

def column_switcher(row, old_col, new_col, string):
    if str(string) in str(row[old_col]):
        pieces=[i for i in str(row[old_col]).split(',') if string in i]
        print(str(row[old_col]))
        row[new_col]=' ,'.join(pieces) +' , '+str(row[new_col])
        row[old_col]=multi_remove(str(row[old_col]), pieces)
        
    return row

def buffer_switcher(row): 
    row=column_switcher(row, 'Agronomic Practices', 'Pesticide Practices', 'spray')
    row=column_switcher(row, 'Agronomic Practices', 'Pesticide Practices', 'pesticides')
    row=column_switcher(row, 'Agronomic Practices', 'Pesticide Practices', 'buffer zones')
    return row

def performance_switcher(row): 
    row= column_switcher(row, 'Performance Standard', 
                           'Non-Quantitative Performance Standards', 
                          'vague')
    return column_switcher(row, 'Performance Standard', 
                           'Non-Quantitative Performance Standards', 
                          'free')

def threshold_fixer(row):
    if 'threshold' in row['Monitoring']:
        row['Threshold']=row['Monitoring']
    return row






# In[7]:

    

df=pd.read_csv('all_data.csv')
df.fillna('', inplace=True)
df=df[df['CertName']!='Eco Apple Stonefruit']
df['CertName']=df['CertName'].apply(ecoapple_fix)


df=df.apply(buffer_switcher, axis=1)
df=df.apply(performance_switcher, axis=1)
df=df.apply(threshold_fixer, axis=1)





df['MOA rotation']=df['Pesticide Practices'].str.contains('moa rotation')
df['Sprayer Calibration']=df['Pesticide Practices'].str.contains('calibration')




df['crop rotation']=df['Agronomic Practices'].str.contains('rotation')
df['cover crop']=df['Agronomic Practices'].str.contains('cover crop')


df['weather model']=df['Monitoring'].str.contains('dd model|weather model', regex=True)
df['pesticide containers']=df['Materials/Waste Mgmt'].str.contains('containers')
df['ppe']=df['Worker Safety'].str.contains('ppe')




df=df[[c for c in df.columns if 'Unnamed' not in c]]
columns=['CertName']+[c for c in df.columns if c!="CertName"]

df=df[columns]


# In[12]:


df['requirement']=(df['Required/Core or Improvement'].str.lower()=='required') | (df['Required/Core or Improvement'].str.lower()=='level 2')
df['scorecard']=(df['Required/Core or Improvement'].str.lower()=='scorecard') | (df['Required/Core or Improvement']=='C')
df['improvement']=(df['Required/Core or Improvement'].str.lower()=='improvement') | (df['Required/Core or Improvement']=='B')



counts=pd.concat([df[df.columns[:9]],df[df.columns[9:]].applymap(is_valid)], axis=1)
counts=pd.concat([df[df.columns[:9]],df[df.columns[9:]].applymap(is_valid)], axis=1)
gb=counts.groupby('CertName')

data=gb.sum()
data['total']=gb.count().max(axis=1)

for column in data.columns[:-1]:
    data[column]=data[column]/data['total']
data.drop(columns='total', inplace=True)

counts=gb.sum()

results={}
for column in data.columns:
    results[column]=data[column].astype(bool).sum()
    
results





struct=counts[['requirement', 'improvement', 'scorecard', 'Performance Standard']]



# In[21]:


prim_scorecard=struct[struct['requirement']<=3].index.to_list()


prim_req=struct[(struct['scorecard']==0) & (struct['improvement']==0)].index.to_list()


# In[23]:


mixed=[i for i in struct.index if (i not in prim_scorecard) & (i not in prim_req)]


# In[24]:


for li in [prim_scorecard, prim_req, mixed]:
    print(len(li))


# In[25]:


results=[len(li) for li in [prim_scorecard, prim_req, mixed]]
plt.pie(results, labels=['Score Card', "Requirements", 'Mixed'])
print(results)
plt.title('Types of Certifications')
plt.savefig(os.path.join(charts_dir, 'struct_pie.png'))
plt.show()


#classifiers of certificationsL
single_commodity=['RTRS', 'CAFE', 'RSPO', 'LIVE', 'SIP', 'LODI Rules',
       'FlorVerde', 'Red Tomato Eco Apple', 'Protected Harvest Citrus',
       'BCI', 'BloomCheck', 'CMIA', 'HealthyGrown', 'TrueEarth']

multi_commodity=[ 'Sustainable Food Group', 'Global GAPS', 'Food Alliance',
       'Rainforest Alliance', 'Fair Trade USA', "UTZ"]

regional_designation=['SIP', 'LODI Rules', 'BloomCheck', 
                      'Red Tomato Eco Apple', 'HealthyGrown', 'LIVE', 'Protected Harvest Citrus', 'TrueEarth']

global_south=['BCI', "CMIA", 'CAFE', 'RSPO', 'RTRS', 
              'Fair Trade USA', 'Rainforest Alliance', 
              "FlorVerde", 'Global GAPS', "UTZ"]

domestic=['Sustainable Food Group',
                       'Food Alliance']


# In[ ]:






def intersect_lists(l1, l2):
    '''Return a list of elements that are in both l1 and l2'''
    return list(set(l1).intersection(set(l2)))


# In[31]:


from itertools import combinations




dom_single_com=intersect_lists(single_commodity, domestic)
dom_multi_com=intersect_lists(multi_commodity, domestic)
reg_single_com=intersect_lists(single_commodity, regional_designation)
south_single_com=intersect_lists(single_commodity, global_south)
south_multi_com=intersect_lists(multi_commodity, global_south)

categories=[dom_multi_com, 
            #dom_single_com, 
            reg_single_com, 
            south_single_com, 
            south_multi_com]

# In[33]:


labels=['Domestic, Multi-Commodity', 
        #'Domestic, Single-Commodity',
        "Regional Designation, Single Commodity", 
        'Global South, Single Commodity',
       'Global South, Multi-Commodity']


# In[34]:


results=[len(li) for li in categories]
plt.pie(results, labels=labels)
print(results)
plt.title('Types of Certifications in the Sample')
plt.savefig(os.path.join(charts_dir, ))


# In[35]:

df=df[df['CertName']!='']
re_index=[x for y in categories for x in y]
print(re_index)

# In[77]:


def re_indexer(df):
    '''Change the order of certs so that they are grouped by geo/commodity scope.'''
    df=df.reset_index()
    df=df[df['CertName']!='']
    df['reindex_num']=df['CertName'].apply(lambda x: re_index.index(x)).to_list()
    df=df.sort_values(by='reindex_num')
    return df.drop(columns='reindex_num')



# In[38]:


counts=re_indexer(counts)
data=re_indexer(data)




# In[39]:




commodities={'coffee': ['CAFE'], 
            'potatoes': ['HealthyGrown'],
            'flowers': ['FlorVerde', 'BloomCheck'],
            'cotton': ['BCI', 'CMIA'],
            'Palm Oil': ['RSPO'],
            'wine/grapes':['LIVE', 'SIP', 'LODI Rules'],
            'Citrus': ['Protected Harvest Citrus'],
            'Apple/StoneFruit': ['Red Tomato Eco Apple', 'TrueEarth']}


# In[42]:

plt.close()
results=[len(v) for v in commodities.values()]
plt.pie(results, labels=commodities.keys())
plt.title('Single Commodity Certifications')
plt.savefig(os.path.join(charts_dir, 'cropspie.png'))



# In[47]:
category_colors=['salmon', 
                 #'goldenrod',  
                 'skyblue', 
                 'palegreen', 
                 'plum']

colors=[]
for cat, color in zip(categories, category_colors):
    colors+=[color]*len(cat)
labels=['US, Multi-Comm',
#'US, Single-Comm',
'Regional, 1-Comm',
 'Low-Inc, 1-Comm',
'Low-Inc, Multi-Comm']


# In[50]:


def uniq(inp):
    '''Return a list of all unique values in iterable, while keeping order'''
    output = []
    for x in inp:
        if x not in output:
            output.append(x)
    return output


import matplotlib.patches as mpatches


abbrv_dict={}
for certName in re_index:
    if len(certName)<5:
        abbrv_dict[certName]=certName
    elif certName in ['FlorVerde', 'HealthyGrown', 'Protected Harvest Citrus',
                      'BloomCheck', 'TrueEarth', 'Sustainable Food Group',
                      'Food Alliance', 'Fair Trade USA', 'Rainforest Alliance'
                      ]:
        abbrv_dict[certName]=''.join([c for c in certName if c.upper()==c and c])
    else:
        print(certName)

abbrv_dict['Global GAPS']='GGAPS'
abbrv_dict['LODI Rules']='LODI'
abbrv_dict['Red Tomato Eco Apple']='EA'

#%%
abbrvs=[]
for name in re_index:
    abbrvs.append(abbrv_dict[name])



#%%

data['scorecard/improvement']=data['scorecard']+data['improvement']




# In[767]:


summary=pd.DataFrame(np.array([data[data.index.isin(cat)].astype(bool).mean() 
                               for cat in categories]).T, 
                     index=data.columns, columns=labels)


summary['Category']=summary.index

lf=summary.melt(id_vars='Category')


# In[772]:


lf.rename(columns={'value':'proportion', 'variable': "Cert Type"}, inplace=True)



lf=lf[lf['Category'].str.contains('Applies')==False]




lf1=lf.iloc[:6]



'''
lf1=lf[lf['Category'].isin(lf['Category'].unique()[:6])]
sns.barplot(x='Category', y='proportion', hue='Cert Type', data=lf1)
plt.xticks(rotation=90)


# In[777]:


lf2=lf[lf['Category'].isin(lf['Category'].unique()[6:12])]
sns.barplot(x='Category', y='proportion', hue='Cert Type', data=lf2)
plt.xticks(rotation=90)


# In[778]:


lf3=lf[lf['Category'].isin(lf['Category'].unique()[12:18])]
sns.barplot(x='Category', y='proportion', hue='Cert Type', data=lf3)
plt.xticks(rotation=90)


# In[779]:


lf4=lf[lf['Category'].isin(lf['Category'].unique()[18:])]
sns.barplot(x='Category', y='proportion', hue='Cert Type', data=lf4)
plt.xticks(rotation=90)
'''




crop_1=data[data.index.isin(single_commodity)].astype(bool)
any_crop=data[data.index.isin(multi_commodity)].astype(bool)


# In[254]:


by_com=pd.DataFrame(np.array([crop_1.mean(), any_crop.mean()]).T, index=data.columns, 
             columns=['Single Commodity', 'Multi-Commodity'])




for name in df['Performance Standard'].unique():
    if name:
        print(df[df['Performance Standard']==name]['CertName'].iloc[0])
        print (name)
        


counts=pd.concat([df[df.columns[:9]],df[df.columns[9:]].applymap(is_valid)], axis=1)
counts=pd.concat([df[df.columns[:9]],df[df.columns[9:]].applymap(is_valid)], axis=1)
gb=counts.groupby(['CertName', 'requirement', 'improvement'])

data=gb.sum()


# In[1204]:


data.reset_index()['requirement']


# In[86]:


counts=pd.concat([df[df.columns[:9]],df[df.columns[9:]].applymap(is_valid)], axis=1)
counts=pd.concat([df[df.columns[:9]],df[df.columns[9:]].applymap(is_valid)], axis=1)
gb=counts.groupby(['CertName', 'requirement', 'improvement'])

data=gb.sum()



req_indexer=data.reset_index()['requirement']==1
data_req=data[req_indexer.tolist()]
improve_indexer=(data.reset_index()['improvement']==1)
data_impr=data[improve_indexer.tolist()]

non_reqindexer=(req_indexer+improve_indexer)==0
data_nonreq=data[non_reqindexer.tolist()]


#%%%
data_sets=[data_req, data_impr, data_nonreq]

def append_null_rows(df, col_name, row_names):
    '''Add empty data for columns without data.'''
    rows=[]
    for row_name in [r for r in row_names if (r not in df[col_name].unique() and r)]:
        dic={col: 0 for col in df.columns }
        dic.update({col_name:row_name})
        rows.append(dic)
    return df.append(rows).reset_index()



def transform_for_plotting(ds):
    print(ds.shape[0])
    ds.reset_index(inplace=True)
    print(ds.shape[0])
    ds=append_null_rows(ds, 'CertName', [c for c in counts['CertName'].unique() ])
    print(ds.shape[0])
    ds.drop(columns=['index'], inplace=True)
    print(ds.shape[0])
    ds=re_indexer(ds)
    return ds

data_req=transform_for_plotting(data_req)
data_nonreq=transform_for_plotting(data_nonreq)
data_impr=transform_for_plotting(data_impr)
    

# In[87]:




# In[88]:
'''
data_req=append_null_rows(data_req, 'CertName', [c for c in counts['CertName'].unique() ])
data_nonreq=append_null_rows(data_nonreq, 'CertName', [c for c in counts['CertName'].unique() ])


data_req.drop(columns=['requirement', 'index'], inplace=True)
data_nonreq.drop(columns=['requirement', 'index'], inplace=True)

#data_nonreq.drop(0, inplace=True)




data_req=re_indexer(data_req)
data_nonreq=re_indexer(data_nonreq)
'''

cols=['Agronomic Practices',
 'Biosecurity/Sanitation',
 'Planning',
 'Education/training']



# In[97]:
patches=[]
for color, label in zip(uniq(colors), labels):
    print(color)
    patches.append(mpatches.Patch(color=color, label=label))

patches2=patches+[ mpatches.Patch(label='Score-Card', hatch=r'\\\\', edgecolor='black', facecolor='white'),
                  mpatches.Patch(label='Improvement to Mandate', edgecolor='black', hatch='O', facecolor='white'),
                  mpatches.Patch(label='Mandatory', facecolor='white', edgecolor='black'), 
                 
                 ]

patches_p=patches+[ mpatches.Patch(label='Not Registered by EPA', hatch=r'\\\\', edgecolor='black', facecolor='white'),
                  mpatches.Patch(label='Registered by EPA', facecolor='white', edgecolor='black'), 
                 ]
# In[98]:




def stackplot_col(ax, col, dfs, hatches):
    b=np.zeros(dfs[0].shape[0])
    for df, hatch in zip(dfs, hatches):
        
        ax.bar(x=abbrvs, bottom=b, height=df[col], color=colors, edgecolor='black', hatch=hatch)
        b=df[col].to_numpy()+b
        print(b)
        plt.setp(ax.get_xticklabels(), rotation=90)
        ax.set_title(col, size=20)
    return ax

def stack_plotter(cols, *dfs, hatches):
    
    fig, axes=plt.subplots(1,len(cols), figsize=(20,5))
    #axes=[a for l in axes for a in l]
    #axes[0].legend(handles=patches)
    #axes[0].axis('off')
    if len (cols)>1:
        for i, column in enumerate(cols):
            ax=axes[i]
            stackplot_col(ax, column, dfs, hatches)
    elif len (cols)==1:
        stackplot_col(axes, cols[0], dfs, hatches)
    else:
        raise ValueError
        
    return fig, axes

def std_stack_plot(cols,  df1, df2, df3, save_name):
    fig, axes=stack_plotter(cols,  df1, df2, df3, hatches=['',  'O', r'\\\\',])
    axes[0].set_ylabel('Number of Criteria', size=20)
    plt.legend(handles=patches2, markerscale=4)
    plt.tight_layout()
    plt.savefig(os.path.join('figures',save_name))


#%%
def p_df(index, val_dict, pest_cat):
    data=[val_dict.get(i) for i in index]
    df=pd.DataFrame({f"{pest_cat} Pesticides".title(): data, 'CertName': index})
    df=re_indexer(df)
    return df

def make_pesticide_dfs(index):
    dfs=[]
    for category in ('banned', 'restricted'):
        pest_lists=json.loads(open(f'cert_pest_lists_{category}.txt').read())
        pesticide_dfs=[p_df(index, pest_lists[key], category) for key in ('num_epa_reg', 'num_not_reg_epa')]
        dfs.append(pesticide_dfs)
    return dfs
                            


pest_dfs=make_pesticide_dfs(counts['CertName'].unique())

def pest_list_plotter(pest_dfs):
    fig, axes=plt.subplots(1,2, figsize=(20,5))
    for group, string , ax in zip(pest_dfs, ('Banned', 'Restricted'), axes):
        ax=stackplot_col(ax, f'{string} Pesticides', group, ['', r'\\\\'])
    axes[0].set_ylabel('Number of Pesticides', size=20)
    plt.legend(handles=patches_p, markerscale=4)
    plt.tight_layout()
    axes[0].scatter(6, 150, marker='*', s=350)
    
    
    plt.savefig(os.path.join('figures', 'pesticide_plotter.png'))


pest_list_plotter(pest_dfs)
#%%
cols=['Record-Keeping',
 'Planning',
 'Education/training',
 'Monitoring',]
std_stack_plot(cols, data_req, data_impr, data_nonreq,  "admin.png")





std_stack_plot([
 'Worker Safety',
 'Materials/Waste Mgmt'], data_req, data_impr, data_nonreq,  "worker.png")


cols=['Pesticide Practices', 'Prohibited Practice',  'Agronomic Practices','Biosecurity/Sanitation',]
std_stack_plot(cols,data_req, data_impr, data_nonreq,  "farm_practices.png")





plt.legend(handles=patches2)
ax=plt.gca()
ax.axis('off')





# In[ ]:




