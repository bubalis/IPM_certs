"""
@author: bdube

This script generates charts for analyzing the data of certifications.
"""

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from collections import OrderedDict


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
    '''Move data in row that contains {string} from old_col
    to new_col.'''
    
    if str(string) in str(row[old_col]):
        pieces=[i for i in str(row[old_col]).split(',') if string in i]
        #print(str(row[old_col]))
        
        row[new_col]=' ,'.join(pieces) +' , '+str(row[new_col])
        row[old_col]=multi_remove(str(row[old_col]), pieces)
        
    return row


def buffer_switcher(row): 
    '''Switch data from  Agronomic Practices to Pesticide Practices'''
    for string in ['spray', 'pesticide', 'buffer']:    
        row = column_switcher(row, 'Agronomic Practices', 
                        'Pesticide Practices', string)
    return row

def performance_switcher(row):
    '''Change over data that contains "free" or
    "vague" from performance to non-quantitative'''
    row = column_switcher(row, 'Performance Standard', 
                           'Non-Quantitative Performance Standards', 
                          'vague')
    return column_switcher(row, 'Performance Standard', 
                           'Non-Quantitative Performance Standards', 
                          'free')

def threshold_fixer(row):
    '''Make data for threshold column from Monitoring column
    if the word threshold appears. '''
    if 'threshold' in row['Monitoring']:
        row['Threshold']=row['Monitoring']
    return row



def append_null_rows(df, col_name, row_names):
    '''Add empty data for columns without data.'''
    rows=[]
    for row_name in [r for r in row_names if (r not in df[col_name].unique() and r)]:
        dic={col: 0 for col in df.columns }
        dic.update({col_name:row_name})
        rows.append(dic)
    return df.append(rows).reset_index()



def transform_for_plotting(ds):
    '''Add null rows to dataset for where x val is empty.
    drop index column.
    Re-order dataset to the '''
    #print(ds.shape[0])
    ds.reset_index(inplace=True)
    #print(ds.shape[0])
    ds=append_null_rows(ds, 'CertName', [c for c in counts['CertName'].unique() ])
    #print(ds.shape[0])
    ds.drop(columns=['index'], inplace=True)
    
    ds=re_indexer(ds, re_index_list)
    return ds



def re_indexer(df, re_index_list):
    '''Change the order of certs so that they are grouped by geo/commodity scope.'''
    df=df.reset_index()
    df=df[df['CertName']!='']
    df['reindex_num']=df['CertName'].apply(lambda x: re_index_list.index(x)).to_list()
    df=df.sort_values(by='reindex_num')
    return df.drop(columns='reindex_num')


def intersect_lists(l1, l2):
    '''Return a list of elements that are in both l1 and l2'''
    return list(set(l1).intersection(set(l2)))


def uniq(inp):
    '''Return a list of all unique values in iterable, while keeping order'''
    output = []
    for x in inp:
        if x not in output:
            output.append(x)
    return output


def stackplot_col(ax, col, dfs, hatches, **plot_kwargs):
    '''Plot a single ax as a stack plot.'''
    
    b=np.zeros(dfs[0].shape[0])
    for df, hatch in zip(dfs, hatches):
        
        ax.bar(x=abbrvs, bottom=b, height=df[col], color=colors, edgecolor='black', 
               hatch=hatch, **plot_kwargs)
        b=df[col].to_numpy()+b
        print(b)
        plt.setp(ax.get_xticklabels(), rotation=90)
        ax.set_title(col, size=20)
    return ax

def stack_plotter(cols, *dfs, hatches, **plot_kwargs):
    '''Make a plot of multiple stacked bar charts next to one another.
    Each column is plotted on its own ax.
    *dfs represents the data to be stacked. 
    hatches represent the hatches for each piece of the stack plot.
    **plot_kwargs: kwargs for the ax.bar function. 
    '''
    
    ymax = np.sum([np.array(df[cols]) for df in dfs], axis=0).max()*1.1
    fig, axes=plt.subplots(1,len(cols), figsize=(20,5))
    #axes=[a for l in axes for a in l]
    #axes[0].legend(handles=patches)
    #axes[0].axis('off')
    if len (cols)>1:
        for i, column in enumerate(cols):
            ax=axes[i]
            stackplot_col(ax, column, dfs, hatches,  **plot_kwargs)
            ax.set_ylim(0, ymax)
    elif len (cols)==1:
        stackplot_col(axes, cols[0], dfs, hatches, **plot_kwargs)
        axes.set_ylim(0, ymax)
    else:
        raise ValueError
        
    return fig, axes

def std_stack_plot(cols,  df1, df2, df3, save_name, handles, **plot_kwargs):
    '''Make a standard stack plot of the data in [cols],
    stacking tha values of df1, df2, and df3 on top of each other.
    save the figure as save_name.
    '''
    
    fig, axes=stack_plotter(cols,  df1, df2, df3, hatches=['',  'O', r'\\\\',])
    axes[0].set_ylabel('Number of Criteria', size=20)
    plt.legend(handles=handles, markerscale=4)
    plt.tight_layout()
    plt.savefig(os.path.join('figures',save_name))


#%%
def p_df(index, val_dict, pest_cat, re_index_list):
    '''Make a dataframe of numbers of pesticide counts.'''
    data=[val_dict.get(i) for i in index]
    df=pd.DataFrame({f"{pest_cat} Pesticides".title(): data, 'CertName': index})
    df=re_indexer(df, re_index_list)
    return df

def make_pesticide_dfs(index, re_index_list):
    '''Assemble dfs for pesticides.'''
    dfs=[]
    for category in ('banned', 'restricted'):
        pest_lists=json_from_file(
            os.path.join('data', f'cert_pest_lists_{category}.txt'))
        pesticide_dfs=[p_df(index, pest_lists[key], category, re_index_list) for key in ('num_epa_reg', 'num_not_reg_epa')]
        dfs.append(pesticide_dfs)
    return dfs
                            
def pest_list_plotter(pest_dfs, handles):
    '''Plotter function for pesticide data.'''
    
    fig, axes=plt.subplots(1,2, figsize=(20,5))
    for group, string , ax in zip(pest_dfs, ('Banned', 'Restricted'), axes):
        ax=stackplot_col(ax, f'{string} Pesticides', group, ['', r'\\\\'])
    axes[0].set_ylabel('Number of Pesticides', size=20)
    plt.legend(handles=handles, markerscale=4)
    plt.tight_layout()
    axes[0].scatter(6, 150, marker='*', s=350)
    
    
    plt.savefig(os.path.join('figures', 'pesticide_plotter.png'))



def clean_data(df):
    '''Clean the loaded dataframe.'''
    df.fillna('', inplace=True)
    df=df[df['CertName']!='']
    df=df[df['CertName']!='Eco Apple Stonefruit']
    df['CertName']=df['CertName'].apply(ecoapple_fix)
    df=df.apply(buffer_switcher, axis=1)
    df=df.apply(performance_switcher, axis=1)
    df=df.apply(threshold_fixer, axis=1)
    
    return df[[c for c in df.columns if 'Unnamed' not in c]]

def add_columns(df):
    '''Add additional calculated columns to dataframe based on 
    text the elements of the data.'''
    
    df['MOA rotation']=df['Pesticide Practices'].str.contains('moa rotation')
    df['Sprayer Calibration']=df['Pesticide Practices'].str.contains('calibration')    
    df['crop rotation']=df['Agronomic Practices'].str.contains('rotation')
    df['cover crop']=df['Agronomic Practices'].str.contains('cover crop')    
    df['weather model']=df['Monitoring'].str.contains('dd model|weather model', regex=True)
    df['pesticide containers']=df['Materials/Waste Mgmt'].str.contains('containers')
    df['ppe']=df['Worker Safety'].str.contains('ppe')
    
    df['requirement']=(df['Required/Core or Improvement'].str.lower()=='required') | (df['Required/Core or Improvement'].str.lower()=='level 2')
    df['scorecard']=(df['Required/Core or Improvement'].str.lower()=='scorecard') | (df['Required/Core or Improvement']=='C')
    df['improvement']=(df['Required/Core or Improvement'].str.lower()=='improvement') | (df['Required/Core or Improvement']=='B')
    
    return df

def split_on_structure(counts):
    '''Return dictionary of different certification structures.
    keys: structure labels
    values: List of certification names classified as that structure'''
    out_dict = {}
    struct=counts[['requirement', 'improvement', 'scorecard', 'Performance Standard']]
    
    out_dict['Score Card'] = struct[struct['requirement']<=3].index.to_list()
    
    out_dict['Requirements'] = struct[(struct['scorecard']==0) & (struct['improvement']==0)].index.to_list()
    
    out_dict['Mixed'] = [i for i in struct.index if 
                         (i not in out_dict['Score Card']) & 
                         (i not in out_dict['Requirements'])]
    return out_dict



def plot_cert_structures(dic):
    results=[len(li) for li in dic.values()]
    plt.pie(results, labels=dic.keys())
    print(results)
    plt.savefig(os.path.join(charts_dir, 'struct_pie.png'))
    plt.show()


def make_counts_df(df):
    '''Return a dataframe of counts of different column categories, 
    by certification.'''
    counts=pd.concat([df[df.columns[:9]],df[df.columns[9:]].applymap(is_valid)], axis=1)
    counts['CertName'] = df ['CertName']
    gb=counts.groupby('CertName')
    counts=gb.sum()
    counts['total']=gb.count().max(axis=1)
    return counts

def set_abbrvs(re_index_list, corrections):
    '''Return a list of abbreviations for plotting.
    Make them in the same order as re_index_list'''
    abbrv_dict={}
    for certName in re_index_list:
        if len(certName)<5:
            abbrv_dict[certName]=certName
        else:
            abbrv_dict[certName]=''.join(
                            [c for c in certName if c.upper()==c and c])
    
    abbrv_dict.update(corrections)
    abbrvs=[abbrv_dict[name] for name in re_index_list]
    return abbrvs, abbrv_dict

def group_certs(c_dict):
    category_dict = OrderedDict()
    for key1 in ['Domestic', 'Regional Designation', 'Global South']:
        for key2 in ['Single Commodity', 'Multi-Commodity']:
            category_dict[f'{key1}, {key2}'] = intersect_lists(
                                                c_dict[key1],
                                                c_dict[key2])
    
    return {k:v for k,v in category_dict.items() if v}

def pie_from_dict(di):
    '''Make a piechart from a dictionary.
    dic: keys: names of categories
    values: the names that fit those categories.'''
    ax = plt.pie([len (v) for v in di.values()], labels = di.keys())
    return ax


def cert_type_pie(category_dict):
    ax = pie_from_dict(category_dict)
    plt.title('Types of Certifications in the Sample')
    plt.savefig(os.path.join(charts_dir, 'Cert_type_pie'))
    plt.show()

def json_from_file(path):
    return json.loads(open(path).read())

df=pd.read_csv(os.path.join('data', 'all_data.csv'))
df = add_columns(clean_data(df))

columns=['CertName']+[c for c in df.columns if c!="CertName"]


counts = make_counts_df(df)

norm_df = counts.copy()
for column in norm_df.columns[:-1]:
    norm_df[column]=norm_df[column]/norm_df['total']
norm_df.drop(columns='total', inplace=True)


struct_dict = split_on_structure(counts)
plot_cert_structures(struct_dict)

#classifiers of certifications

classifier_dict = json_from_file(
    os.path.join('data', 'classifications.json'))


category_dict = group_certs(classifier_dict)
re_index_list=[x for y in category_dict.values() for x in y]


norm_df=re_indexer(norm_df, re_index_list)



commodities = json_from_file(os.path.join('data', 'commodities.json'))
ax=pie_from_dict(commodities)
plt.title('Single Commodity Certifications')
plt.savefig(os.path.join(charts_dir, 'cropspie.png'))
plt.show()


# In[47]:
category_colors=['salmon', 
                 #'goldenrod',  
                 'skyblue', 
                 'palegreen', 
                 'plum']

colors=[]
for cat, color in zip(category_dict.values(), category_colors):
    colors+=[color]*len(cat)

labels=['US, Multi-Comm',
#'US, Single-Comm',
'Regional, 1-Comm',
 'Low-Inc, 1-Comm',
'Low-Inc, Multi-Comm']


abbrvs, abbrv_dict = set_abbrvs(re_index_list, 
                    corrections = {'Global GAPS': 'GGAPS',
                                   'LODI Rules' : 'LODI',
                                   'Red Tomato Eco Apple' :'EA',
                                   'Fair Trade Intl' : 'FTMH'})



norm_df['scorecard/improvement'] = norm_df[['scorecard', 'improvement']].sum(axis=1)



summary=pd.DataFrame(np.array(
    [norm_df[norm_df.index.isin(cat)].astype(bool).mean() 
                               for cat in category_dict.values()]).T, 
                     index=norm_df.columns, columns=labels)


summary['Category']=summary.index


crop_1=norm_df[norm_df['CertName'].isin(classifier_dict['Single Commodity'])]
any_crop=norm_df[norm_df['CertName'].isin(classifier_dict['Multi-Commodity'])]


by_com=pd.DataFrame(np.array([crop_1.mean(), any_crop.mean()]).T, index=norm_df.columns, 
             columns=['Single Commodity', 'Multi-Commodity'])
   

counts=pd.concat([df[df.columns[:9]],df[df.columns[9:]].applymap(is_valid)], axis=1)
counts ['CertName'] = df['CertName'] 
gb=counts.groupby(['CertName', 'requirement', 'improvement'])
by_req_type=gb.sum()

by_req_type.reset_index()['requirement']

req_indexer=by_req_type.reset_index()['requirement']==1
data_req= transform_for_plotting(by_req_type[req_indexer.tolist()])
improve_indexer=(by_req_type.reset_index()['improvement']==1)
data_impr=transform_for_plotting(by_req_type[improve_indexer.tolist()])

non_reqindexer=(req_indexer+improve_indexer)==0
data_nonreq = transform_for_plotting(by_req_type[non_reqindexer.tolist()])

data_sets=[data_req, data_impr, data_nonreq]



#Make patches for plotting
base_patches=[]
for color, label in zip(uniq(colors), labels):
    print(color)
    base_patches.append(mpatches.Patch(color=color, label=label))

patches=base_patches+[mpatches.Patch(label='Score-Card', hatch=r'\\\\', edgecolor='black', facecolor='white'),
                  mpatches.Patch(label='Improvement to Mandate', edgecolor='black', hatch='O', facecolor='white'),
                  mpatches.Patch(label='Mandatory', facecolor='white', edgecolor='black'), 
                 
                 ]

patches_p=base_patches+[ mpatches.Patch(label='Not Registered by EPA', hatch=r'\\\\', edgecolor='black', facecolor='white'),
                  mpatches.Patch(label='Registered by EPA', facecolor='white', edgecolor='black'), 
                 ]

pest_dfs=make_pesticide_dfs(counts['CertName'].unique(), re_index_list)
pest_list_plotter(pest_dfs, handles = patches_p)

#dict for plotting:
cols_dict={
    'admin': ['Record-Keeping',
 'Planning',
 'Education/training',
 'Monitoring'],
  
    'worker' : [
 'Worker Safety',
 'Materials/Waste Mgmt'],

'farm_pratices' :
      ['Pesticide Practices', 
      'Prohibited Practice',  
      'Agronomic Practices',
      'Biosecurity/Sanitation',]
     }


for name, cols in cols_dict.items():
    std_stack_plot(cols, data_req, data_impr, data_nonreq,  
               f"{name}.png", handles=patches)

plt.legend(handles=patches)
ax=plt.gca()
ax.axis('off')