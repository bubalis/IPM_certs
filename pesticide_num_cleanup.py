#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:27:03 2020

@author: bdube
"""
from pest_lookup import initial_loader, data_saver
import os
import json
from main_list_fixer import load_ref_num_dict

#%%

results=load_ref_num_dict('chemical_ref_nums.txt')
os.chdir('pesticide_lists')
assert ('ddt' in results)

new_man_fixes={'carbendizum': ['10605-21-7'],
               'beta hexachlorocyclohexane': ['319-85-7'], 
               'carbaryl x': ['63-25-2'], 
                'dimethoate x': ['60-51-5'],
                'naled x': ['300-76-5'],
                'e-phosphamidon': ['13171-21-6', '297-99-4', '23783-98-4'],
                'hexachlorocyclohexane': ['319-84-6', 
                '319-85-7', '58-89-9', '319-86-8', '608-73-1'],
                'boscalid,': ['188425-85-6'],
                'disulfoton': ['298-04-4'],
                'alpha-bhc':  ['319-84-6'],
                'azinphos-methyl':['86-50-0'],
                'dinitro-ortho-cresol': ['497-56-3'],
                'methyl-parathion': ['56-38-2'],
                 'perfluorooctane sulfonyl fluoride': ['307-35-7'],
                 '1,2-dibromo-3-chloropropane (dbcp)':['96-12-8'],
                 'arsenic oxide (3)': ['1327-53-3'],
                 'pyrazoxon':['108-34-9'],
                   '2,4-d, dimethylamine salt': ['2008-39-1'],
  'perfluorooctane sulfonic acidâ\xa0(pfos), its salts andâ\xa0perfluorooctane sulfonyl fluoride':
      ['335-67-1', '307-35-7'],
 'trichlorophenoxyacetic acid, alkyl (c3-c7) ester': ['93-76-5'],
   'tributyltin chloride, myristylamine salt': ['1461-22-9', '2016-42-4']  ,
   'tributyltin iso-thiocyanate-triphenylarsine oxide':['56035-9'],
   'tributyltin chloride complex of ethylene oxide condensate of abietylamine': ['56573-85-4'],
   'tributyltin 3-pyridine carboxylate': ['27189-59-9'],
  'Tributyltin 2-pyridinecarboxylate' : ['73149-61-8'], 
   'Tributyltin naphthenate': ['85409-17-2'],
      'pcp, sodium salt, other related':['87-86-5','87-85-4', '131-52-2'],
      'propylene glycol isobutyl ether 2,4,5-trichlorophenoxyacetate': ['53466-86-7'],
      'chloromethoxy propyl mercuric acetamide': ['1319-86-4'],
      'hydroxymercury cresol': ['12379-66-7'], 
       'mercury naphthenate'  : ['1336-96-5'],
   'trichlorophenoxyacetic acid, triethanolamine salt': [],
   '3-chloro-1, 2-propanediol': ['96-24-2'],
   'zinc phosphide (zp)': ['1314-84-7'],
   'î²-hexachlorocyclohexane': ['319-85-7'],
   	'dichlorvos ddvp':['62-73-7'],
       'dinoseb and dinoseb salts': ['88-85-7'],
       'pentachlorophenol, zinc salt of alkyl*-N-propanediamine' : ['87-86-5', ],
   '2,4,5-t, 2-ethylhexyl ester': ['1928-47-8'],
   '(3-hydroxy-2-methoxypropyl)mercuric acetate': ['69653-69-6'],
       '2,4,5-T, Triethylamine Salt': ['2008-46-0'],
   '2,4,5-t, butoxyethanol ester': ['2545-59-7'],
     '2,4,5-t, butyl ester':     ['93-79-8'],
     '2,4,5-t, butyric acid': ['93-80-1'],
 'p-tert-octylphenoxyethoxyethyl dimethyl benzyl ammonium mercuric chloride':['53433-01-5'],
'methylmercury dicyano diamide': ['502-39-6'],
'2,4,5-trichlorophenoxyacetic acid, tripropylene glycol isobutyl ether ester': ['53535-32-3'],
'2,4,5-trichlorophenoxyacetic acid, n,n-dimethyloleylamine salt': ['53404-89'],
'2,4,5-trichlorophenoxyacetic acid, n,n-dimethyl oleyl-linoleyl amine salt': ['55256-33-2'],
'2,4,5-t, dodecylamine salt': ['53404-84-5'],
'2,4,5-t, isooctyl ester': ['25168-15-4'],

'2,4,5-t, n-oleyl-1,3-propylenediamine salt':['53404-87-8'],
'2,4,5-t, propylene glycol butyl ether ester': ['1928-48-9', '7173-98-0'],
'2,4,5-trichlorophenoxyacetic acid, 2-ethyl-4-methylpentyl ester':['69462-12-0'],
'alkyl mercury compounds':[],
'2,4,5-trichlorophenoxyacetic acid, butoxyethoxypropanol ester': ['1928-58-1'],
'2,4,5-trichlorophenoxyacetic acid, diethylethanolamine salt': ['53404-86-7'],
'alkyloxyl and aryl mercury compounds':[],

 'bis (tributyltin) adipate': ['7437-35-6'], 
"ethyl hexylene glycol": ["107-41-5"],

'di(phenylmercuri) ammonium propionate': ['18467-88-4'],
  

 'phenylmercuriammonium propionate':['53404-68-5'],

'diethanolamine dinoseb ( 2-sec-butyl-4,6-dinitrophenol )': ['53404-43-6'],

'dipropylene glycol isobutyl ether 2,4,5-trichlorophenoxyacetate': ['53535-31-2'],

'chloromethoxypropylmercuric acetate': ['1319-86-4'],

'ethylmercury pentachlorophenate': ['22232-28-6'],

'mercury and inorganic mercury compounds':['7487-94-7', '21908-53-2', '1319-86-4',
                                           '27236-65-3', '104-68-9', '62-38-4'],
'mercury and mercury compounds':['7487-94-7', '21908-53-2', '1319-86-4',
                                           '27236-65-3', '104-68-9', '62-38-4'],
 'mercury pentanedione': ['14024-55-6'],
 'methoxymethyl mercury compounds': ['123-88-6'],
'fatty acid (c6-20) esters of pentachlorophenol':[],
'pma': ['62-38-4'],
'pcp':['87-86-5,87-85-4'],
'chloromethoxypropylmercuric acetate ':['1319-86-4'],
'Ethyl hexyleneglycol (6-12) ': ['94-96-2'],
'(E)-Mevinphos': ['298-01-1'],
 '(Z)-Mevinphos': ['338-45-4'],
 'Dinitro cresol': ['53240-95-2 534-52-1'],
 'Thallium(I) sulfate': ['7446-18-6'],
 '(3-Ethoxypropyl)mercury bromide': ['6012-84-6'],
 '2,2,2-Trichloro-N-(pentachlorophenyl)acetimidoyl chloride': ['61881-19-4'],
 '2-(acetoxymercuri)ethanol': ['4665-55-8'],
 '3-(Hydroxymercuri)-4-nitro-o-phenol, sodium salt': ['1300-34-1'],
 'Arochlor (composition unspecified, PCBs and/or PCTs)': ['12767-79-2'],
 'Bis (tributyltin) sulfone': ['4808-30-4'],
 'Bis(tributyltin) dodecenylsuccinate': ['12379-54-3'],
 'Bis(tributyltin) salicylate': ['22330-14-9'],
 'Bis(tributyltin) succinate': ['4644-96-6'],
 'Bis(tributyltin) sulfosalicylate': ['4419-22-1'],
 'Carbonic acid, Bis(pentachlorophenyl) ester': ['7497-08-7'],
 'Di(phenylmercuric) dodecenyl succinate': ['2195843'],
 'Hexachlorocyclohexane': ['58-89-9'],
 'N-(Ethylmercury)-p-toluenesulfonanilide': ['517-16-8'],
 'N-(Phenylmercuri) urea': ['2279-64-3'],
 'o-(Chloromercuri)phenol': ['90-03-9'],
 'o-(Hydroxymercuri)benzoic acid, cyclic anhydride': ['5722-59-8'],
 'PMAA': ['61840-45-7 62-38-4'],
 'Poly (methylmethacrylate-co-tributyltin methacrylate)': ['26354-18-7'],
 'Polychlorinated terphenyls': ['26140-60-3 61788-33-8 84-15-1 98849-70-8'],
 '1,2-Dibromo-3-Chloropropane (DBCP)': ['145667-72-7 67708-83-2 96-12-8'],
 '2,3,4,5-Bis (2-Butylene) Tetrahydro-2-Furaldehyde': ['126-15-8'],
 'Arsenic Oxide (3)': ['10102-53-1 10124-50-2 13464-58-9 13592-22-8'],
 'Di(Phenylmercury)Dodecenylsuccinate ': ['2195843'],
 'Potassium 2,4,5-Trichlorophenate ': ['35471-43-3'],
  'alpha-BHC alpha-HCH': ['319-84-6'],
'paraffin oils': ['64741-88-4',
 '64741-89-5',
 '64741-97-5',
 '64742-46-7',
 '64742-54-7',
 '64742-55-8',
 '64742-65-0',
 '72623-86-0',
 '97862-82-3'],

'Glyphosate and its salts': [	
'1071-83-6', 
'38641-94-0', 
'70393-85-0', '81591-81-3'], 
  
'bhc': ['319-84-6', '319-85-7'], 

   'Tributyltin compounds': '''266-560-5
266-955-2
277-638-3
279-149-0
284-083-0
284-301-4
287-083-9
288-886-7
288-887-2
291-997-3
292-191-4
292-193-5
297-707-1
298-515-0
299-691-1
300-673-3
300-971-3
300-977-6
301-734-7
302-265-0
215-958-7
215-959-2
216-202-9
217-847-9
218-452-4
221-433-3
221-434-9
222-721-1
222-873-9
223-469-5
223-699-6
224-398-2
224-399-8
225-291-3
225-327-8
225-726-7
225-727-2
225-831-8
227-437-1
231-291-4
231-618-0
234-226-8
236-381-7
236-563-6
211-704-4
212-383-3
213-933-5
213-939-8
200-268-0
200-269-6
'''.split('\n'), 
'Captafol': ['2425-06-1'],
'CPMA' : ['1319-86-4'],
'Hexabromdiphenylether '       :  ['446255-03-4', '36483-60-0'],
'Heptabromdiphenylether' : ['446255-20-5','68928-80-3'],
'lambda-cyhalothrin' : ['91465-08-6'],
'Etofenprox' : ['80844-07-1'],
'1,2,3,4,5,6-hch': ['319-84-6', '319-85-7', '58-89-9', '319-86-8', '608-73-1'],
'heptaclor': ['76-44-8'],
'2,4,5-trichlorophenate':['95-95-4'],
'Pentachlorphenol': ['87-86-5'],
'DNOC and its salts': ['534-52-1'],
'Tecnazene': ['117-18-0'],
'Triorganostannic compounds': [ i.strip() for  i in '''
14254-22-9
639-58-7
661-69-8
892-20-6
994-31-0
994-32-1
1066-44-0
1066-45-1
2587-76-0
2767-54-6
13121-70-5
13356-08-6
90552-69-5
56-24-6
3151-41-5
76-63-1
                     '''.split('\n')
                               if i],
'technazene': ['117-18-0'],

'demeton (demeton s-methyl)': ['126-75-0'],

'propylene oxide, oxirane': ['75-56-9'],
'dichlorodiphenyl  dichloroethan' : ['72-54-8'],
'metamidophos': ['10265-92-6'],

}



alias_man_fixes={'dichloro diphenyl trichloroethane': 'DDT',
    'metomil':'methomyl',   
    'copper acetoarsenite (paris green)':'cupric acetoarsenite',
    'copper hydroxide,': 'copper hydroxide',
                 'carbarylâ\x9c\x95': 'carbaryl',
                 'verdepryn': 'cyclaniliprole',
                  'dinoseb, its acetate and salts': 'dinoseb', 
                 'dichlorvos (ddvp)': 'dichlorvos',
                 'gamma-lindane': 'lindane',
                 'captafol (cis isomer)': 'captafol', 
                 'pma, other related': 'pma',
                 'ddt, p,p\\': 'ddt',
                 'oxydemeton-methyl x': 'oxydemeton-methyl',
                 '2,4,5-trichlorophenoxyacetic acid, n,n-dimethyl oleyl-linoleyl amine salt': '2,4,5-trichlorophenoxyacetic acid',
                 'endrin, other related': 'endrin',
                 'pcp, potassium salt': 'pcp',
                 '2,4,5 tcp': '2,4,5-trichlorophenate',
                 'borax disodium': 'borax',
                 'Z-Phosphamidon' : 'Phosphamidon',
                 'Polybrominated biphenyls mixture ': 'PBB',
                 'Tributyltin' : 'Tributyltin Compounds',
                 'DDD, other related' : 'DDD',
                 '(1alpha,2alpha,3beta,4alpha,5alpha,6beta)-1,2,3,4,5,6-hch': '1,2,3,4,5,6-hch',
                 'hch' : '1,2,3,4,5,6-hch',
                 'Hexchlorocyclohexane': '1,2,3,4,5,6-hch',
                 'PMDS': 'Di(phenylmercury)dodecenylsuccinate',
                 'Lambda-cyhalothin': 'lambda-cyhalothrin' ,
                 'copper (ii) hydroxide': 'copper hydroxide',
                 'BHC (other than gamma isomer': 'bhc',
                 'FENAMIFOS': 'FENAMIPHOS'.lower(),
                 'METAMIDOFOS' : "METAMIDOPHOS".lower(),
                 'ALDICARD': 'ALDICARB'.lower(),
                 
                 }

#%%

name_fixes=json.loads(open('manual_fixes.txt').read())
alias_man_fixes.update(name_fixes)
alias_man_fixes={v:k for k,v in alias_man_fixes.items()}


to_delete_from_alias=['acetate', 'Magnesium sulfate', 'ethyl', 'potassium',
                      'sodium', 'salt', 'Hexane', 
                      'sulfonic', 'sulfonyl', 'fluoride', 
                      'perfluorooctane', 
                      'perfluorooctanoic', 'Perfluorooctanoic acid', 
                      'alkyl', 'condensate', 'Chloride', 'myristylamine',
                      'oxide', 'ammonium', 'fatty', 'acid', 
                      'ether', 'glycol', 'isobutyl', 'aryl', 'mercury and mercury compounds', 
                      'mercury and inorganic mercury compounds']



#%%


aliases=json.loads(open('aliases2.txt').read())
results.update(new_man_fixes)
results ={k.strip().lower(): v for k, v in results.items()}
for key in to_delete_from_alias:
    if key in aliases:
        del aliases[key]


remove_from_ref_nums=[]


for key, values in aliases.items():
    for v in values:
        try:
            results[v.lower().strip()]=results[key.lower().strip()]
        except:
            #print(key)
            continue
    
for key, value in alias_man_fixes.items():
    try:
        results[value.lower().strip()]=results[key.lower().strip()]
    except:
        print(key)
        
    
keys_to_change=[]
for key in results:
    if key.strip() in results and key!=key.strip():
        keys_to_change.append(key)
       
print(keys_to_change)
for key in keys_to_change:
    results[key]=results[key.strip()]
#%%

compounds=['arsenic and its compounds','Arsenic and its compounds', 'arsen and its compounds', 
          'mercury and inorganic mercury compounds', 
          'mercury and mercury compounds',
          'inorganic mercury compounds', 'mercury and its compounds', 'mercury compounds']

for c, string in zip(compounds, ['arsen', 'arsen', 'arsen',  'merc', 'merc', 'merc', 'merc', 'merc', 'merc']):
    data=[]
    for r in [r for r in results if string in r and r not in compounds]:
        try:
            data+=results[r]
        except:
            print(r)
            print(results[r])
            
    results[c]=data        





#%%
#assert ('Tributyltin compounds'.lower() in results)

results={k.lower().strip(): v for k,v in results.items()}
more_results = {}
for k, v in results.items():
    if 'hexachlorocyclohexane' in k:
        more_results[k.replace('hexachlorocyclohexane', 'hch')] = v
        more_results[k.replace('hexachlorocyclohexane', 'bch')] = v

results = {**results, **more_results}

assert ('Mercury and its compounds'.lower() in results)
with open('chemical_ref_nums2.txt', 'w+') as f:
    print(json.dumps(results), file=f)
