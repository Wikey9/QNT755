# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 18:30:11 2022

@author: Alyssa Griswold, Alicia Maffiolini, Steve Marsella, Jerrod Wike
"""
#USER INPUTS - CHECK BEFORE YOU RUN!
Countries = ['CA','DE','FR','GB','IN','US']               #Countries to include
columns = (0,1,4,6,7,8,9,12,13,14)                                    #Columns for clustering
path = "C:/Users/Jerrod/Documents/UHart/QNT755/Project/youtube/"    #Location of files
#END USER INPUTS

import numpy as np
import os as os
from zipfile import ZipFile
from numpy import genfromtxt
import datetime
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling

#Set current working directory to file location
os.chdir(path)

"""
The loop below will extract the analysis data from the raw dataset and create
a new CSV called "compiled_data" which includes an additional row for the
country associated with each given row.
"""

#For each country:
for i in np.arange(0,len(Countries)):
    
    #Target ZIP for this country
    it_path = "%s%svideos.csv.zip" % (path,Countries[i])

    #Extract ZIP
    with ZipFile(it_path, 'r') as zip:
        zip.extractall()

    #Target unzipped CSV    
    it_path = "%s%svideos.csv" % (path,Countries[i])    

    #Generate NumPy array from CSV 
    data = genfromtxt(it_path,delimiter=',',names=True,
                  invalid_raise=False,usecols=columns,encoding='utf-8',
                  deletechars="~!@#$%^&*()-=+~\|]}[{';: /?.>,<. ",
                  dtype=('S100','S32','i4','S1000',
                         'int32','int32','int32',
                         '?','?','?'))

    #New dtype for Country Code
    dtype = data.dtype.descr + [('country_code','U10'),('number_tags','i4')]

    #Create new array which includes extra column for country code
    country_data = np.empty((len(data)),dtype=dtype)

    #Create a list of headers
    headers = ()
    for j in country_data.dtype.descr:
        headers = np.append(headers,j[0])

    #Map original data columns to new array
    for k in np.arange(0,len(headers)-2):
        country_data[headers[k]] = data[headers[k]]
    
    #Add new column for country code 
    country_data['country_code'] = Countries[i]
    
    #Add new column for number of tags
    for j in np.arange(0,len(country_data)):
        if country_data['tags'][j] == b'[none]':
            country_data['number_tags'][j] = 0
        else:
            country_data['number_tags'][j] = country_data['tags'][j].count(b'|') + 1
      
    #Append data to new compiled CSV
    with open('compiled_data.csv','a') as csvfile:
        np.savetxt(csvfile, country_data, delimiter=",",
                   fmt=('%s','%s','%i','%s','%i', '%i',
                        '%i','%s','%s','%s','%s','%i'))
    
    #Delete original CSV
    os.remove("%svideos.csv" % (Countries[i]))

"""
K-Means Clustering Analysis
"""

cluster_columns = (4,5,6,11)
cluster_path = "%scompiled_data.csv" % (path)

cluster_data = genfromtxt(cluster_path,delimiter=',',
                          usecols=cluster_columns,encoding='utf-8',
                          dtype = ('int32'))

kmeans = KMeans(5)
kmeans.fit(cluster_data)

clusters = kmeans.fit_predict(cluster_data)

#Plot Views vs Likes
plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=clusters, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()

#Plot Views vs Dislikes
plt.scatter(cluster_data[:, 0], cluster_data[:, 2], c=clusters, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()

#Plot Views vs Number of Tags
plt.scatter(cluster_data[:, 0], cluster_data[:, 3], c=clusters, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()

