import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlopen
from bs4 import BeautifulSoup

url = "http://www.hubertiming.com/results/2017GPTR10K"
html = urlopen(url)

soup = BeautifulSoup(html, 'lxml')

# create the rows
rows = soup.find_all('tr')
list_rows = []
for row in rows:
    row_td = row.find_all('td')
    str_cells = str(row_td)
    cleantext = BeautifulSoup(str_cells, "lxml").get_text()
    list_rows.append(cleantext)
df = pd.DataFrame(list_rows)
df1 = df[0].str.split(',', expand=True)
df1[0] = df1[0].str.strip('[')

# returns header
col_labels = soup.find_all('th')
all_header = []
col_str = str(col_labels)
cleantext2 = BeautifulSoup(col_str, "lxml").get_text()
all_header.append(cleantext2)
df2 = pd.DataFrame(all_header)
df3 = df2[0].str.split(',', expand=True)

# Concat two df
frames = [df3, df1]
df4 = pd.concat(frames)

# assign headers
df5 = df4.rename(columns=df4.iloc[0])

# drop NA
df6 = df5.dropna(axis=0, how='any')

# drop first rows and more cleaning
df7 = df6.drop(df6.index[0])
df7.rename(columns={'[Place': 'Place'},inplace=True)
df7.rename(columns={' Team]': 'Team'},inplace=True)
df7['Team'] = df7['Team'].str.strip(']')

#df7.head(10)

df7.to_csv("C:/Users/alotf/Documents/GitHub/PythonProjects/Data-Scraping/data.csv", index=False)
