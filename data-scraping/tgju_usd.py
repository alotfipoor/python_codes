import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlopen
from bs4 import BeautifulSoup

url = "http://www.tgju.org/chart/price_dollar_rl/2"
html = urlopen(url)

soup = BeautifulSoup(html, 'lxml')


rows = soup.find_all('tr')
list_rows = []
for row in rows:
    row_td = row.find_all('td')
    str_cells = str(row_td)
    cleantext = BeautifulSoup(str_cells, "lxml").get_text()
    list_rows.append(cleantext)
df = pd.DataFrame(list_rows)
df[0] = df[0].str.strip('[')
df[0] = df[0].str.strip(']')


df[0] = df[0].str.replace(',','')
df1 = df[0].str.split(' ', expand=True)
df2 = df1.drop(df1.index[0])

df2.columns = ['Return', 'Min', 'Max', 'Finish', 'DateM', 'DateP', 'Last']
df3 = df2.drop(['Last'], axis=1)


df3.to_csv("C:/Users/alotf/Documents/GitHub/PythonProjects/Data-Scraping/data.csv", index=False)
