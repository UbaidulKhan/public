import requests
from lxml import html

from bs4 import BeautifulSoup


# import nasdaqdatalink
# mydata = nasdaqdatalink.get("FRED/GDP")

api_key = f'kQMadDsvVdEbs6waz1kN'
ticker = 'DKNG'
# url = f"https://data.nasdaq.com/api/v3/datasets/WIKI/{ticker}/data.json?api_key={api_key}"
url = f"https://data.nasdaq.com/api/v3/datasets/WIKI/DKNG/data.json?api_key=kQMadDsvVdEbs6waz1kN"

headers = {'User-Agent': 'Mozilla/5.0 (Linux; Android 5.1.1; SM-G928X Build/LMY47X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.83 Mobile Safari/537.36'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")
print(soup)

