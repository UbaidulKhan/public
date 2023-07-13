import requests
from lxml import html

url = "https://www.nasdaq.com/market-activity/stocks/dkng"
headers = {'User-Agent': 'Mozilla/5.0 (Linux; Android 5.1.1; SM-G928X Build/LMY47X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.83 Mobile Safari/537.36'}
response = requests.get(url, headers=headers)

tree = html.fromstring(response.content)

sector_xpath = '//span[@class="label__container__3LIA4"]//a'
sector_element = tree.xpath(sector_xpath)[0]
sector = sector_element.text.strip()

print("Sector:", sector)
