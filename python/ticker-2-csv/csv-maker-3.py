import requests
from lxml import html
from bs4 import BeautifulSoup


url = "https://www.nasdaq.com/market-activity/stocks/dkng"

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Referer": "https://www.nasdaq.com/market-activity/stocks/dkng"
}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")
print(soup)

'''
tree = html.fromstring(response.content)
print(tree)

# sector_xpath = '//span[@class="label__container__3LIA4"]//a'
sector_xpath = '//tr/td[normalize-space(text())="Sector"]/../.'
sector_element = tree.xpath(sector_xpath)
# sector = sector_element.text.strip()
sector = sector_element

print("Sector:", sector)
'''