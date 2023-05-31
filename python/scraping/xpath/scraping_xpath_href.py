from bs4 import BeautifulSoup
from lxml import etree
import requests



#------------------------------------------------------------------------------
#  Copyright(c):
#   Ubaidul Khan / ubaidul(dot)khan(at)gmail(dot)com
#   
#------------------------------------------------------------------------------
#
#
#------------------------------------------------------------------------------
#  References
#   https://www.scrapingbee.com/blog/practical-xpath-for-web-scraping/
#
#------------------------------------------------------------------------------

  
  
URL = "https://releases.ubuntu.com/23.04/"
xpath_href_anchor_tag = '//*[@id="main"]/div/div//div[2]/a/@href'

  
HEADERS = ({'User-Agent':
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 \
            (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',\
            'Accept-Language': 'en-US, en;q=0.5'})
  
webpage = requests.get(URL, headers=HEADERS)
soup = BeautifulSoup(webpage.content, "html.parser")
dom = etree.HTML(str(soup))

contents = dom.xpath(xpath_href_anchor_tag)


for content in contents:
  print(f'  href link: {content}')

