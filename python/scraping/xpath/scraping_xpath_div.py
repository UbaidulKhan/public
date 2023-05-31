from bs4 import BeautifulSoup
from lxml import etree
import requests
import sys


url = "https://releases.ubuntu.com/"

headers = ({'User-Agent':
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 \
            (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',\
            'Accept-Language': 'en-US, en;q=0.5'})

#
## Defunct            
xpath_div_tag_a = "//div[contains(@class, 'p-table-wrapper')]"
xpath_div_tag_b = "//*[@id=\'pageWrapper\']/div[2]/div"
xpath_div_tag_c = "/html/body/div/div[2]/"
xpath_div_tag_d = "p-table-wrapper"

xpath_div_tag = [xpath_div_tag_a, xpath_div_tag_b, xpath_div_tag_c, xpath_div_tag_d]


tagged_content_one = None
tagged_content_two = None
  
try:

  response = requests.get(url, headers=headers)
  
except Exception as e:
  print(f' There was an error - failed to reach: {url}')
  print(f' Please check your network connection and try again, error encountered:')
  print(f'  {e}')
  sys.exit(1)

soup = BeautifulSoup(response.content, "html.parser")
content = etree.HTML(str(soup))

for xpath_tag in xpath_div_tag:

  print(f'  \_ Testing with tag: {xpath_tag}')
  print(f'-------------------------------------------------------------------------\n')


  # tagged_content_two = soup.find("div", {"class":xpath_div_tag}).find_all('a')
  try:
    tagged_content_two = soup.find("div", {"class":xpath_div_tag})
  except Exception as e:
    print(f'  \_ tagged_content_one was not found')

  if(tagged_content_two):  
    print(f'  \_ Content: {tagged_content_two}')
  

  print("\n-----------------------------------------------------------------------------")
  try:
    tagged_content_one = content.xpath(xpath_div_tag)
  except Exception as e:
    print(f'  \_ tagged_content_one was not found')
  
  if(tagged_content_one is not None):  
    print(f'  \_ Content: {tagged_content_one}')

  print("\n-----------------------------------------------------------------------------")
  
  
  '''
  for entry in tagged_content:
    print(f'  \_ Release num is: {entry.text}')
  '''