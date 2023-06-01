from bs4 import BeautifulSoup
from lxml import etree
import requests
import sys


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
#
# This script demonstrates web scraping using xpath
#------------------------------------------------------------------------------



url = "https://releases.ubuntu.com/"

headers = ({'User-Agent':
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 \
            (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',\
            'Accept-Language': 'en-US, en;q=0.5'})

#
## Defunct            
xpath_div_tag_a = "//div[contains(@class, 'p-table-wrapper')]" # This works with content.xpath
xpath_div_tag_b = "//*[@id='pageWrapper']/div[2]/div"          # This works with content.xpath
xpath_div_tag_c = "/html/body/div/div[2]/"                     # This does not work 
xpath_div_tag_d = "p-table-wrapper"                            # This works with soup.find_all
xpath_div_tag_e = "//div[@class='p-table-wrapper']"            # This works with content.xpath 


xpath_div_tag = xpath_div_tag_a


xpath_searched_content = None
souped_searched_content = None
  
try:

  response = requests.get(url, headers=headers)
  
except Exception as e:
  print(f' There was an error - failed to reach: {url}')
  print(f' Please check your network connection and try again, error encountered:')
  print(f'  {e}')
  sys.exit(1)

soup = BeautifulSoup(response.content, "html.parser")
# print(f'  \_ Raw content: {content}')

##################### Attempting to search by xpath #####################
print(" +-----------------------------------------------------")
print(" | Attempting to search by xapth")
print(" +-----------------------------------------------------")

try:
  content = etree.HTML(str(soup))
  # xpath_searched_content = content.xpath(xpath_div_tag).text
  xpath_searched_content = content.xpath(xpath_div_tag)
  
except Exception as e:
  print("  \_ content.xpath(xpath_div_tag) was unsuccessful")
  print(e)

if(xpath_searched_content):  
  print("   \_ content.xpath(xpath_div_tag) was successfully found")
  print(f'   \_ Type of xpath_searched_content: {type(xpath_searched_content)}')
  print("\n  -----------------------------------------------------------------------------")
  print("   \_ Content of xpath_searched_content:")
  print(f'   -- {xpath_searched_content} --')
  print("   \_ Type of xpath_searched_content:")
  print(f'   \_   {type(xpath_searched_content)}')
  print(f'   \_   Length of content: {len(xpath_searched_content)}')


  print(f'   \_ DIV content is: {xpath_searched_content}')

  for div_element in xpath_searched_content:
    print(f'   \_ Release num is: {div_element.text}')
    
    ''' 
    elements = div_element.xpath(".//*")
    
    # Iterate over the elements and print their contents
    for element in elements:
      print(etree.tostring(element, encoding='unicode'))
    '''
    
    a_tags = div_element.xpath(".//a")
    
    # Iterate over the <a> tags and print their text content
    for a_tag in a_tags:
        text_content = a_tag.text
        print(f'   \_ Release num is: {text_content}')

############################## TYPE 2: soup.find ##############################
print("\n\n +-----------------------------------------------------")
print(" | Attempting to search soup find_all")
print(" +-----------------------------------------------------")

try:
  # souped_searched_content = soup.find_all("div", {"class":xpath_div_tag})
  souped_searched_content = soup.find_all("div", {"class":xpath_div_tag})


except Exception as e:
  print("  \_ soup.find_all(div, {class:xpath_div_tag}) was unsuccessful")
  print(e)
  print("\n-----------------------------------------------------------------------------")

if(souped_searched_content):  
  print("  \_ soup.find(div, {class:xpath_div_tag}) was successfully found")
  print(f'  \_ Type of souped_searched_content: {type(souped_searched_content)}')
  print("\n  -----------------------------------------------------------------------------")
  print("  \_ Content of souped_searched_content:")
  # print(f'   -- {souped_searched_content} --')
  print(f'   \_   Length of content: {len(xpath_searched_content)}')

  print("\n-----------------------------------------------------------------------------")
  # print(f'{souped_searched_content}')


  for entry in souped_searched_content:
    # print(f'   \_ Release num is: {entry.text}')
    
    if ('<' in entry):
      print(f'   \_ skipping, contains <')
      continue
    else:
      # print(f'   \_ Release num is: {entry}')
      print(f'   \_ Release num is: {entry.text}')


  
