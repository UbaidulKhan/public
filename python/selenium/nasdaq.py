from selenium import webdriver
import time
import os

# Add the directory containing the Firefox binary to the PATH environment variable
os.environ['PATH'] += ':/Applications/Firefox-91.app/Contents/MacOS'

# url = f"https://api.nasdaq.com/api/quote/DKNG/info?assetclass=stocks"
url = 'https://www.nasdaq.com/market-activity/stocks/dkng'

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Referer": "https://www.nasdaq.com/market-activity/stocks/dkng"
}

# Create a new instance of the Firefox driver
driver = webdriver.Firefox()

# Set the maximum amount of time to wait for a page to load
driver.set_page_load_timeout(10)


# Specify the path to the Firefox binary
driver_path = './geckodriver'
firefox_binary_path = '/Applications/Firefox-91.app/Contents/MacOS/firefox'

# Create the WebDriver instance with the specified binary location
# driver = webdriver.Firefox(executable_path='./geckodriver', firefox_binary=firefox_binary)
# driver = webdriver.Firefox(executable_path=driver_path, firefox_binary=firefox_binary_path)
driver = webdriver.Firefox()


try:
    # Open the web page
    # driver.get("https://www.example.com")
    driver.get(url)
    time.sleep(5)
    

    # Find the table element using XPath
    table = driver.find_element("xpath", "//table[@class='summary-data__table']")

    # Find all the table data (td) elements within the table using XPath
    table_data = table.find_element("xpath", "//tr/td[normalize-space(text())='Sector']/../.")

    # Loop through the table data elements and print their text content
    for data in table_data:
        print(data.text)

except Exception as e:
    print("Error:", str(e))

finally:
    # Close the browser window
    driver.quit()
