import csv
import requests
from bs4 import BeautifulSoup

def get_stock_data(ticker):
  url = f"https://finance.yahoo.com/quote/{ticker}"
  response = requests.get(url)
  soup = BeautifulSoup(response.content, "html.parser")

  # Retrieve required data from Yahoo Finance
  sector = soup.find("span", attrs={"data-reactid": "47"}).text
  source = soup.find("span", attrs={"data-reactid": "37"}).text
  current_price = soup.find("span", attrs={"data-reactid": "50"}).text

  # Perform any calculations for additional columns
  # For example:
  min_next_price = 1.1 * float(current_price)
  current_price_offset = float(current_price) - 100
  daily_change = 0.05 * float(current_price)
  profit_target = 1.2 * float(current_price)
  loss_tolerance = 0.9 * float(current_price)
  comment = "Example comment"

  return [url, sector, source, min_next_price, current_price_offset, daily_change, profit_target, loss_tolerance, comment]

# Read tickers from a file
file_path = "tickers.txt"  # Replace with your file path
output_file = "stock_data.csv"  # Replace with your desired output file path

stock_data = []

with open(file_path, "r") as file:
  for line in file:
    ticker = line.strip()
    stock_info = get_stock_data(ticker)
    stock_data.append(stock_info)

# Write the data to a CSV file
with open(output_file, "w", newline="") as file:
  writer = csv.writer(file)
  writer.writerow(["Yahoo Finance URL", "Sector", "Source", "Min Next Price", "Current Price Offset", "Daily Change",
                   "Profit Target", "Loss Tolerance", "Comment"])
  writer.writerows(stock_data)

print("CSV file generated successfully!")
