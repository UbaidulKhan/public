import requests
import json



def get_stock_quote(ticker):
    url = f"https://api.nasdaq.com/api/quote/{ticker}/info?assetclass=stocks"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Referer": "https://www.nasdaq.com/market-activity/stocks/{ticker}"
    }

    response = requests.get(url, headers=headers)
    
    data = response.json()
    
    json_data = json.dumps(data, indent=2)
    
    print(json_data)
    
    
    # Extract relevant data from the response
    quote = data["data"]["primaryData"]["lastSalePrice"]
    # sector = data["data"]["primaryData"]["sector"]
    # source = data["data"]["source"]
    
    # return quote, sector, source
    return (quote)


ticker = "AAPL"  # Replace with the desired stock ticker
# quote, sector, source = get_stock_quote(ticker)
quote = get_stock_quote(ticker)


# print("Ticker:", ticker)
print("Quote:", quote)
# print("Sector:", sector)
# print("Source:", source)