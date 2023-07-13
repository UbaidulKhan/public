import requests

def get_sector(ticker, api_key):
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()

    if 'Sector' in data:
        sector = data['Sector']
    else:
        sector = "N/A"

    return sector

ticker = "DKNG"  # Replace with the desired stock ticker
api_key = "7LGEVITZO0D15U4C"  # Replace with your Alpha Vantage API key
sector = get_sector(ticker, api_key)

print("Ticker:", ticker)
print("Sector:", sector)
