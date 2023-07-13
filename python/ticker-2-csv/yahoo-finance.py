import yfinance as yf

def get_sector(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    sector = info.get('sector', 'N/A')
    return sector

ticker = "AAPL"  # Replace with the desired stock ticker
sector = get_sector(ticker)

print("Ticker:", ticker)
print("Sector:", sector)
