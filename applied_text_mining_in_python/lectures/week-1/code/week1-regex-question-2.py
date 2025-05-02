tweet = "@nltk Text analysis is awesome! #regex #pandas #python"

print([word for word in tweet.split() if word.startswith('#')])
