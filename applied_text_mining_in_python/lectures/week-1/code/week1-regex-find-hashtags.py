tweet = "@nltk Text analysis is awesome! #regex #pandas #python"
text = tweet.split()

for i in text:
  if(i.find('#') != -1):
    print(i)
