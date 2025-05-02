''' 
import pprint
stuff = ['spam', 'eggs', 'lumberjack', 'knights', 'ni']
stuff.insert(0, stuff[:])
pp = pprint.PrettyPrinter(indent=10)
pp.pprint(stuff)

'''

from pprint import pformat

nested_dict = [{"language": "Python", "application": ["Data Science", "Automation", "Scraping", "API"]}, {"language": "Javascript", "application": ["Web Development", "API", "Web Apps", "Games"]}]

string_representation = pformat(nested_dict)
print(string_representation)