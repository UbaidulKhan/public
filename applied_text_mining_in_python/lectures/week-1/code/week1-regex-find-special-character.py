'''
Objective: find special character
'''

import re

text = 'ouagadougou'
vowels = re.findall(r'[aeiou]', text)
print(f"vowels are: {vowels}")

non_vowels = re.findall(r'[^aeiou]', text)
print(f"non-vowels are: {non_vowels}")
