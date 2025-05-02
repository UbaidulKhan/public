import re

dateStr = '23-10-2020\n23/10/2002\n23/20/02\n10/23/2002\n23 Oct 2002\n23 October 2002\nOct 23, 2002\nOctber 23, 2002\n'

dates1 = re.findall(r'\d{2}[/-]\d{2}[/-]\d{4}', dateStr)
print(dates1)

dates2 = re.findall(r'\d{2}[/-]\d{2}[/-]\d{2,4}', dateStr)
print(dates2)

dates3 = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', dateStr)
print(dates3)

# Find all dates with month spelled out in 3 letters - Jan, Feb, Mar             
dates4 = re.findall(r'\d{2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}', dateStr)
print(dates4) 


# Find all dates with month spelled out, ending with 4 digit year
dates4 = re.findall(r'(?:\d{2} )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* (?:\d{2}, )?\d{4}', dateStr)
print(dates4) 

