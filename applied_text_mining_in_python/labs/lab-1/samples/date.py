import pandas as pd
import re


string = "s 20 yo m carries dx of bpad, presents for psychopharm consult.  moved to independence area for school as of september 1985."

regex = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* [\d]{2,4}"


regex_found = re.findall(regex, string)

if(regex_found):
  print(f"  Regex matched: {regex_found}")
    
else:
  print(f"\nLine: {line_counter} {regex} did not match: {row}")
    

