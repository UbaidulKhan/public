import pandas as pd
import re



'''
The goal of this assignment is to correctly identify all of the different date variants encoded 
in this dataset and to properly normalize and sort the dates. 

Here is a list of some of the variants you might encounter in this dataset:

Date in the data file can be one of the following
format:

  04/20/2009; 04/20/09; 4/20/09; 4/3/09
  Mar-20-2009; Mar 20, 2009; March 20, 2009; Mar. 20, 2009; Mar 20 2009;
  20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
  Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
  Feb 2009; Sep 2009; Oct 2010
  6/2008; 12/2009
  2009; 2010

Objective:

Once you have extracted these date patterns from the text, the next step is to sort them in 
ascending chronological order accoring to the following rules:

* Assume all dates in xx/xx/xx format are mm/dd/yy
* Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
* If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
* If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).
* Watch out for potential typos as this is a raw, real-life derived dataset.

With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices.

For example if the original series was this:

    0    1999
    1    2010
    2    1978
    3    2015
    4    1985

Your function should return this:

    0    2
    1    4
    2    0
    3    1
    4    3
'''

doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

#
## Create a data-frame and store the list into the data-frame
lower_case_df = pd.DataFrame()
df = pd.Series(doc)

#
## Convert the data frame to lower case
lower_case_df = df.str.lower()
line_counter = 1
for row in lower_case_df:
  #
  # Remove:
  #
  #  1) All strings containing a to z [a-z]+
  #  2) Remove open ( and close ) paranthesis
  #  3) Remove numbers with more than 4 consecutive digits
  #

  #
  ## Pattern for finding:
  #    6 Oct 85
  #    6 October 1985
  #
  regex_1 = r"[\d]{1,2} [adfjmnso][a-z]* [\d]{2,4}"
  
  #
  ## Pattern for finding:
  #    Oct 6, 85
  #    October 6, 1985
  # 
  regex_2 = r"[adfjmnos][a-z]* [\d]{1,2}, [\d]{2}"      # Jan 10, 1980, Jan 10, 80

  regex_3 = r"[adfjmnos][a-z]* [\d]{1,2}, [\d]{4}"      # Jan 10, 1980, Jan 10, 1980


  regex_4 = r"[adfjmnos][a-z]*\. [\d]{1,2}, [\d]{2}"    # Jan 10. 80, Jan 10. 1980
  regex_4 = r"[adfjmnos][a-z]*\. [\d]{1,2}, [\d]{4}"    # Jan 10. 80, Jan 10. 1980


  regex_5 = r"[ap|au|de|fe|ja|ju|ma|no|oc|se][a-z]* [\d]{2}"                 #
  regex_6 = r"[ap|au|de|fe|ja|ju|ma|no|oc|se][a-z]* [\d]{4}"                 #

  regex_7 = r"[adfjmnos][a-z]* [\d]{1,2}[a-z]*, [\d]{2}"  #

  regex_8 = r"[adfjmnos][a-z]* [\d]{1,2}[a-z]*, [\d]{4}"  #


    
  #
  ## Pattern for finding 
  #    6/18/85
  #    
  regex_9 = r'[\d]{1,2}[\/-][\d]{1,2}[\/-][\d]{2,4}'       # 7/11/77

  # regex_9 = r"(?<!\.)[\d]{1,2}[\/-][\d]{1,2}[\/-][\d]{2,4}"  # Uses negative look behind to find digits 
                                                           # that does not have a period(.) in front.

  
  regex_list = [regex_1, regex_2, regex_3, regex_4, regex_5, regex_6, regex_7, regex_8, regex_9]
  
  ENDC = '\033[0m'                                                                
  BOLD = "\033[1m"                                                                
                                                                                
  
  regex_found = False
  date_found = None
  for regex in regex_list: 

    print(f"Checking for regex {regex}")
    date_found = re.findall(regex, row)
    if(regex_found):
      print(f"  Line # {line_counter}: {row}")
      print(f"  Date found: {date_found}")
      print(f"  Pattern matched: {regex}")
      break
    else:
      print(f"  Line # {line_counter} NO MATCH: {row}")

      
  line_counter += 1
  print("--------------------------------")



