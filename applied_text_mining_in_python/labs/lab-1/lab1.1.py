import pandas as pd
import re
from regex import *

'''
 To do:
   https://stackoverflow.com/questions/10715965/create-a-pandas-dataframe-by-appending-one-row-at-a-time
'''

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


'''
  
'''
def inspect_missing_date_fields(date, regex, clean_date_array) -> None:

  month_dict = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 
                'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}

  day_month_year = date[0]
  day_month_year = re.sub('[\.,]', '', day_month_year)
  

  if(regex == regex_1):
    # day_month_year = date[0]
    day_month_year_split = day_month_year.split(' ')
    day = day_month_year_split[0]
    month = day_month_year_split[1]
    month_short =  month[0:3]
    month_numeric = month_dict.get(month_short)
    year = day_month_year_split[2]   
    date = f"{month_numeric}/{day}/{year}"
    
    date_dict= {'date': date}
    clean_date_array.append(date_dict)
    # print(f"  Corrected date is: {date}")


  if(regex == regex_2):
    # day_month_year = date[0]
    day_month_year_split = day_month_year.split('/')
    day = day_month_year_split[1]
    month = day_month_year_split[0]
    year = day_month_year_split[2]
    year_len = len(year)
    if(year_len == 2):
      year = '19' + year
    date = f"{month}/{day}/{year}"
    
    date_dict= {'date': date}
    clean_date_array.append(date_dict)
    # print(f"  Corrected date is: {date}")
            
  if(regex == regex_3):
    day = 1
    month_year = date[0]
    month_year_split = month_year.split('/')
    month = month_year_split[0]
    year = month_year_split[1]
    date = f"{month}/{day}/{year}"

    date_dict= {'date': date}
    clean_date_array.append(date_dict)
    # print(f"  Corrected date is: {date}")
    
  if(regex == regex_4):
    # day_month_year = date[0]
    day_month_year_split = day_month_year.split(' ')
    day = day_month_year_split[1]
    month = day_month_year_split[0]
    month_short =  month[0:3]
    month_numeric = month_dict.get(month_short)
    year = day_month_year_split[2]   
    year_len = len(year)
    if(year_len == 2):
      year = '19' + year
    date = f"{month_numeric}/{day}/{year}"

    date_dict= {'date': date}
    clean_date_array.append(date_dict)
    # print(f"  Corrected date is: {date}")

  if(regex == regex_5):
    # day_month_year = date[0]
    day_month_year_split = day_month_year.split(' ')
    day = day_month_year_split[1]
    month = day_month_year_split[0]
    month_short =  month[0:3]
    month_numeric = month_dict.get(month_short)
    year = day_month_year_split[2]   
    year_len = len(year)
    if(year_len == 2):
      year = '19' + year
    date = f"{month_numeric}/{day}/{year}"

    date_dict= {'date': date}
    clean_date_array.append(date_dict)
    # print(f"  Corrected date is: {date}")

  if(regex == regex_6):
    # day_month_year = date[0]
    day_month_year_split = day_month_year.split(' ')
    day = day_month_year_split[1]
    month = day_month_year_split[0]
    month_short =  month[0:3]
    month_numeric = month_dict.get(month_short)
    year = day_month_year_split[2]   
    year_len = len(year)
    if(year_len == 2):
      year = '19' + year
    date = f"{month_numeric}/{day}/{year}"

    date_dict= {'date': date}
    clean_date_array.append(date_dict)
    # print(f"  Corrected date is: {date}")

  if(regex == regex_8):
    day_month_year_split = day_month_year.split(' ')
    day = 1
    month = day_month_year_split[0]
    year = day_month_year_split[1]
    month_short =  month[0:3]
    month_numeric = month_dict.get(month_short)
    date = f"{month_numeric}/{day}/{year}"

    date_dict= {'date': date}
    clean_date_array.append(date_dict)
    # print(f"  Corrected date is: {date}")

  if(regex == regex_10):
    day_month_year_split = day_month_year.split(' ')
    day = 1
    month = day_month_year_split[0]
    year = day_month_year_split[1]
    month_short =  month[0:3]
    month_numeric = month_dict.get(month_short)
    date = f"{month_numeric}/{day}/{year}"

    date_dict= {'date': date}
    clean_date_array.append(date_dict)
    # print(f"  Corrected date is: {date}")
              
  if(regex == regex_11):
    # day_month_year = date[0]
    day_month_year_split = day_month_year.split('-')
    day = day_month_year_split[1]
    month = day_month_year_split[0]
    year = day_month_year_split[2]   
    year_len = len(year)
    if(year_len == 2):
      year = '19' + year
    date = f"{month}/{day}/{year}"

    date_dict= {'date': date}
    clean_date_array.append(date_dict)
    # print(f"  Corrected date is: {date}")
        
  if(regex == regex_12):
    month = 1
    day = 1
    date = f"{month}/{day}/{date[0]}"

    date_dict= {'date': date}
    clean_date_array.append(date_dict)
    # print(f"  Corrected date is: {date}")



'''
  
'''
def display_data_frame(sorted_data_frame) -> None:
  counter = 0
  for index, row in sorted_data_frame .iterrows():
    print(f"{counter} {index} {row['date']}")
    counter += 1


'''
  
'''
def read_date_from_file(lower_case_df, sorted_data_frame) -> None:

  #
  ## Create a list for storing all the dates
  clean_date_array = []

  #
  ## Create a pandas data frame for storing final sorted data
  sorted_data_frame = pd.DataFrame(columns=['date'])
  
  #
  ## Set an initializer for the for lopp
  line_counter = 1
  
  for row in lower_case_df:
    #
    # Remove:
    #
    #  1) All strings containing a to z [a-z]+
    #  2) Remove open ( and close ) paranthesis
    #  3) Remove numbers with more than 4 consecutive digits
    #
  
    
    regex_found = None
    for regex in regex_list: 
  
      month = None
      day = None
      
      # print(f"Checking for regex {regex}")
      regex_found = re.findall(regex, row)
      # print(f"  Type of date found is: {type(regex_found)}")
  
      if(regex_found):
        # print(f"Line # {line_counter}: {row}")
        # print(f"Line # {line_counter}")

        # print(f"  Regex matched: {regex_found}")
        # print(f"  Pattern matched: {regex}")
  
        #
        ## It is possible we only found year, then we need to populate month and day with 1 and 1
        ## call inspect_missing_date_fields to fill in missing fields
        inspect_missing_date_fields(regex_found, regex, clean_date_array)
        
        # Since a pattern has matched, break out of the loop
        break
        
      '''
      else:
        print(f"\nLine: {line_counter} {regex} did not match: {row}")
      '''
  
    line_counter += 1
    # print("\n--------------------------------")
    
  # 
  ## Create a new data-frame from the array of dictionaries
  sorted_data_frame = pd.DataFrame(clean_date_array)
  
  
  #
  ## Convert column to date
  sorted_data_frame['date'] = pd.to_datetime(sorted_data_frame['date'], format='%m/%d/%Y')
  
  #
  ## Sort the data-frame in ascending order of the date
  sorted_data_frame = sorted_data_frame.sort_values(by=['date'], ascending=True)
  
  #
  ## Print the data-frame in the format required for grading
  display_data_frame(sorted_data_frame)


  
  ''' 
  with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
    print(sorted_data_frame)  
  '''
    
  '''
  sorted_data_frame['date'] = pd.to_datetime(sorted_data_frame['date'])

  sorted_data_frame = sorted_data_frame.sort_values(by=['date'], ascending=True)


  with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
    print(sorted_data_frame)
  '''



'''
  
'''
def main():
  date = None
  
  doc = []
  file_name = 'dates.txt'
  with open('dates.txt') as file:
      for line in file:
          doc.append(line)
 
  #
  ## Create a data-frame and store the list into the data-frame
  lower_case_df = pd.DataFrame()
  df = pd.Series(doc)
  
  #
  ## Create a data-frame for storing sorted dates
  sorted_data_frame = pd.DataFrame()
 
  #
  ## Convert the data frame to lower case
  lower_case_df = df.str.lower()
  
  
  read_date_from_file(lower_case_df, sorted_data_frame)


'''
  if main is defined, then call main 
'''
if __name__ == '__main__':
    main()

