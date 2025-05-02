
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Working with Text Data in pandas




import pandas as pd

time_sentences = ["Monday: The doctor's appointment is at 2:45pm.", 
                  "Tuesday: The dentist's appointment is at 11:30 am.",
                  "Wednesday: At 7:00pm, there is a basketball game!",
                  "Thursday: Be back home by 11:15 pm at the latest.",
                  "Friday: Take the train at 08:10 am, arrive at 09:00am."]

df = pd.DataFrame(time_sentences, columns=['text'])
print(f"Data frame created from text:\n{df}")
print("\n")


# find the number of characters for each string in df['text']
num_chars_in_each_string = df['text'].str.len()
print(f"number of characters for each string in df\n{num_chars_in_each_string}")
print("\n")

# find the number of tokens for each string in df['text']
num_tokens_in_each_string = df['text'].str.split().str.len()
print(f"number of tokens in each string in df\n{num_tokens_in_each_string}")
print("\n")

# find all entries containing the word 'appointment'
entries_containing_appt = df['text'].str.contains('appointment')
print(f"Entries contaning the word appointment: \n{entries_containing_appt}")
print("\n")

# find how many times a digit occurs in each string
num_digits_in_each_string = df['text'].str.count(r'\d')
print(f"Number of times digits occur in each string: \n{num_digits_in_each_string}")
print("\n")


# find all occurances of the digits
occurances_of_digit = df['text'].str.findall(r'\d')
print(f"Number of occurances of digits: \n{num_digits_in_each_string}")
print("\n")


# group and find the hours and minutes
hours_mins_grouped = df['text'].str.findall(r'(\d?\d):(\d\d)')
print(f"Hours and minutes grouped together: \n{hours_mins_grouped}")
print("\n")


# replace weekdays with '???'
week_days_repalced_with_question = df['text'].str.replace(r'\w+day\b', '???')
print(f"Weekdays replaced with question mark: \n{week_days_repalced_with_question}")
print("\n")


# replace weekdays with 3 letter abbrevations
weekdays_replace_with_3letter_abbr = df['text'].str.replace(r'(\w+day\b)', lambda x: x.groups()[0][:3])
print(f"Weekdays replace with 3 letter abbre: \n{weekdays_replace_with_3letter_abbr}")


# create new columns from first match of extracted groups                     
new_columns_with_matched_digits = df['text'].str.extract(r'(\d?\d):(\d\d)')   
print(f"New columns with matched digitse: \n{new_columns_with_matched_digits}")


# extract the entire time, the hours, the minutes, and the period
new_columns_with_matched_digits_and_period = df['text'].str.extractall(r'((\d?\d):(\d\d) ?([ap]m))')
print(f"New columns with matched digits & period: \n{new_columns_with_matched_digits_and_period}") 



'''

# extract the entire time, the hours, the minutes, and the period with group names
df['text'].str.extractall(r'(?P<time>(?P<hour>\d?\d):(?P<minute>\d\d) ?(?P<period>[ap]m))')
''' 
