#
## Pattern for finding:
#    6 Oct 85
#    6 October 1985
#
regex_1 = r"[\d]{1,2} [adfjmnso][a-z]* [\d]{2,4}"

#
## Pattern for finding:
#   6/6/1998
#   
regex_2 = r"(?<!\.)[\d]{1,2}\/[\d]{1,2}\/[\d]{2,4}"      

#
## Pattern for finding:
# 6/1998
#   
regex_3 = r"(?<!\.)[\d]{1,2}\/[\d]{2,4}"    


#
## Pattern for finding:
#    Oct 6, 85
#    October 6, 1985
# 
regex_4 = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* [\d]{1,2}, [\d]{2,4}"      # Jan 10, 1980, Jan 10, 80


#
## Pattern for finding:
#    Oct. 6 85
#    Jan 10. 80, Jan 10. 1980
# 
regex_5 = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\. [\d]{1,2}, [\d]{2,4}"  

#
## Pattern for finding:
#    October 6 85
# 
regex_6 = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* [\d]{1,2} [\d]{2,4}"    # jan 24 1986

#
## Matches Jan 20, January 2022
regex_7 = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* [\d]{2,4}\, [\d]{2,4}"


#
## Matches september 1985
regex_8 = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* [\d]{2,4}"
  
#
## Matches January 10, 2022
regex_9 = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* [\d]{1,2}[a-z]*, [\d]{2,4}"  #


#
## Matches January, 2022
regex_10 = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*, [\d]{2,4}"  #

  
#
## Pattern for finding 
#  Matches 06/06/2022, 06-06-2022
#  Will not match .06-06-2022
# Uses negative look behind to find digits that does not have a period(.) in front.
regex_11 = r"(?<!\.)[\d]{1,2}[\/-][\d]{1,2}[\/-][\d]{2,4}"  

#
## Matches 4 digit year only - 1984
regex_12 = r"[\d]{4}" 


regex_list = [regex_1, regex_2, regex_3, regex_4, regex_5, regex_6, regex_7, regex_8, regex_9, regex_10, regex_11, regex_12]
