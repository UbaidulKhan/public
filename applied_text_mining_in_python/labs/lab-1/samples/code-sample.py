  # Following removes all alphabetical characters including months - jan, feb, march, etc.
  #
  # regex = '[a-z]+|\(+|\)+|\:+|\*|\_|\-\s|\.|\?|\[|\]|\;|d[3]|\s-\s|\s\/\s'
  # regex = '\(+|\)+|\:+|\*|\_|\-\s|\.|\?|\[|\]|\;|d[3]|\s-\s|\s\/\s'
  # regex = '\b(?!Jan\b|Feb\b|Mar\b|Apr\b|May\b|Jun\b|Jul\b|Aug\b|Sep\b|Oct\b|Nov\b|Dec\b'
  # regex = '\b(?!Jan\b|Feb\b|Mar\b|Apr\b|May\b|Jun\b|Jul\b|Aug\b|Sep\b|Oct\b|Nov\b|Dec\b)[A-Za-z]+'
  # regex1 = '\d{2}(\b(?!Jan\b|Feb\b|Mar\b|Apr\b|May\b|Jun\b|Jul\b|Aug\b|Sep\b|Oct\b|Nov\b|Dec\b)[A-Za-z]+)\d{2,4}'

  # new_row = re.sub(regex, '', row)
  #
  ## Regex for find dates in MM/DD/YYYY format:
  #
  #   r"[\d]{1,2}/[\d]{1,2}/[\d]{4}"gm
  #
  ## Regex for finding dates in "10 Oct 2015" format:
  #
  #   r"[\d]{1,2} [A|D|F|J|M|N|O]\w* [\d]{4}"gm
  #
  ## Regex for finding dates in "10 Oct 2015" or "10 oct 2015"
  #
  #   r"[\d]{1,2}\s(A|D|F|J|M|N|O|a|d|f|j|m|n|o)\w* [\d]{4}"gm
  #   r"[\d]{1,2}\s(A|D|F|J|M|N|O|a|d|f|j|m|n|o)\w*\s[\d]{4}"gm
  #   r"[\d]{1,2}\s(A|D|F|J|M|N|O|a|d|f|j|m|n|o)\w*\s[\d]{4}"gm
  #
  ## Regex for finding "10 Oct 2015" or "10 oct 2015" :
  #
  #   r"[\d]{1,2} [A|D|F|J|M|N|O|a|d|f|j|m|n|o]\w* [\d]{4}"gm
  #   r"[\d]{1,2}\s[ADFJMNOS]\w*\s[\d]{4}"gm
  #
  #  This will also pickup "10 Octshmeer 2015"
  #
  ## Regex for finding "10 Oct 2015"
  #
  #   r"[\d]{1,2}\s(Jan|Oct)\s[\d]{4}"gm
  #
  #  Will not pickup "10 Octshmeer 2015"
  #
  ## Regex for finding 10-10-15 or 10-10-2015
  #
  #   r"[\d]{1,2}-[\d]{1,2}-[\d]{2,4}"gm
