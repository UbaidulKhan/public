#
## Following is frokm 1156
## 

  # feature_names = np.array(vectorizer.get_feature_names() + ['length_of_doc', 'digit_count', 'non_word_char_count'])
  feature_names = np.array(vectorizer.get_feature_names_out() + ['length_of_doc', 'digit_count', 'non_word_char_count'])
  # feature_names = np.array(vectorizer.get_feature_names_out())

  
  sorted_coef_index = classifier.coef_[0].argsort()
  smallest = feature_names[sorted_coef_index[:10]]
  largest = feature_names[sorted_coef_index[:-11:-1]]

  #
  ##  Also find the 10 smallest and 10 largest coefficients from the model and return 
  ##  them along with the AUC score in a tuple.
  coefficients = classifier.coef_  
  coefficients_list = coefficients.tolist()
  # print(f"| Coefficient contain: {len(coefficients_list)} tuples")
  
  ''' 
  for element in coefficients_list:
    print(f"{element}\n")
  
  print(f"| Coefficient values: {type(coefficients)}")
  '''
  
  ten_largest_coefs = coefficients[:10]
  ten_smallest_coefs = coefficients[:-11:-1]

  # for i in np.nditer(ten_smallest_coefs):
  #   print(f'| {i}\n')

  # print(f"| Largest coefficient data type is: {type(ten_largest_coefs)}")

  '''   
  print("+------------------------------------------------------------")
  print('| \033[1m 10 Largest coefficients:\033[0m')
  print("+------------------------------------------------------------")
  # for num in ten_largest_coefs: print(f"| {num}\n")
  for (idx, val) in enumerate(ten_largest_coefs):
    print(f"| {idx} Largest coefficient value is: {val}")

  print("+------------------------------------------------------------")
  print('| \033[1m 10 Smallest coefficients:\033[0m')
  print("+------------------------------------------------------------")
  # for num in ten_smallest_coefs: print(f"| {num}\n")
  for (idx, val) in enumerate(ten_smallest_coefs):
    print(f"| {idx} Smallest coefficient value is: {val}")
  print("+=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
  '''
