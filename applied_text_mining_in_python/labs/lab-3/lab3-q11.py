def answer_eleven(spam_data_df):
  
  print("\n")  
  print("+------------------------------------------------------------")
  print("|  \033[1m Question 11 \033[0m")
  print("+------------------------------------------------------------")

  #
  ## df consists of two columns:
  ##  - text 
  ##  - target 
  
  #
  ## Print column names
  '''     
  # for col in spam_data_df.columns:
  for col in X_train.columns:
    
      print(f'   Column name is: {col}')
  '''

  print("| \033[1m Vectorizing Training Data \033[0m")

    
  #
  ## Create a Tfidf Vectorizer ignoring terms that have a document frequency 
  ## strictly lower than 5. 
  vectorizer = CountVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb')

  #
  ## Learn the vocabulary dictionary and return document-term matrix.
  X_train_vectorized = vectorizer.fit_transform(X_train)
  
  #
  ## Display the CountVectorizer vocabulary & length:
  # print(f"|  Vectorizer has : {len(vectorizer.vocabulary_)} tokens")
  # print(f"|  Vocabulary in Vectorizing: {vectorizer.vocabulary_}")

  
  #
  ## summarize encoded vector
  print(f"| \033[1m Vectorizer shape(rows/columns): {X_train_vectorized.shape} tokens \033[0m ")
  # print(vector.toarray())
  
  # print(f"|  Vectorizing shape after fit_transform: {vectorizer.shape}")

  
  #
  ## From X_train(pandas data series) get additional features
  (X_train_doclen, X_train_numdigits, X_train_nonalpha) = feature_extractor(X_train) 
  
  #
  ## Iterate over the data returned by feature_extractor and add each
  ## individual feature series to X_train_vectorized
  for feature in (X_train_doclen, X_train_numdigits, X_train_nonalpha):
    print(f"|  nameof(feature) is of type: {type(feature)}")
    # print(f"|  Appending feature {nameof(feature)} to X_train_vectorized")
    X_train_vectorized = add_feature(X_train_vectorized, feature)
    # print(f"|  nameof(X_train_vectorized) is of type: {type(X_train_vectorized)}")

    
  print("| \033[1m Vectorizing Testing Data \033[0m")

  #
  ## Create a sparse-matrix from X_test 
  X_test_vectorized = vectorizer.transform(X_test)
  
  ## From X_train(pandas data series) get additional features
  (X_test_doclen, X_test_numdigits, X_test_nonalpha) = feature_extractor(X_test) 
  
  # X_train_vectorized = add_feature(X_train_vectorized, X_train.str.len())
  # X_test_vectorized = add_feature(X_test_vectorized, X_test_doclen)
  # X_test_vectorized = add_feature(X_test_vectorized, X_test.str.len())

  #
  ## Iterate over the data returned by feature_extractor and add each
  ## individual feature series to X_test_vectorized
  run_count=0
  for feature in (X_test_doclen, X_test_numdigits, X_test_nonalpha):
    # print(f"|  Appending feature {nameof(feature)} to X_test_vectorized")
    # print(f"| \033[1m Vectorizer after reshaping {run_count} \033[0m ")
    # print(f"| \033[1mshape(rows/columns): {X_train_vectorized.shape} tokens \033[0m ")
    X_test_vectorized = add_feature(X_test_vectorized, feature)
    run_count += 1
  
  #
  ## Create a logistic-regression 
  # classifier = SVC(C=10000)
  classifier = LogisticRegression(C=100, solver='liblinear')
  
  #
  ## Fit the training data 
  classifier.fit(X_train_vectorized, y_train)
  
  #
  ## Predict the X_test_vectorizer
  y_predicted = classifier.predict(X_test_vectorized)
  
  #
  ## Calculate the AUC score
  auc = roc_auc_score(y_test, y_predicted)
  


  # feature_names = np.array(vectorizer.get_feature_names() + ['length_of_doc', 'digit_count', 'non_word_char_count'])
  # feature_names = np.array(vectorizer.get_feature_names_out() + ['length_of_doc', 'digit_count', 'non_word_char_count'])
  # feature_names = np.array(vectorizer.get_feature_names_out())

  
  sorted_coef_index = classifier.coef_[0].argsort()
  print(f"|  Sorted coefficients are of type: {type(sorted_coef_index)}")
  # smallest = feature_names[sorted_coef_index[:10]]
  # largest = feature_names[sorted_coef_index[:-11:-1]]

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

