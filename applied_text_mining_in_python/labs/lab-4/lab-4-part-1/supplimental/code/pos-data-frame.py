
import pandas as pd
import nltk

#
## Reference(s):
#
#  https://pandas.pydata.org/docs/dev/user_guide/merging.html
#
#

def df_incremental_update():
  doc = 'Ms Stewart, the chief executive, was not expected to attend.'
  pos_df = pd.DataFrame(columns=[doc])
  
  tokens = nltk.word_tokenize(doc.lower())
  
  #
  ## Add the synset to df_entry_string after converting it to string
  df_entry_string = ''
  
  df_row_index = 0
  
  for token in tokens:
    
    #
    ## Create a dictionary 
    df_entry_dict = {doc:token}
    # print(f" | \_(dts) Data-framw dict ** {df_entry_dict} **")
  
    #
    ## Create a data-frame using the dictionary 
    new_df_row = pd.DataFrame(df_entry_dict, index=[df_row_index])
          
          
    # print(f" | \_(dts) Data-framw row ** {new_df_row} **")

  
    # pos_df = pos_df.append({doc:df_entry_string},ignore_index=True)
  
    frames = [pos_df, new_df_row]

    pos_df = pd.concat(frames)
    
    df_row_index = df_row_index + 1
    
  print(pos_df)

df_incremental_update()  
  
  


