import pandas as pd
from tabulate import tabulate

# Create a sample DataFrame with long column names
df = pd.DataFrame({'MS. Jamison was not expected\n to attend the meeting': [1, 2, 3],
                   'AT the meeting, Ms. Jamison was absent': [4, 5, 6]})

# Modify the column names by truncating to a specified maximum width
max_width = 500
df.columns = [name[:max_width] if len(name) > max_width else name for name in df.columns]

# Convert DataFrame to tabular format
table = tabulate(df, headers='keys', tablefmt='psql')

# Display the tabulated data
print(table)