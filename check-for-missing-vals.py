# Check for missing values
print(data.isnull().sum())

# Fill or drop missing values
data = data.dropna()  # or use data.fillna(method='ffill') for forward fill

## this code is not used here, chedk ut out in the notebook/model.