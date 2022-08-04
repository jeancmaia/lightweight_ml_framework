"""
Base method for feature manipulation on the raw_data dataset. You should use this method 
for cumbersome feature creation steps, such as crafting new columns from actual covariates. 
An ordinary action that might declare as any classical transformer of scikit-sklearn, is that
to say null-imputation and scaling, must be implemented in this fashion.
"""
import dask.dataframe as dd
import dask.array as np

def extract_last_name(str_series: pd.Series):
    """Method to extract the last name of a name. It splits a string by the spaces and
    returns the last position.  
    
    Args:
        str_series (pd.Series): pandas column with names to extract the last name.
    """
    return str_series.split(' ')[-1]

if __name__ == '__main__':
    data = dd.read_csv('assets/raw_data.csv')
    
    data.to_csv('assets/cleaned_data.csv', index=False)