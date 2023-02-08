# Process

## Description of input features
## Import libs
## load dataset
## exploratory data analysis
### Understanding of dataset
#### head of the dataset
#### shape of the dataset
#### list types of all cols
    df.types
#### information 
    df.info()
#### summary of the dataset
    df.describe()
    
### Data cleaning
#### Drop the duplicate
    df = df.drop_duplicate()
    check shape
#### check null values
    df.isnull.sum()
#### check zero values and replace with mean values
    ```python 
    df['Glucose']= df[df['Glucose']==o].shape[0]
    # replace
    
    df['Glucose']= df['Glucose'].replace(0,df['BloodPressure'].mean())
    ```
## Data Visualization
