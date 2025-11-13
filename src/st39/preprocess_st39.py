import pandas as pd
def load_st39(path):
    df=pd.read_excel(path,sheet_name="VAR0800",header=16)
    months=["January","February","March","April","May","June","July","August","September","October","November","December"]
    df_long=df.melt(id_vars=[df.columns[0]],value_vars=months+["2024"],
                    var_name="Month",value_name="Production")
    return df_long.dropna()
