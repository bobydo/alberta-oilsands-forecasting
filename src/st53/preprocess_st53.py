import pandas as pd
def load_st53(path):
    df=pd.read_excel(path,header=3)
    months=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    df_long=df.melt(id_vars=["Operator","Scheme Name","Area","Approval Number","Recovery Method"],
                    value_vars=months+["Monthly Average"],var_name="Month",value_name="Bitumen")
    return df_long.dropna()
