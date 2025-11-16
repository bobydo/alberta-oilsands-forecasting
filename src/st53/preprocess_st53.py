import pandas as pd
class ST53DataProcessor:
    """Processes ST53 Excel files from Alberta Energy Regulator."""
    @staticmethod
    def load(path: str) -> pd.DataFrame:
        """Load and transform ST53 Excel data from wide to long format. Args: path: Path to ST53 Excel file. Returns: DataFrame with columns: Operator, Scheme Name, Area, Approval Number, Recovery Method, Month, Bitumen."""
        df = pd.read_excel(path, header=3)
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        df_long = df.melt(
            id_vars=["Operator","Scheme Name","Area","Approval Number","Recovery Method"],
            value_vars=months+["Monthly Average"],
            var_name="Month",
            value_name="Bitumen"
        )
        return df_long.dropna()
