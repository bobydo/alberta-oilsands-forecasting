import pandas as pd

class ST39DataProcessor:
    """Processes ST39 Excel files from Alberta Energy Regulator."""
    
    @staticmethod
    def load(path: str) -> pd.DataFrame:
        """Load and transform ST39 Excel data from wide to long format. Args: path: Path to ST39 Excel file. Returns: DataFrame with columns: Category, Month, Production."""
        # Read from Crude Bitumen section (row 96 has months, row 99 has production)
        df = pd.read_excel(path, sheet_name="VAR0800-ST39Extracts_xls", header=96)
        
        # Get production row (row 2 after header: Opening Inventory, Receipts, Production)
        production_row = df.iloc[2:3].copy()
        
        # Month columns are in positions 1-12 (January-December)
        months = ["January", "February", "March", "April", "May", "June", 
                  "July", "August", "September", "October", "November", "December"]
        
        # Extract production values for each month
        data = []
        for i, month in enumerate(months, start=1):
            value = production_row.iloc[0, i]
            data.append({"Month": month, "Production": value})
        
        df_long = pd.DataFrame(data)
        return df_long.dropna()
