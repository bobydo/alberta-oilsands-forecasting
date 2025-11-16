import pandas as pd
import os
from src.common.logger import FileLogger

class ST53DataProcessor:
    """Processes ST53 Excel files from Alberta Energy Regulator."""
    
    @staticmethod
    def load(path: str) -> pd.DataFrame:
        """Load and transform ST53 Excel data from wide to long format. Args: path: Path to ST53 Excel file. Returns: DataFrame with columns: Operator, Scheme Name, Area, Approval Number, Recovery Method, Month, Bitumen."""
        logger = FileLogger.setup("preprocess_st53")
        
        try:
            logger.info(f"Loading ST53 data from: {path}")
            
            if not os.path.exists(path):
                error_msg = f"Excel file not found: {path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            df = pd.read_excel(path, header=3)
            logger.info(f"Loaded {len(df)} rows from Excel file")
            
            if df.empty:
                error_msg = f"Excel file is empty: {path}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            required_cols = ["Operator","Scheme Name","Area","Approval Number","Recovery Method"]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                error_msg = f"Missing required columns: {missing_cols}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            df_long = df.melt(
                id_vars=required_cols,
                value_vars=months+["Monthly Average"],
                var_name="Month",
                value_name="Bitumen"
            )
            
            df_clean = df_long.dropna()
            if df_clean.empty:
                error_msg = "No valid data after cleaning (all values are NaN)"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"Successfully processed {len(df_clean)} valid records")
            return df_clean
            
        except FileNotFoundError as e:
            logger.error(f"ST53 data file error: {e}")
            raise FileNotFoundError(f"ST53 data file error: {e}")
        except ValueError as e:
            logger.error(f"ST53 data validation error: {e}")
            raise ValueError(f"ST53 data validation error: {e}")
        except Exception as e:
            logger.error(f"ST53 data processing error: {e}", exc_info=True)
            raise Exception(f"ST53 data processing error: {e}")
