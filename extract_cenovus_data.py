"""Extract last 8 months of Cenovus production data for API testing."""
from src.st53.preprocess_st53 import ST53DataProcessor

# Load ST53 data
df = ST53DataProcessor.load('data/st53/ST53_2024-12.xls')

# Filter for Cenovus operations
cenovus = df[df['Operator'].str.contains('Cenovus', case=False, na=False)]

print("="*70)
print("CENOVUS ENERGY - SAGD Operations")
print("="*70)

for scheme in cenovus['Scheme Name'].unique():
    scheme_data = cenovus[cenovus['Scheme Name'] == scheme]
    last_8 = scheme_data['Bitumen'].tail(8).tolist()
    
    print(f"\n{scheme}:")
    print(f"  Total records: {len(scheme_data)} months")
    print(f"  Last 8 months (m³): {last_8}")
    print(f"\n  API Test JSON:")
    print(f'  {{"values": {last_8}}}')

print("\n" + "="*70)
print("COMBINED (All Cenovus SAGD operations)")
print("="*70)
all_last_8 = cenovus['Bitumen'].tail(8).tolist()
print(f"Last 8 values: {all_last_8}")
print(f'\nAPI Test JSON:\n{{"values": {all_last_8}}}')
