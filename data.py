import pandas as pd
import numpy as np

# Generate 1000 random data entries based on the format provided
np.random.seed(42)

data = {
    'harga': np.random.randint(1, 6, size=1000),
    'baterai': np.random.randint(1, 6, size=1000),
    'kamera': np.random.randint(1, 6, size=1000),
    'RAM': np.random.randint(1, 6, size=1000),
    'memori_internal': np.random.randint(1, 6, size=1000),
    'tahun_rilis': np.random.randint(2017, 2022, size=1000),
    'kondisi_fisik': np.random.randint(1, 6, size=1000),
    'merk': np.random.choice(['iphone', 'vivo', 'samsung'], size=1000),
    'rating_pengguna': np.round(np.random.uniform(1.0, 5.0, size=1000), 1),
    'layak_beli': np.random.choice(['ya', 'tidak'], size=1000)
}

# Map 'ya' to 1 and 'tidak' to 0 for layak_beli column
data['layak_beli'] = np.where(data['layak_beli'] == 'ya', 1, 0)

# Create DataFrame
df = pd.DataFrame(data)

# Save to Excel
excel_file = 'handphone_data_1000.xlsx'
df.to_excel(excel_file, index=False)

excel_file
