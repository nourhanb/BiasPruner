

import pandas as pd

# Read the CSV file
df = pd.read_csv("/ubc/ece/home/ra/grads/nourhanb/Documents/miccai_2024/GbP/GbP/Data_Entry_2017.csv")

# Define the classes
classes = ['Atelectasis', 'Infiltration', 'Consolidation', 'Pneumothorax', 'Edema',
           'Emphysema', 'Mass', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
           'Cardiomegaly', 'Nodule', 'Hernia']

# Filter rows based on conditions for all classes
filtered_df = df[df['FindingLabels'].isin(classes)]

# Save the filtered DataFrame to a new CSV file
filtered_df.to_csv("NIH_filtered_14classes.csv", index=False)
print("NIH_filtered_14classes.csv")
