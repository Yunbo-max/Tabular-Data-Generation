import os
import sys
import pandas as pd
import torch
from tabula import Tabula

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_IB_DISABLE"] = "1"

# Print environment variable value
nccl_ib_disable = os.getenv('NCCL_IB_DISABLE')
print(f"The current value for NCCL_IB_DISABLE is: {nccl_ib_disable}")

# Load and preprocess data
data = pd.read_excel('/home/yl892/rds/hpc-work/Tabular-Data-Generation/Tabula-main/Real_Datasets/dataco_final.xlsx')
dataset = data
dataset['Customer Full Name'] = dataset['customer_fname'].astype(str) + dataset['customer_lname'].astype(str)

data = dataset.drop(['customer_email', 'customer_id', 'customer_password', 'customer_fname', 'customer_lname',
                     'product_description', 'product_image_url', 'order_zipcode', 'product_status', 'order_profit_per_order',
                     'product_price', 'order_city_en', 'order_city', 'order_country', 'customer_street'], axis=1)

data['customer_zipcode'] = data['customer_zipcode'].fillna(0)  # Fill NaN columns with zero

# Time series features
data['order_year'] = pd.DatetimeIndex(data['order_date']).year
data['order_month'] = pd.DatetimeIndex(data['order_date']).month
data['order_week_day'] = pd.DatetimeIndex(data['order_date']).day_name()
data['order_hour'] = pd.DatetimeIndex(data['order_date']).hour
data['shipping_year'] = pd.DatetimeIndex(data['shipping_date']).year
data['shipping_month'] = pd.DatetimeIndex(data['shipping_date']).month
data['shipping_week_day'] = pd.DatetimeIndex(data['shipping_date']).day_name()
data['shipping_hour'] = pd.DatetimeIndex(data['shipping_date']).hour

# Prepare data for modeling
label_data = data[['shipping_week_day', 'order_week_day', 'payment_type', 'category_name', 'customer_city', 
                   'customer_country', 'customer_segment', 'customer_state', 'department_name', 'market',
                   'order_country_en', 'order_state', 'order_status', 'product_name', 'shipping_mode', 
                   'order_region', 'order_city_corrected']]

data['target'] = data['delivery_status']
data = data.drop(['delivery_status'], axis=1)

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_columns)

categorical_columns = ['payment_type', 'category_name', 'customer_city', 'customer_country',
                       'customer_segment', 'customer_state', 'department_name', 'market',
                       'order_city_corrected', 'order_country_en', 'order_region',
                       'order_state', 'order_status', 'product_name', 'shipping_mode',
                       'Customer Full Name', 'order_week_day', 'shipping_week_day', 'target']

# Initialize and train model
model = Tabula(llm='distilgpt2', experiment_dir="insurance_training", batch_size=32, epochs=400, categorical_columns=categorical_columns)

# Load pretrained model
# Uncomment the following line if using pretrained model
# model.model.load_state_dict(torch.load("pretrained-model/adult_king_insurance_intrusion_covtype_pretrained.pt"), strict=False)

# Train model
model.fit(data)

# Save model
torch.save(model.model.state_dict(), "insurance_training/model_400epoch.pt")

# Generate synthetic data and save to CSV
synthetic_data = model.sample(n_samples=1338)
synthetic_data.to_csv("insurance_400epoch.csv", index=False)
