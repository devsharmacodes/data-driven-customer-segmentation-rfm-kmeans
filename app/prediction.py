import pickle
import numpy as np


# 1. Load Model & Scaler

with open("models/kmeans_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)



# 2. Segment Mapping Function

def get_customer_segment(cluster):
    if cluster == 0:
        return "High Value"
    elif cluster == 1:
        return "Regular"
    elif cluster == 2:
        return "At Risk"
    else:
        return "Low Value"



# 3. Prediction Function

def predict_customer(recency, frequency, monetary):
    
    # Convert input to numpy array
    data = np.array([[recency, frequency, monetary]])
    
    # Scale input
    data_scaled = scaler.transform(data)
    
    # Predict cluster
    cluster = int(model.predict(data_scaled)[0])
    
    # Get segment label
    segment = get_customer_segment(cluster)
    
    return {
        "Cluster": cluster,
        "Customer_Segment": segment
    }


