from fastapi import FastAPI, HTTPException
from app.schema import CustomerInput
from app.prediction import predict_customer

app = FastAPI(
    title="Data Driven Customer Segmentation API",
    description="A Machine Learning API that predicts customer segmentation.",
    version="1.0.0"
)

@app.get("/", tags=["Home"])
def home():
    return {
        "message": "Welcome to the Data Driven Customer Segmentation API",
        "author": "Aryan Dev",
        "model": "KMeans Clustering"
    }

@app.get("/Customer", tags=["Customer"])
def customer_check():
    return {"status": "API is running successfully"}

@app.post("/predict", tags=["Prediction"])
def predict(data: CustomerInput):
    try:
        result = predict_customer(
            data.Recency,
            data.Frequency,
            data.Monetary
        )

        return {
            "success": True,
            "Cluster": result["Cluster"],
            "Customer_Segment": result["Customer_Segment"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )