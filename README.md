# 🚀 Data-Driven Customer Segmentation API

A Machine Learning project that segments customers based on purchasing behavior using **RFM Analysis (Recency, Frequency, Monetary)** and **K-Means Clustering**, deployed as a **FastAPI** service.

---

## 📌 Project Overview

This project analyzes customer transaction data and groups customers into meaningful segments such as:

* 💰 High Value Customers
* 🔁 Regular Customers
* ⚠️ At Risk Customers
* 💤 Low Value Customers

The goal is to help businesses make **data-driven marketing decisions** and improve customer retention.

---

## 🧠 Tech Stack

* **Python**
* **Pandas, NumPy**
* **Scikit-learn (KMeans, StandardScaler, PCA)**
* **FastAPI**
* **Uvicorn**
* **Pickle (Model Serialization)**

---

## 📂 Project Structure

```
Customer_Segmentation/
│
├── app/
│   ├── main.py              # FastAPI application
│   ├── prediction.py        # Prediction logic
│   ├── schema.py            # Input/Output schemas
│   ├── model_loader.py      # Load model & scaler
│   ├── __init__.py
│
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│
├── training/
│   ├── elbow_method.py
│   ├── kmeans_training.py
│
├── data/
│   ├── OnlineRetail.csv
│   ├── rfm_dataset.csv
│
├── README.md
```

---

## ⚙️ How It Works

1. Raw transactional data is converted into **RFM dataset**
2. Data is **scaled using StandardScaler**
3. Optimal clusters selected using **Elbow Method**
4. **K-Means clustering** is applied
5. Customers are labeled into segments
6. Model is saved as `.pkl` and served via FastAPI

---

## 🚀 Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/customer-segmentation-api.git
cd customer-segmentation-api
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run FastAPI Server

```bash
uvicorn app.main:app --reload
```

---

### 4. Open in Browser

* API Home:
  http://127.0.0.1:8000/

* Swagger Docs (Test API):
  http://127.0.0.1:8000/docs

---

## 📥 API Usage

### 🔹 POST `/predict`

#### Request:

```json
{
  "Recency": 10,
  "Frequency": 5,
  "Monetary": 1000
}
```

#### Response:

```json
{
  "success": true,
  "Cluster": 0,
  "Customer_Segment": "High Value"
}
```

---

## 📊 Model Details

* Algorithm: **K-Means Clustering**
* Features Used:

  * Recency
  * Frequency
  * Monetary
* Preprocessing:

  * Standard Scaling
* Visualization:

  * PCA (2D projection)

---

## 💡 Business Use Cases

* Targeted Marketing Campaigns
* Customer Retention Strategies
* Personalized Recommendations
* Identifying High-Value Customers

---

## 🔥 Future Improvements

* Add **Silhouette Score evaluation**
* Deploy API on **Render / Railway**
* Build frontend dashboard (Streamlit / React)
* Real-time customer segmentation

---

## 👨‍💻 Author

**Aryan Dev**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share it!

---
