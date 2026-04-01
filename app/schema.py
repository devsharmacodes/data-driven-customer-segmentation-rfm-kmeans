from pydantic import BaseModel

class CustomerInput(BaseModel):
    Recency: int
    Frequency: int
    Monetary: float

class CustomerOutput(BaseModel):
    Recency: int
    Frequency: int
    Monetary: float

    R: int
    F: int
    M: int

    RFM_Group: int
    RFM_Score: int

    Segment: str
    Cluster: int
    Customer_Segment: str