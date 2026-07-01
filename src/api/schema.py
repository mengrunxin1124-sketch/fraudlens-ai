from pydantic import BaseModel


class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    oldbalanceDest: float
