from fastapi import FastAPI
from pydantic import BaseModel

from src.inference.predict import predict

app = FastAPI(title="German Credit Risk API")


class Applicant(BaseModel):
    checking_account_status: str
    duration_months: int
    credit_history: str
    purpose: str
    credit_amount: int
    savings_account: str
    employment_years: str
    installment_rate_pct: int
    personal_status_sex: str
    other_debtors: str
    residence_since_years: int
    property: str
    age: int
    other_installment_plans: str
    housing: str
    num_existing_credits: int
    job: str
    num_dependents: int
    telephone: str
    foreign_worker: str


@app.get("/")
def root():
    return {"status": "German Credit Risk API is running"}


@app.post("/predict")
def predict_endpoint(applicant: Applicant):
    result = predict(applicant.model_dump())
    return result