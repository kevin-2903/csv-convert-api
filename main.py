# 2. Import Libraries
import os
import csv
import io
import re
import numpy as np
from datetime import datetime
import pandas as pd
import fitz  # PyMuPDF
import pdfplumber
from io import BytesIO
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, firestore
import uuid
import json

# Initialize FastAPI app
app = FastAPI(title="Financial Statement Analyzer")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Firebase (if not already initialized)
if not firebase_admin._apps:
    try:
        cred_dict = {
            "type": "service_account",
            "project_id": os.environ.get("FIREBASE_PROJECT_ID"),
            "private_key_id": os.environ.get("FIREBASE_PRIVATE_KEY_ID", ""),
            "private_key": os.environ.get("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n"),
            "client_email": os.environ.get("FIREBASE_CLIENT_EMAIL"),
            "client_id": os.environ.get("FIREBASE_CLIENT_ID", ""),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.environ.get("FIREBASE_CLIENT_CERT_URL", "")
        }
        cred = credentials.Certificate(cred_dict)
    except Exception as e:
        service_account_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "service-account.json")
        if os.path.exists(service_account_path):
            cred = credentials.Certificate(service_account_path)
        else:
            raise Exception(f"Firebase credentials not found: {e}")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# OCR for ICICI
def extract_text_using_ocr(pdf_bytes):
    images = convert_from_bytes(pdf_bytes)
    text = ""
    for i, image in enumerate(images):
        page_text = pytesseract.image_to_string(image)
        print(f"[OCR Page {i+1}]")
        print(page_text)
        text += page_text + "\n"
    return text

# Define models
class Transaction(BaseModel):
    date: str
    description: str
    amount: float
    category: str
    transaction_type: str  # "expense" or "income"

class ProcessingResult(BaseModel):
    success: bool
    message: str
    transactions: List[Transaction] = []
    stats: Dict[str, Any] = {}

# Category prediction model
class CategoryPredictor:
    def __init__(self):
        self.model = None
        self.categories = [
            "Food & Dining", "Transportation", "Entertainment", "Housing",
            "Utilities", "Shopping", "Healthcare", "Education", "Travel",
            "Personal Care", "Gifts & Donations", "Investments", "Salary",
            "Business", "Other Income", "Other"
        ]
        self.expense_categories = self.categories[:-3]
        self.income_categories = self.categories[-4:]

        try:
            self.model = joblib.load("category_model.joblib")
            print("Loaded pre-trained category prediction model")
        except:
            print("Training new category prediction model")
            self._train_model()

    def _train_model(self):
        training_data = [
            ("Restaurant", "Food & Dining"), ("Cafe", "Food & Dining"), ("Grocery", "Food & Dining"),
            ("Swiggy", "Food & Dining"), ("Zomato", "Food & Dining"), ("McDonald's", "Food & Dining"),
            ("Starbucks", "Food & Dining"), ("Food delivery", "Food & Dining"),
            ("Uber", "Transportation"), ("Ola", "Transportation"), ("Petrol", "Transportation"),
            ("Fuel", "Transportation"), ("Metro", "Transportation"), ("Bus", "Transportation"),
            ("Train", "Transportation"), ("Taxi", "Transportation"),
            ("Movie", "Entertainment"), ("Netflix", "Entertainment"), ("Amazon Prime", "Entertainment"),
            ("Disney+", "Entertainment"), ("Hotstar", "Entertainment"), ("Concert", "Entertainment"),
            ("Game", "Entertainment"), ("Spotify", "Entertainment"),
            ("Rent", "Housing"), ("Mortgage", "Housing"), ("Property tax", "Housing"),
            ("Home insurance", "Housing"), ("Maintenance", "Housing"), ("Repair", "Housing"),
            ("Electricity", "Utilities"), ("Water", "Utilities"), ("Gas", "Utilities"),
            ("Internet", "Utilities"), ("Mobile", "Utilities"), ("Phone", "Utilities"),
            ("Broadband", "Utilities"), ("Wifi", "Utilities"),
            ("Amazon", "Shopping"), ("Flipkart", "Shopping"), ("Myntra", "Shopping"),
            ("Clothing", "Shopping"), ("Electronics", "Shopping"), ("Furniture", "Shopping"),
            ("Appliance", "Shopping"),
            ("Doctor", "Healthcare"), ("Hospital", "Healthcare"), ("Pharmacy", "Healthcare"),
            ("Medicine", "Healthcare"), ("Medical", "Healthcare"), ("Health insurance", "Healthcare"),
            ("Dental", "Healthcare"),
            ("Tuition", "Education"), ("School", "Education"), ("College", "Education"),
            ("University", "Education"), ("Course", "Education"), ("Books", "Education"),
            ("Stationery", "Education"),
            ("Flight", "Travel"), ("Hotel", "Travel"), ("Booking", "Travel"),
            ("Vacation", "Travel"), ("MakeMyTrip", "Travel"), ("Airbnb", "Travel"),
            ("Resort", "Travel"),
            ("Salon", "Personal Care"), ("Haircut", "Personal Care"), ("Spa", "Personal Care"),
            ("Gym", "Personal Care"), ("Fitness", "Personal Care"),
            ("Gift", "Gifts & Donations"), ("Donation", "Gifts & Donations"),
            ("Charity", "Gifts & Donations"), ("Present", "Gifts & Donations"),
            ("Investment", "Investments"), ("Mutual fund", "Investments"),
            ("Stock", "Investments"), ("Shares", "Investments"), ("Dividend", "Investments"),
            ("Fixed deposit", "Investments"), ("FD", "Investments"),
            ("Salary", "Salary"), ("Payroll", "Salary"), ("Income", "Salary"),
            ("Wage", "Salary"), ("Pay", "Salary"), ("Compensation", "Salary"),
            ("Business", "Business"), ("Client", "Business"), ("Customer", "Business"),
            ("Sale", "Business"), ("Revenue", "Business"), ("Commission", "Business"),
            ("Refund", "Other Income"), ("Reimbursement", "Other Income"),
            ("Cashback", "Other Income"), ("Interest", "Other Income"), ("Bonus", "Other Income"),
            ("Miscellaneous", "Other"), ("Unknown", "Other"), ("General", "Other"), ("Various", "Other"),
        ]
        X = [item[0] for item in training_data]
        y = [item[1] for item in training_data]
        self.model = Pipeline([
            ('vectorizer', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))),
            ('classifier', MultinomialNB())
        ])
        self.model.fit(X, y)
        joblib.dump(self.model, "category_model.joblib")

    def predict_category(self, description: str, amount: float) -> Tuple[str, str]:
        transaction_type = "income" if amount > 0 else "expense"
        clean_desc = re.sub(r'[^a-zA-Z0-9\s]', ' ', description).lower()
        try:
            if self.model:
                predicted_category = self.model.predict([clean_desc])[0]
            else:
                predicted_category = "Other"
            if transaction_type == "expense" and predicted_category in self.income_categories:
                predicted_category = "Other"
            elif transaction_type == "income" and predicted_category in self.expense_categories:
                predicted_category = "Other Income"
            return predicted_category, transaction_type
        except Exception as e:
            print(f"Error predicting category: {e}")
            return "Other" if transaction_type == "expense" else "Other Income", transaction_type

category_predictor = CategoryPredictor()

def clean_amount(amount_str: str) -> float:
    if not amount_str or amount_str.strip() == '-':
        return 0.0
    try:
        return float(amount_str.replace(',', '').strip())
    except ValueError:
        return 0.0

def preprocess_statement_text(text: str) -> str:
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\n(-[A-Z0-9]+.*?)\n', r' \1\n', text)
    text = re.sub(r'(\d{2}/\d{2}/\d{2} .+?)\n\s*(\d{4,20}-)', r'\1 \2', text)
    return text

def extract_text_from_pdf(pdf_file: bytes) -> str:
    try:
        with fitz.open(stream=pdf_file, filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            if not text.strip():
                print("Trying OCR...")
                with tempfile.TemporaryDirectory() as tempdir:
                    images = convert_from_bytes(pdf_file, dpi=300)
                    for image in images:
                        text += pytesseract.image_to_string(image)
            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def parse_pdf_statement(pdf_file: bytes) -> List[Dict[str, Any]]:
    text = extract_text_from_pdf(pdf_file)
    text = preprocess_statement_text(text)
    bank_format = "unknown"
    if "HDFC BANK" in text.upper():
        bank_format = "hdfc"
    elif "KOTAK MAHINDRA BANK" in text.upper():
        bank_format = "kotak"
    elif "STATE BANK OF INDIA" in text.upper():
        bank_format = "sbi"
    elif "ICICI BANK" in text.upper():
        bank_format = "icici"
    elif "KVB" in text.upper() or "KARUR VYSYA BANK" in text.upper():
        bank_format = "kvb"
    transactions = []
    if bank_format == "hdfc":
        pattern = re.compile(
            r'(?P<date>\d{2}/\d{2}/\d{2})\s+(?P<description>.+?)\s+[A-Z0-9-]+\s+'
            r'(?P<valuedate>\d{2}/\d{2}/\d{2})\s+'
            r'(?P<amount>[0-9,]+\.\d{2})\s+(?P<balance>[0-9,]+\.\d{2})',
            re.DOTALL
        )
        last_balance = None
        matches = pattern.finditer(text)
        for match in matches:
            try:
                groups = match.groupdict()
                description = groups['description'].strip().replace('\n', ' ')
                balance = clean_amount(groups['balance'])
                if last_balance is not None:
                    if balance > last_balance:
                        txn_type = "income"
                        amount = balance - last_balance
                    elif balance < last_balance:
                        txn_type = "expense"
                        amount = last_balance - balance
                    else:
                        txn_type = "unknown"
                        amount = clean_amount(groups['amount'])
                else:
                    txn_type = "unknown"
                    amount = clean_amount(groups['amount'])
                last_balance = balance
                date_str = datetime.strptime(groups['date'], "%d/%m/%y").strftime('%Y-%m-%d')
                category, _ = category_predictor.predict_category(description, amount)
                transactions.append({
                    'date': date_str,
                    'description': description,
                    'amount': round(amount, 2),
                    'category': category,
                    'transaction_type': txn_type,
                    'balance': balance
                })
            except Exception as e:
                print(f"Error parsing HDFC transaction at line: {match.group(0)} - {e}")
                continue
    elif bank_format == "icici":
        print("Detected ICICI Mini Statement with flexible parsing")
        if not text or len(text.strip()) < 50:
            print("ICICI detected blank or image-based statement, switching to OCR...")
            text = extract_text_using_ocr(pdf_file)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines:
            try:
                match = re.match(r"^(\d{2}/\d{2}/\d{4})\s+(.+?)\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})$", line)
                if match:
                    date_str, description, amount_str, balance_str = match.groups()
                    date_obj = datetime.strptime(date_str, "%d/%m/%Y")
                    date_str = date_obj.strftime("%Y-%m-%d")
                    amount = float(amount_str.replace(',', ''))
                    balance = float(balance_str.replace(',', ''))
                    upper_desc = description.upper()
                    txn_type = "income" if "CREDITED" in upper_desc or any(k in upper_desc for k in ["REFUND", "CREDIT", "SALARY", "RECEIVED", "FROM"]) else "expense"
                    category, _ = category_predictor.predict_category(description, amount)
                    transactions.append({
                        "date": date_str,
                        "description": description.strip(),
                        "amount": amount,
                        "balance": balance,
                        "category": category,
                        "transaction_type": txn_type
                    })
            except Exception as e:
                print(f"[ICICI Parse Error]: {e}")
                continue
    elif bank_format == "kotak":
        pattern = r'(\d{2}-\d{2}-\d{4})\s+(.+?)\s+[A-Z0-9\-]+?\s+([\d,]+\.\d{2})\(?(Dr|Cr)\)?\s+([\d,]+\.\d{2})\(Cr\)'
        matches = re.finditer(pattern, text)
        for match in matches:
            try:
                date_str = datetime.strptime(match.group(1), "%d-%m-%Y").strftime("%Y-%m-%d")
                description = match.group(2).strip()
                amount = float(match.group(3).replace(',', ''))
                drcr = match.group(4).lower()
                txn_type = "expense" if drcr == "dr" else "income"
                if txn_type == "expense":
                    amount = -amount
                category, _ = category_predictor.predict_category(description, amount)
                transactions.append({
                    'date': date_str,
                    'description': description,
                    'amount': abs(amount),
                    'category': category,
                    'transaction_type': txn_type
                })
            except Exception as e:
                print(f"Error parsing Kotak transaction: {e}")
                continue
    elif bank_format == "sbi":
        print("Detected SBI Mini Statement with advanced regressive parsing")
        lines = [line.strip() for line in text.split("\n")]
        i = 0
        while i < len(lines) - 2:
            try:
                if any(keyword in lines[i].upper() for keyword in ["ACCOUNT", "BRANCH", "IFSC", "MICR", "BALANCE AS ON", "ADDRESS", "INTEREST", "POWER", "NAME", "CIF"]):
                    i += 1
                    continue
                if re.match(r"^\d{2} \w{3} \d{4}$", lines[i]):
                    date_str = lines[i]
                    i += 1
                    desc_lines = []
                    numeric_values = []
                    while i < len(lines):
                        line = lines[i].strip()
                        if re.match(r"^\d{2} \w{3} \d{4}$", line):
                            break
                        elif re.match(r"^\d[\d,]*\.\d{2}$", line):
                            numeric_values.append(float(line.replace(",", "")))
                            if len(numeric_values) == 2:
                                i += 1
                                break
                        else:
                            desc_lines.append(line)
                        i += 1
                    if len(numeric_values) < 1:
                        continue
                    amount = numeric_values[0]
                    description = " ".join(desc_lines).strip()
                    txn_type = "income" if "CR/" in description.upper() else "expense"
                    try:
                        date_str = datetime.strptime(date_str, "%d %b %Y").strftime("%Y-%m-%d")
                    except:
                        pass
                    category, _ = category_predictor.predict_category(description, amount)
                    transactions.append({
                        "date": date_str,
                        "description": description,
                        "amount": amount,
                        "category": category,
                        "transaction_type": txn_type
                    })
                else:
                    i += 1
            except Exception as e:
                print(f"[SBI Parse Error near line {i}]: {e}")
                i += 1
                continue
    elif bank_format == "kvb":
        with pdfplumber.open(io.BytesIO(pdf_file)) as pdf:
            for page in pdf.pages:
                try:
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if row and len(row) >= 5:
                                date_str = row[0]
                                particulars = row[3]
                                debit = row[4]
                                credit = row[5] if len(row) > 5 else None
                                if re.match(r"\d{2}/\d{2}/\d{4}", date_str):
                                    try:
                                        date_obj = datetime.strptime(date_str, "%d/%m/%Y")
                                        date_str = date_obj.strftime("%Y-%m-%d")
                                    except:
                                        continue
                                    amount = None
                                    txn_type = None
                                    if debit and re.match(r"[\d,]+\.\d{2}", debit):
                                        amount = float(debit.replace(",", ""))
                                        txn_type = "expense"
                                    elif credit and re.match(r"[\d,]+\.\d{2}", credit):
                                        amount = float(credit.replace(",", ""))
                                        txn_type = "income"
                                    if amount:
                                        category, _ = category_predictor.predict_category(particulars, amount)
                                        transactions.append({
                                            "date": date_str,
                                            "description": particulars,
                                            "amount": amount,
                                            "category": category,
                                            "transaction_type": txn_type
                                        })
                except Exception as e:
                    print(f"[Page Error]: {e}")
    else:
        transaction_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+([A-Za-z0-9\s\.,&\-/]+?)\s+((?:\d+,)*\d+\.\d{2})',
            r'(\d{4}-\d{2}-\d{2})\s+([A-Za-z0-9\s\.,&\-/]+?)\s+((?:\d+,)*\d+\.\d{2})',
        ]
        for pattern in transaction_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    date_str = match.group(1)
                    description = match.group(2).strip()
                    amount_str = match.group(3).replace(',', '')
                    amount = float(amount_str)
                    txn_type = "income" if "credit" in description.lower() or "deposit" in description.lower() else "expense"
                    for fmt in ('%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%d/%m/%y', '%d-%m-%y'):
                        try:
                            date_str = datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
                            break
                        except ValueError:
                            continue
                    category, transaction_type = category_predictor.predict_category(description, amount)
                    transactions.append({
                        'date': date_str,
                        'description': description,
                        'amount': abs(amount),
                        'category': category,
                        'transaction_type': transaction_type
                    })
                except Exception as e:
                    print(f"Error parsing generic transaction: {e}")
                    continue
    return transactions

def parse_csv_statement(csv_file: bytes) -> List[Dict[str, Any]]:
    try:
        df = pd.read_csv(io.StringIO(csv_file.decode('utf-8')))
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        column_mappings = {
            'date': ['date', 'transaction_date', 'txn_date', 'value_date'],
            'description': ['description', 'narration', 'particulars', 'details', 'transaction_details'],
            'amount': ['amount', 'transaction_amount', 'debit', 'credit'],
            'type': ['type', 'transaction_type', 'dr/cr']
        }
        column_map = {}
        for our_col, possible_cols in column_mappings.items():
            for col in possible_cols:
                if col in df.columns:
                    column_map[our_col] = col
                    break
        required_cols = ['date', 'description']
        if not all(col in column_map for col in required_cols):
            raise ValueError(f"CSV is missing required columns. Found: {df.columns.tolist()}")
        if 'amount' not in column_map:
            debit_col = next((col for col in df.columns if 'debit' in col), None)
            credit_col = next((col for col in df.columns if 'credit' in col), None)
            if debit_col and credit_col:
                df['amount'] = df[credit_col].fillna(0) - df[debit_col].fillna(0)
                column_map['amount'] = 'amount'
            else:
                raise ValueError("Could not find amount, debit, or credit columns")
        transactions = []
        for _, row in df.iterrows():
            try:
                date_str = str(row[column_map['date']])
                try:
                    for fmt in ('%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%d/%m/%y', '%d-%m-%y'):
                        try:
                            date_obj = datetime.strptime(date_str, fmt)
                            date_str = date_obj.strftime('%Y-%m-%d')
                            break
                        except ValueError:
                            continue
                except Exception:
                    pass
                description = str(row[column_map['description']])
                amount = float(row[column_map['amount']])
                if 'type' in column_map:
                    type_value = str(row[column_map['type']]).lower()
                    if 'dr' in type_value or 'debit' in type_value:
                        amount = -abs(amount)
                    elif 'cr' in type_value or 'credit' in type_value:
                        amount = abs(amount)
                category, transaction_type = category_predictor.predict_category(description, amount)
                transactions.append({
                    'date': date_str,
                    'description': description,
                    'amount': abs(amount),
                    'category': category,
                    'transaction_type': transaction_type
                })
            except Exception as e:
                print(f"Error processing row: {e}")
                continue
        return transactions
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return []

async def save_transactions_to_db(transactions: List[Dict[str, Any]], user_id: str) -> Dict[str, Any]:
    stats = {
        "total_transactions": len(transactions),
        "expenses_count": 0,
        "income_count": 0,
        "total_expense": 0,
        "total_income": 0,
        "categories": {}
    }
    try:
        for transaction in transactions:
            try:
                transaction_id = str(uuid.uuid4())
                transaction_data = {
                    "id": transaction_id,
                    "userId": user_id,
                    "amount": transaction["amount"],
                    "category": transaction["category"],
                    "description": transaction["description"],
                    "date": transaction["date"],
                    "createdAt": datetime.now().isoformat()
                }
                collection_name = "expenses" if transaction["transaction_type"] == "expense" else "income"
                db.collection(collection_name).document(transaction_id).set(transaction_data)
                if transaction["transaction_type"] == "expense":
                    stats["expenses_count"] += 1
                    stats["total_expense"] += transaction["amount"]
                else:
                    stats["income_count"] += 1
                    stats["total_income"] += transaction["amount"]
                category = transaction["category"]
                if category not in stats["categories"]:
                    stats["categories"][category] = {
                        "count": 0,
                        "total": 0
                    }
                stats["categories"][category]["count"] += 1
                stats["categories"][category]["total"] += transaction["amount"]
            except Exception as e:
                print(f"Error saving transaction: {e}")
                continue
        try:
            user_ref = db.collection("users").document(user_id)
            user_ref.set({
                "lastStatementUpload": datetime.now().isoformat(),
                "statementUploadStats": stats
            }, merge=True)
        except Exception as e:
            print(f"Error updating user metadata: {e}")
        return stats
    except Exception as e:
        print(f"Error in save_transactions_to_db: {e}")
        return {"error": str(e)}

@app.post("/process-statement", response_model=ProcessingResult)
async def process_statement(
    file: UploadFile = File(...),
    user_id: str = Form(...),
):
    try:
        file_content = await file.read()
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension == '.pdf':
            transactions = parse_pdf_statement(file_content)
        elif file_extension == '.csv':
            transactions = parse_csv_statement(file_content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a PDF or CSV file.")
        if not transactions:
            return ProcessingResult(
                success=False,
                message="No transactions could be extracted from the file. Please check the file format.",
                transactions=[],
                stats={}
            )
        stats = await save_transactions_to_db(transactions, user_id)
        transaction_models = [
            Transaction(
                date=t["date"],
                description=t["description"],
                amount=t["amount"],
                category=t["category"],
                transaction_type=t["transaction_type"]
            ) for t in transactions
        ]
        return ProcessingResult(
            success=True,
            message=f"Successfully processed {len(transactions)} transactions",
            transactions=transaction_models,
            stats=stats
        )
    except Exception as e:
        print(f"Error processing statement: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
