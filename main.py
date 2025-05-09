    import os
    import csv
    import io
    import re
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from typing import List, Dict, Any, Tuple, Optional
    import pdfplumber
    from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import firebase_admin
    from firebase_admin import credentials, firestore
    import uuid
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    import joblib
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
        # Use environment variables or a service account file
        try:
            # Try to use environment variables first
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
            # Fall back to service account file if available
            service_account_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "service-account.json")
            if os.path.exists(service_account_path):
                cred = credentials.Certificate(service_account_path)
            else:
                raise Exception(f"Firebase credentials not found: {e}")
        
        firebase_admin.initialize_app(cred)

    db = firestore.client()

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
            self.expense_categories = self.categories[:-3]  # All except the last 3
            self.income_categories = self.categories[-4:]   # Last 4 categories
            
            # Try to load pre-trained model if it exists
            try:
                self.model = joblib.load("category_model.joblib")
                print("Loaded pre-trained category prediction model")
            except:
                print("Training new category prediction model")
                self._train_model()
        
        def _train_model(self):
            # Sample training data - in production, this would be replaced with real data
            training_data = [
                # Food & Dining examples
                ("Restaurant", "Food & Dining"),
                ("Cafe", "Food & Dining"),
                ("Grocery", "Food & Dining"),
                ("Swiggy", "Food & Dining"),
                ("Zomato", "Food & Dining"),
                ("McDonald's", "Food & Dining"),
                ("Starbucks", "Food & Dining"),
                ("Food delivery", "Food & Dining"),
                
                # Transportation examples
                ("Uber", "Transportation"),
                ("Ola", "Transportation"),
                ("Petrol", "Transportation"),
                ("Fuel", "Transportation"),
                ("Metro", "Transportation"),
                ("Bus", "Transportation"),
                ("Train", "Transportation"),
                ("Taxi", "Transportation"),
                
                # Entertainment examples
                ("Movie", "Entertainment"),
                ("Netflix", "Entertainment"),
                ("Amazon Prime", "Entertainment"),
                ("Disney+", "Entertainment"),
                ("Hotstar", "Entertainment"),
                ("Concert", "Entertainment"),
                ("Game", "Entertainment"),
                ("Spotify", "Entertainment"),
                
                # Housing examples
                ("Rent", "Housing"),
                ("Mortgage", "Housing"),
                ("Property tax", "Housing"),
                ("Home insurance", "Housing"),
                ("Maintenance", "Housing"),
                ("Repair", "Housing"),
                
                # Utilities examples
                ("Electricity", "Utilities"),
                ("Water", "Utilities"),
                ("Gas", "Utilities"),
                ("Internet", "Utilities"),
                ("Mobile", "Utilities"),
                ("Phone", "Utilities"),
                ("Broadband", "Utilities"),
                ("Wifi", "Utilities"),
                
                # Shopping examples
                ("Amazon", "Shopping"),
                ("Flipkart", "Shopping"),
                ("Myntra", "Shopping"),
                ("Clothing", "Shopping"),
                ("Electronics", "Shopping"),
                ("Furniture", "Shopping"),
                ("Appliance", "Shopping"),
                
                # Healthcare examples
                ("Doctor", "Healthcare"),
                ("Hospital", "Healthcare"),
                ("Pharmacy", "Healthcare"),
                ("Medicine", "Healthcare"),
                ("Medical", "Healthcare"),
                ("Health insurance", "Healthcare"),
                ("Dental", "Healthcare"),
                
                # Education examples
                ("Tuition", "Education"),
                ("School", "Education"),
                ("College", "Education"),
                ("University", "Education"),
                ("Course", "Education"),
                ("Books", "Education"),
                ("Stationery", "Education"),
                
                # Travel examples
                ("Flight", "Travel"),
                ("Hotel", "Travel"),
                ("Booking", "Travel"),
                ("Vacation", "Travel"),
                ("MakeMyTrip", "Travel"),
                ("Airbnb", "Travel"),
                ("Resort", "Travel"),
                
                # Personal Care examples
                ("Salon", "Personal Care"),
                ("Haircut", "Personal Care"),
                ("Spa", "Personal Care"),
                ("Gym", "Personal Care"),
                ("Fitness", "Personal Care"),
                
                # Gifts & Donations examples
                ("Gift", "Gifts & Donations"),
                ("Donation", "Gifts & Donations"),
                ("Charity", "Gifts & Donations"),
                ("Present", "Gifts & Donations"),
                
                # Investments examples
                ("Investment", "Investments"),
                ("Mutual fund", "Investments"),
                ("Stock", "Investments"),
                ("Shares", "Investments"),
                ("Dividend", "Investments"),
                ("Fixed deposit", "Investments"),
                ("FD", "Investments"),
                
                # Salary examples
                ("Salary", "Salary"),
                ("Payroll", "Salary"),
                ("Income", "Salary"),
                ("Wage", "Salary"),
                ("Pay", "Salary"),
                ("Compensation", "Salary"),
                
                # Business examples
                ("Business", "Business"),
                ("Client", "Business"),
                ("Customer", "Business"),
                ("Sale", "Business"),
                ("Revenue", "Business"),
                ("Commission", "Business"),
                
                # Other Income examples
                ("Refund", "Other Income"),
                ("Reimbursement", "Other Income"),
                ("Cashback", "Other Income"),
                ("Interest", "Other Income"),
                ("Bonus", "Other Income"),
                
                # Other examples
                ("Miscellaneous", "Other"),
                ("Unknown", "Other"),
                ("General", "Other"),
                ("Various", "Other"),
            ]
            
            # Create and train the model
            X = [item[0] for item in training_data]
            y = [item[1] for item in training_data]
            
            self.model = Pipeline([
                ('vectorizer', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5))),
                ('classifier', MultinomialNB())
            ])
            
            self.model.fit(X, y)
            
            # Save the model for future use
            joblib.dump(self.model, "category_model.joblib")
        
        def predict_category(self, description: str, amount: float) -> Tuple[str, str]:
            """
            Predict category and transaction type (expense/income) based on description and amount
            """
            # Default to expense for negative amounts, income for positive
            transaction_type = "income" if amount > 0 else "expense"
            
            # Clean description
            clean_desc = re.sub(r'[^a-zA-Z0-9\s]', ' ', description).lower()
            
            # Predict category
            try:
                if self.model:
                    predicted_category = self.model.predict([clean_desc])[0]
                else:
                    # Fallback if model isn't loaded
                    predicted_category = "Other"
                    
                # Validate category based on transaction type
                if transaction_type == "expense" and predicted_category in self.income_categories:
                    predicted_category = "Other"
                elif transaction_type == "income" and predicted_category in self.expense_categories:
                    predicted_category = "Other Income"
                    
                return predicted_category, transaction_type
            except Exception as e:
                print(f"Error predicting category: {e}")
                # Default fallback
                return "Other" if transaction_type == "expense" else "Other Income", transaction_type

    # Initialize category predictor
    category_predictor = CategoryPredictor()

    # PDF Processing Functions
    def extract_text_from_pdf(pdf_file: bytes) -> str:
        """Extract all text from a PDF file"""
        text = ""
        try:
            with pdfplumber.open(io.BytesIO(pdf_file)) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
        return text

    def parse_pdf_statement(pdf_file: bytes) -> List[Dict[str, Any]]:
        """Parse a bank statement PDF and extract transactions"""
        text = extract_text_from_pdf(pdf_file)
        
        # This is a simplified approach - in production, you would need more robust patterns
        # for different bank statement formats
        
        # Common patterns for transaction entries in bank statements
        # Format: date, description, debit amount, credit amount
        transaction_patterns = [
            # Pattern 1: DD/MM/YYYY or DD-MM-YYYY followed by description and amount
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+([A-Za-z0-9\s\.,&\-/]+?)\s+((?:\d+,)*\d+\.\d{2})',
            
            # Pattern 2: YYYY-MM-DD format
            r'(\d{4}-\d{2}-\d{2})\s+([A-Za-z0-9\s\.,&\-/]+?)\s+((?:\d+,)*\d+\.\d{2})',
            
            # Add more patterns as needed for different bank formats
        ]
        
        transactions = []
        
        for pattern in transaction_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    date_str = match.group(1)
                    description = match.group(2).strip()
                    amount_str = match.group(3).replace(',', '')
                    
                    # Try to determine if it's a debit or credit
                    # This is simplified - real implementation would need to check context
                    is_debit = True
                    if "credit" in description.lower() or "deposit" in description.lower():
                        is_debit = False
                    
                    amount = float(amount_str)
                    if is_debit:
                        amount = -amount
                    
                    # Standardize date format
                    try:
                        # Try different date formats
                        for fmt in ('%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%d/%m/%y', '%d-%m-%y'):
                            try:
                                date_obj = datetime.strptime(date_str, fmt)
                                date_str = date_obj.strftime('%Y-%m-%d')
                                break
                            except ValueError:
                                continue
                    except Exception:
                        # If date parsing fails, keep original
                        pass
                    
                    # Predict category and transaction type
                    category, transaction_type = category_predictor.predict_category(description, amount)
                    
                    transactions.append({
                        'date': date_str,
                        'description': description,
                        'amount': abs(amount),  # Store absolute amount
                        'category': category,
                        'transaction_type': transaction_type
                    })
                except Exception as e:
                    print(f"Error parsing transaction: {e}")
                    continue
        
        return transactions

    # CSV Processing Functions
    def parse_csv_statement(csv_file: bytes) -> List[Dict[str, Any]]:
        """Parse a bank statement CSV and extract transactions"""
        try:
            # Read CSV into pandas DataFrame
            df = pd.read_csv(io.StringIO(csv_file.decode('utf-8')))
            
            # Normalize column names (lowercase and remove spaces)
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Map common column names to our standard format
            column_mappings = {
                # Date columns
                'date': ['date', 'transaction_date', 'txn_date', 'value_date'],
                # Description columns
                'description': ['description', 'narration', 'particulars', 'details', 'transaction_details'],
                # Amount columns
                'amount': ['amount', 'transaction_amount', 'debit', 'credit'],
                # Type columns (optional)
                'type': ['type', 'transaction_type', 'dr/cr']
            }
            
            # Find the best matching columns
            column_map = {}
            for our_col, possible_cols in column_mappings.items():
                for col in possible_cols:
                    if col in df.columns:
                        column_map[our_col] = col
                        break
            
            # Check if we have the minimum required columns
            required_cols = ['date', 'description']
            if not all(col in column_map for col in required_cols):
                raise ValueError(f"CSV is missing required columns. Found: {df.columns.tolist()}")
            
            # Handle amount columns - might be in separate debit/credit columns
            if 'amount' not in column_map:
                debit_col = next((col for col in df.columns if 'debit' in col), None)
                credit_col = next((col for col in df.columns if 'credit' in col), None)
                
                if debit_col and credit_col:
                    # Create a new amount column
                    df['amount'] = df[credit_col].fillna(0) - df[debit_col].fillna(0)
                    column_map['amount'] = 'amount'
                else:
                    raise ValueError("Could not find amount, debit, or credit columns")
            
            # Extract transactions
            transactions = []
            for _, row in df.iterrows():
                try:
                    # Get date and standardize format
                    date_str = str(row[column_map['date']])
                    try:
                        # Try different date formats
                        for fmt in ('%d/%m/%Y', '%d-%m-%Y', '%Y-%m-%d', '%d/%m/%y', '%d-%m-%y'):
                            try:
                                date_obj = datetime.strptime(date_str, fmt)
                                date_str = date_obj.strftime('%Y-%m-%d')
                                break
                            except ValueError:
                                continue
                    except Exception:
                        # If date parsing fails, keep original
                        pass
                    
                    # Get description
                    description = str(row[column_map['description']])
                    
                    # Get amount
                    amount = float(row[column_map['amount']])
                    
                    # Get transaction type if available, otherwise infer from amount
                    if 'type' in column_map:
                        type_value = str(row[column_map['type']]).lower()
                        if 'dr' in type_value or 'debit' in type_value:
                            amount = -abs(amount)
                        elif 'cr' in type_value or 'credit' in type_value:
                            amount = abs(amount)
                    
                    # Predict category and transaction type
                    category, transaction_type = category_predictor.predict_category(description, amount)
                    
                    transactions.append({
                        'date': date_str,
                        'description': description,
                        'amount': abs(amount),  # Store absolute amount
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

    # Database Functions
    async def save_transactions_to_db(transactions: List[Dict[str, Any]], user_id: str) -> Dict[str, Any]:
        """Save extracted transactions to Firestore database"""
        stats = {
            "total_transactions": len(transactions),
            "expenses_count": 0,
            "income_count": 0,
            "total_expense": 0,
            "total_income": 0,
            "categories": {}
        }
        
        try:
            # Process each transaction
            for transaction in transactions:
                try:
                    # Generate a unique ID
                    transaction_id = str(uuid.uuid4())
                    
                    # Prepare data for Firestore
                    transaction_data = {
                        "id": transaction_id,
                        "userId": user_id,
                        "amount": transaction["amount"],
                        "category": transaction["category"],
                        "description": transaction["description"],
                        "date": transaction["date"],
                        "createdAt": datetime.now().isoformat()
                    }
                    
                    # Determine collection based on transaction type
                    collection_name = "expenses" if transaction["transaction_type"] == "expense" else "income"
                    
                    # Add to Firestore
                    db.collection(collection_name).document(transaction_id).set(transaction_data)
                    
                    # Update stats
                    if transaction["transaction_type"] == "expense":
                        stats["expenses_count"] += 1
                        stats["total_expense"] += transaction["amount"]
                    else:
                        stats["income_count"] += 1
                        stats["total_income"] += transaction["amount"]
                    
                    # Update category stats
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
            
            # Update user metadata
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

    # API Endpoints
    @app.post("/process-statement", response_model=ProcessingResult)
    async def process_statement(
        file: UploadFile = File(...),
        user_id: str = Form(...),
    ):
        """
        Process a bank statement file (PDF or CSV) and extract transactions
        """
        try:
            # Read file content
            file_content = await file.read()
            
            # Process based on file type
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
            
            # Save transactions to database
            stats = await save_transactions_to_db(transactions, user_id)
            
            # Convert to Pydantic models for response
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

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    # Main entry point
    if __name__ == "__main__":
        import uvicorn
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
