Installation
1. Clone the project

Code

git clone 

cd Renewable-Energy-Forecaster 

2. Create a virtual environment

Code

python -m venv venv

source venv/bin/activate   # Mac/Linux

venv\Scripts\activate      # Windows

3. Install dependencies

Code

pip install -r requirements.txt

▶️ Running the API

Start the FastAPI server:

Code

uvicorn app.main:app --reload

Open your browser:

http://127.0.0.1:8000
