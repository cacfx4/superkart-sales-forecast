# Base Image

FROM python:3.11-slim


# Working Directory

WORKDIR /app


# Install Dependencies

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# Copy Application Files

COPY . .


# Expose Streamlit Port


EXPOSE 8501


# Start Streamlit


CMD ["streamlit",
     "run",
     "streamlit_app.py",
     "--server.port=8501",
     "--server.address=0.0.0.0"]
