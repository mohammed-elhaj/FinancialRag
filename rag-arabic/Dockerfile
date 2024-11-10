# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Create a config directory for Streamlit
RUN mkdir -p .streamlit

# Create Streamlit config file
RUN echo "\
[server]\n\
port = 8080\n\
address = '0.0.0.0'\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
" > .streamlit/config.toml

# Expose port 8080
EXPOSE 8080

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
