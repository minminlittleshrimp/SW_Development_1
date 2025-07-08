FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/ ./data/
COPY tests/ ./tests/
COPY notebooks/ ./notebooks/
COPY models/ ./models/
COPY . .

# Command to run the training script
CMD ["python", "src/training/train.py"]