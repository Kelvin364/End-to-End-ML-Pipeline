version: "3.9"

services:

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
      - SUPABASE_BUCKET=${SUPABASE_BUCKET:-Upload Image}
      - RETRAIN_THRESHOLD=${RETRAIN_THRESHOLD:-50}
      - RETRAIN_CHECK_HOURS=${RETRAIN_CHECK_HOURS:-1}
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3

  ui:
    image: python:3.10-slim
    working_dir: /app
    command: >
      sh -c "pip install -q streamlit requests matplotlib pandas pillow numpy
      && streamlit run app.py --server.port 8501 --server.address 0.0.0.0"
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    volumes:
      - .:/app
    depends_on:
      - api
    restart: unless-stopped

# Scale the API for load testing:
#   1 container : docker-compose up --scale api=1
#   2 containers: docker-compose up --scale api=2
#   3 containers: docker-compose up --scale api=3
#
# Add a load balancer (nginx) for multi-container setups:
#
#   nginx:
#     image: nginx:alpine
#     ports:
#       - "80:80"
#     volumes:
#       - ./nginx.conf:/etc/nginx/nginx.conf:ro
#     depends_on:
#       - api
