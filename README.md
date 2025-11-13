# Alberta Oil Sands Forecasting (ST53 + ST39)

A unified machine learning system for forecasting Alberta oil sands production using real AER datasets:

- **ST53** – Cenovus In-Situ (SAGD) Bitumen Production  
- **ST39** – Mineable Oil Sands Plant Production  

This repository includes:

- Full preprocessing pipelines  
- TensorFlow LSTM forecasting models  
- Production-grade FastAPI inference service  
- Dockerized deployment  
- CI/CD automation  
- Evaluation module  
- Architecture diagrams  

## 🚀 Features
- End-to-end ML pipeline  
- Real AER XLS file ingestion  
- Modular architecture  
- FastAPI inference endpoints  
- Docker support  
- CI/CD GitHub Actions pipeline  

## 📁 Repository Structure
```
src/
  common/
  st53/
  st39/
api/
models/
data/
README.md
Dockerfile
requirements.txt
.github/workflows/ci.yml
```

## 🔧 Training
```
python src/st53/train_st53.py data/ST53_2024-12.xls models/
python src/st39/train_st39.py data/ST39-2024.xls models/
```

## 🌐 API Endpoints
- `/sagd/predict`
- `/mining/predict`

## 🐳 Docker
```
docker build -t oilsands-forecast .
docker run -p 8080:8080 oilsands-forecast
```

