# Alberta Oil Sands Forecasting (ST53 + ST39)

A unified machine learning system for forecasting Alberta oil sands production using real AER datasets:

- **ST53** – Cenovus In-Situ (SAGD - Steam-Assisted Gravity Drainage) Bitumen Production  
- **ST39** – Mineable Oil Sands Plant Production  

This repository includes:

- Full preprocessing pipelines  
- TensorFlow LSTM (Long Short-Term Memory) forecasting models  
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
```bash
python -m src.st53.train_st53 data/st53/ST53_2024-12.xls models/
python -m src.st39.train_st39 data/st39/ST39-2024.xls models/
```

## 🌐 API Endpoints
- `/sagd/predict`
- `/mining/predict`

## 🐳 Docker
```
docker build -t oilsands-forecast .
docker run -p 8080:8080 oilsands-forecast
```

