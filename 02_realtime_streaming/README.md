
## Aperçu

##  Résumé
Système complet de détection de fraude bancaire capable de traiter des milliers de transactions par seconde.
Combine le **Machine Learning (XGBoost)** pour la précision et une architecture temps réel avec  **(Kafka)** 

##  Architecture Technique
* **Ingestion :** Simulation de flux bancaire (Python Producer)
* **Streaming :** Apache Kafka & Zookeeper
* **Traitement :** Consumer Python (Simulation Spark Streaming)
* **ML Engine :** Modèle XGBoost entraîné sur dataset déséquilibré (SMOTE)
* **Stockage :** PostgreSQL (Transactions & Prédictions)
* **Visualisation :** Grafana Live Dashboard

## Comment lancer le projet
1. `docker-compose up -d`
2. `python src/producer.py`
3. `python src/detector.py`