"""
Detector - Real-time fraud detection with ML
"""

import json
import time
import joblib
import pandas as pd
import psycopg2
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import os

# Configuration
MODEL_PATH = '../01_ml_offline/models/model.pkl'
SCALER_PATH = '../01_ml_offline/models/scaler.pkl'

KAFKA_BROKER = 'localhost:9092'
KAFKA_TOPIC = 'bank_transactions'
CONSUMER_GROUP = 'fraud_detection_group'

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'fraud_detection_db',
    'user': 'user',
    'password': 'password'
}

FRAUD_THRESHOLD = 0.5


def load_models():
    """Load XGBoost model and scaler from Phase 1"""
    print("Loading models from Phase 1...")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print(f"XGBoost model loaded: {MODEL_PATH}")
    print(f"StandardScaler loaded: {SCALER_PATH}")

    return model, scaler


def wait_for_kafka(max_retries=30, delay=1):
    """Wait for Kafka to be ready"""
    print("Waiting for Kafka...")

    for attempt in range(max_retries):
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_BROKER,
                group_id=CONSUMER_GROUP,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                auto_commit_interval_ms=1000
            )
            print(f"Connected to Kafka on {KAFKA_BROKER}")
            print(f"Subscribed to topic: {KAFKA_TOPIC}")
            return consumer
        except NoBrokersAvailable:
            print(f"Attempt {attempt + 1}/{max_retries}: Kafka unavailable, retrying in {delay}s...")
            time.sleep(delay)

    raise Exception(f"Could not connect to Kafka after {max_retries} attempts")


def wait_for_postgres(max_retries=30, delay=1):
    """Wait for PostgreSQL to be ready"""
    print("Waiting for PostgreSQL...")

    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            print("Connected to PostgreSQL")
            return conn
        except psycopg2.OperationalError:
            print(f"Attempt {attempt + 1}/{max_retries}: PostgreSQL unavailable, retrying in {delay}s...")
            time.sleep(delay)

    raise Exception(f"Could not connect to PostgreSQL after {max_retries} attempts")


def predict_fraud(model, scaler, transaction):
    """Predict if transaction is fraudulent"""
    features = []

    for i in range(1, 29):
        features.append(transaction[f'V{i}'])

    amount = transaction['Amount']
    amount_df = pd.DataFrame([[amount]], columns=['Amount'])
    amount_scaled = scaler.transform(amount_df)[0][0]

    features.append(amount_scaled)

    feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']
    X = pd.DataFrame([features], columns=feature_names)

    fraud_probability = model.predict_proba(X)[0][1]
    is_fraud = fraud_probability > FRAUD_THRESHOLD

    return is_fraud, fraud_probability


def save_prediction_to_db(cursor, transaction_id, is_fraud, fraud_probability):
    """Save prediction to PostgreSQL"""
    query = """
        INSERT INTO predictions (transaction_id, is_fraud, fraud_probability)
        VALUES (%s, %s, %s)
    """

    cursor.execute(query, (int(transaction_id), bool(is_fraud), float(fraud_probability)))


def main():
    """Main detector loop"""
    print("="*70)
    print("DETECTOR - Real-time Fraud Detection")
    print("="*70)

    model, scaler = load_models()

    print("\nWaiting for services to start (15 seconds)...")
    time.sleep(15)

    consumer = wait_for_kafka()
    conn = wait_for_postgres()
    cursor = conn.cursor()

    print(f"\nFraud threshold: {FRAUD_THRESHOLD * 100}%")
    print("\n" + "="*70)
    print("Starting detection...")
    print("="*70 + "\n")

    prediction_count = 0
    fraud_count = 0

    try:
        for message in consumer:
            try:
                transaction = message.value
                transaction_id = transaction['transaction_id']

                is_fraud, fraud_probability = predict_fraud(model, scaler, transaction)

                save_prediction_to_db(cursor, transaction_id, is_fraud, fraud_probability)
                conn.commit()

                prediction_count += 1
                if is_fraud:
                    fraud_count += 1

                if is_fraud:
                    print(f"🔴 FRAUD ALERT! | "
                          f"Transaction #{transaction_id} | "
                          f"Amount: {transaction['Amount']:.2f}€ | "
                          f"Probability: {fraud_probability*100:.2f}% | "
                          f"Total frauds: {fraud_count}/{prediction_count}")
                else:
                    if prediction_count % 50 == 0:
                        print(f"🟢 Transaction OK | "
                              f"Transaction #{transaction_id} | "
                              f"Amount: {transaction['Amount']:.2f}€ | "
                              f"Probability: {fraud_probability*100:.2f}% | "
                              f"Total: {prediction_count}")

            except Exception as e:
                print(f"Error processing message: {e}")
                conn.rollback()
                continue

    except KeyboardInterrupt:
        print("\n\nManual stop")
        print(f"\nFinal statistics:")
        print(f"   • Total predictions: {prediction_count:,}")
        print(f"   • Frauds detected: {fraud_count:,}")
        if prediction_count > 0:
            print(f"   • Fraud rate: {fraud_count/prediction_count*100:.2f}%")
    except Exception as e:
        print(f"\nFatal error: {e}")
    finally:
        print("\nClosing connections...")
        consumer.close()
        cursor.close()
        conn.close()
        print("Detector stopped")


if __name__ == "__main__":
    main()
