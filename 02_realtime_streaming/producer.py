"""
Producer - Real-time bank transaction streaming simulation
"""

import pandas as pd
import psycopg2
import json
import time
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# Configuration
CSV_PATH = '../01_ml_offline/data/test/creditcard_test.csv'  # Test set only (unseen data)

KAFKA_BROKER = 'localhost:9092'
KAFKA_TOPIC = 'bank_transactions'

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'fraud_detection_db',
    'user': 'user',
    'password': 'password'
}

TRANSACTION_DELAY = 0.05
CHUNK_SIZE = 1000

def wait_for_kafka(max_retries=30, delay=1):
    """Wait for Kafka to be ready"""
    print("Waiting for Kafka...")

    for attempt in range(max_retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3
            )
            print(f"Connected to Kafka on {KAFKA_BROKER}")
            return producer
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


def insert_transaction_to_db(cursor, transaction):
    """Insert transaction into PostgreSQL with ground truth (Class)"""
    columns = ['time', 'amount'] + [f'v{i}' for i in range(1, 29)] + ['class']
    values_placeholder = ', '.join(['%s'] * len(columns))
    columns_str = ', '.join(columns)

    query = f"""
        INSERT INTO transactions ({columns_str})
        VALUES ({values_placeholder})
        RETURNING id
    """

    values = [transaction['Time'], transaction['Amount']] + \
             [transaction[f'V{i}'] for i in range(1, 29)] + \
             [int(transaction['Class'])]

    cursor.execute(query, values)
    transaction_id = cursor.fetchone()[0]

    return transaction_id


def send_to_kafka(producer, transaction, transaction_id):
    """Send transaction to Kafka topic"""
    message = {
        'transaction_id': transaction_id,
        'Time': float(transaction['Time']),
        'Amount': float(transaction['Amount']),
        **{f'V{i}': float(transaction[f'V{i}']) for i in range(1, 29)}
    }

    producer.send(KAFKA_TOPIC, value=message)


def main():
    """Main producer loop"""
    print("="*70)
    print("PRODUCER - Bank Transaction Streaming")
    print("="*70)

    print("\nWaiting for services to start (15 seconds)...")
    time.sleep(15)

    producer = wait_for_kafka()
    conn = wait_for_postgres()
    cursor = conn.cursor()

    print(f"\nDataset: {CSV_PATH}")
    print(f"Transaction delay: {TRANSACTION_DELAY}s")
    print(f"Kafka topic: {KAFKA_TOPIC}")
    print("\n" + "="*70)
    print("Starting transaction stream...")
    print("="*70 + "\n")

    transaction_count = 0

    try:
        for chunk in pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE):
            for index, row in chunk.iterrows():
                try:
                    transaction = row.to_dict()

                    transaction_id = insert_transaction_to_db(cursor, transaction)
                    conn.commit()

                    send_to_kafka(producer, transaction, transaction_id)

                    transaction_count += 1

                    if transaction_count % 100 == 0:
                        fraud_status = "🔴 FRAUD" if transaction.get('Class', 0) == 1 else "🟢 OK"
                        print(f"Transaction #{transaction_count:,} sent | "
                              f"ID: {transaction_id} | "
                              f"Amount: {transaction['Amount']:.2f}€ | "
                              f"Status: {fraud_status}")

                    time.sleep(TRANSACTION_DELAY)

                except Exception as e:
                    print(f"Error processing transaction: {e}")
                    conn.rollback()
                    continue

        print("\n" + "="*70)
        print(f"Completed - {transaction_count:,} transactions processed")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\nManual stop")
    except Exception as e:
        print(f"\nFatal error: {e}")
    finally:
        print("\nClosing connections...")
        producer.flush()
        producer.close()
        cursor.close()
        conn.close()
        print("Producer stopped")


if __name__ == "__main__":
    main()
