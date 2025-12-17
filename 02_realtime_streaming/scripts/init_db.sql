-- =============================================================================
-- Script d'initialisation de la base de données - Détection de Fraude
-- =============================================================================

-- Table: transactions
-- Stores all bank transactions with ground truth (class)
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    time FLOAT NOT NULL,
    amount FLOAT NOT NULL,
    v1 FLOAT,
    v2 FLOAT,
    v3 FLOAT,
    v4 FLOAT,
    v5 FLOAT,
    v6 FLOAT,
    v7 FLOAT,
    v8 FLOAT,
    v9 FLOAT,
    v10 FLOAT,
    v11 FLOAT,
    v12 FLOAT,
    v13 FLOAT,
    v14 FLOAT,
    v15 FLOAT,
    v16 FLOAT,
    v17 FLOAT,
    v18 FLOAT,
    v19 FLOAT,
    v20 FLOAT,
    v21 FLOAT,
    v22 FLOAT,
    v23 FLOAT,
    v24 FLOAT,
    v25 FLOAT,
    v26 FLOAT,
    v27 FLOAT,
    v28 FLOAT,
    class INTEGER NOT NULL,  -- Ground truth: 0=Normal, 1=Fraud
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: predictions
-- Stores ML model predictions
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    transaction_id INTEGER NOT NULL,
    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_fraud BOOLEAN NOT NULL,
    fraud_probability FLOAT NOT NULL,
    FOREIGN KEY (transaction_id) REFERENCES transactions(id) ON DELETE CASCADE
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions(created_at);
CREATE INDEX IF NOT EXISTS idx_transactions_class ON transactions(class);
CREATE INDEX IF NOT EXISTS idx_predictions_transaction_id ON predictions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_predictions_is_fraud ON predictions(is_fraud);
CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(prediction_time);

-- Show created tables
\dt

-- Confirmation
SELECT 'Database initialized successfully!' AS status;
