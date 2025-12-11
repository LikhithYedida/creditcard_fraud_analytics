-- 1) Fraud rate by hour of day
SELECT
    hour_of_day,
    COUNT(*) AS total_txn,
    SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraud_txn,
    ROUND(
        100.0 * SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) / COUNT(*),
        3
    ) AS fraud_rate_pct
FROM creditcard_transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- 2) Fraud rate by amount bucket
SELECT
    CASE
        WHEN amount < 10  THEN '0-10'
        WHEN amount < 50  THEN '10-50'
        WHEN amount < 100 THEN '50-100'
        WHEN amount < 200 THEN '100-200'
        WHEN amount < 500 THEN '200-500'
        ELSE '500+'
    END AS amount_bucket,
    COUNT(*) AS total_txn,
    SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraud_txn,
    ROUND(
        100.0 * SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) / COUNT(*),
        3
    ) AS fraud_rate_pct
FROM creditcard_transactions
GROUP BY amount_bucket
ORDER BY
    CASE amount_bucket
        WHEN '0-10' THEN 1
        WHEN '10-50' THEN 2
        WHEN '50-100' THEN 3
        WHEN '100-200' THEN 4
        WHEN '200-500' THEN 5
        ELSE 6
    END;
