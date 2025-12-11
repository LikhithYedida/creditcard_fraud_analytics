-- 1) Row count
SELECT COUNT(*) AS row_count
FROM creditcard_transactions;

-- 2) Fraud distribution
SELECT class,
       COUNT(*) AS txn_count
FROM creditcard_transactions
GROUP BY class
ORDER BY class;

-- 3) Missing values in key fields
SELECT
    SUM(CASE WHEN amount   IS NULL THEN 1 ELSE 0 END) AS amount_nulls,
    SUM(CASE WHEN time_sec IS NULL THEN 1 ELSE 0 END) AS time_nulls,
    SUM(CASE WHEN class    IS NULL THEN 1 ELSE 0 END) AS class_nulls
FROM creditcard_transactions;

-- 4) Duplicate check (same time_sec + amount + class)
SELECT
    time_sec,
    amount,
    class,
    COUNT(*) AS dup_count
FROM creditcard_transactions
GROUP BY time_sec, amount, class
HAVING COUNT(*) > 1
ORDER BY dup_count DESC;
