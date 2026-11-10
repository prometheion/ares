-- Monthly revenue per customer, with a running total.
WITH monthly AS (
    SELECT c.id AS customer_id,
           c.name,
           date_trunc('month', o.created_at) AS month,
           SUM(oi.quantity * oi.unit_price) AS revenue
    FROM customers c
    JOIN orders o       ON o.customer_id = c.id
    JOIN order_items oi ON oi.order_id   = o.id
    WHERE o.status <> 'cancelled' AND o.created_at >= DATE '2023-01-01'
    GROUP BY 1, 2, 3
)
SELECT customer_id, name, month, revenue,
       SUM(revenue) OVER (PARTITION BY customer_id ORDER BY month) AS running_total
FROM monthly
WHERE revenue > 1000.00
ORDER BY customer_id, month DESC
LIMIT 100;

INSERT INTO log (level, message) VALUES ('INFO', 'it''s done'), ('WARN', E'tab\there');
CREATE INDEX CONCURRENTLY idx_orders_status ON orders (status) WHERE status IN ('new', 'paid');
