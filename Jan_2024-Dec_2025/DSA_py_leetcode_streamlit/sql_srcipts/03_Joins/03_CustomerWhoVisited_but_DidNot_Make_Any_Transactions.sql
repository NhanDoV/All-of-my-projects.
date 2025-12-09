/* ---
        Reference: https://leetcode.com/problems/customer-who-visited-but-did-not-make-any-transactions/description/
        Input: 
                Visits                                          Transactions
                    +----------+-------------+                      +----------------+----------+--------+
                    | visit_id | customer_id |                      | transaction_id | visit_id | amount |
                    +----------+-------------+                      +----------------+----------+--------+
                    | 1        | 23          |                      | 2              | 5        | 310    |
                    | 2        | 9           |                      | 3              | 5        | 300    |
                    | 4        | 30          |                      | 9              | 5        | 200    |
                    | 5        | 54          |                      | 12             | 1        | 910    |
                    | 6        | 96          |                      | 13             | 2        | 970    |
                    | 7        | 54          |                      +----------------+----------+--------+
                    | 8        | 54          |
                    +----------+-------------+ 
--- */

SELECT Visits.customer_id, COUNT(*) AS count_no_trans
FROM Visits
LEFT JOIN Transactions ON Visits.visit_id = Transactions.visit_id
WHERE Transactions.visit_id IS NULL
GROUP BY Visits.customer_id;

/* ---
        Expected Output: 
                    +-------------+----------------+
                    | customer_id | count_no_trans |
                    +-------------+----------------+
                    | 54          | 2              |
                    | 30          | 1              |
                    | 96          | 1              |
                    +-------------+----------------+
--- */