/* ---
    References: https://leetcode.com/problems/managers-with-at-least-5-direct-reports/description/
        Input:
            Employee table:                                     Worst case:
                +-----+-------+------------+-----------+            SELECT e.name FROM Employee AS e
                | id  | name  | department | managerId |            JOIN (
                +-----+-------+------------+-----------+                SELECT id
                | 101 | John  | A          | null      |                FROM (
                | 102 | Dan   | A          | 101       |                    SELECT managerId AS id, 
                | 103 | James | A          | 101       |                            COUNT(*) AS n_report
                | 104 | Amy   | A          | 101       |                    FROM Employee
                | 104 | Amy   | A          | 101       |                    GROUP BY managerId
                | 105 | Anne  | A          | 101       |                    ) AS temp
                | 106 | Ron   | B          | 101       |                WHERE n_report >= 5
                +-----+-------+------------+-----------+            ) AS b ON e.id = b.id

--- */

SELECT e1.name FROM Employee e1     # Better Solution
JOIN Employee e2
ON e1.id = e2.managerID
GROUP BY e1.id, e1.name
HAVING count(e1.id) >= 5

/* ---
        Expected Output:
                    +------+
                    | name |
                    +------+
                    | John |
                    +------+
--- */