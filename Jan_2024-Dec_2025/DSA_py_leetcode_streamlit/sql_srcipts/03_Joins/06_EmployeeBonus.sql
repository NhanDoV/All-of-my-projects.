/* ---
        References: https://leetcode.com/problems/employee-bonus/
        Input :
                Employee table:                                             Bonus table:
                    +-------+--------+------------+--------+                    +-------+-------+
                    | empId | name   | supervisor | salary |                    | empId | bonus |
                    +-------+--------+------------+--------+                    +-------+-------+
                    | 3     | Brad   | null       | 4000   |                    | 2     | 500   |
                    | 1     | John   | 3          | 1000   |                    | 4     | 2000  |
                    | 2     | Dan    | 3          | 2000   |                    +-------+-------+
                    | 4     | Thomas | 3          | 4000   |
                    +-------+--------+------------+--------+        
--- */

SELECT e.name, b.bonus
FROM Employee AS e
LEFT JOIN Bonus AS b 
        ON e.empId = b.empId 
WHERE b.bonus < 1000                 # The employee has a bonus less than 1000.
    OR b.bonus IS NULL

/* ---
        Expected Output:
                    +------+-------+
                    | name | bonus |
                    +------+-------+
                    | Brad | null  |
                    | John | null  |
                    | Dan  | 500   |
                    +------+-------+
--- */