/* ---
        References: https://leetcode.com/problems/find-customer-referee/description/
        Input: 
            Customer table:
                +----+------+------------+
                | id | name | referee_id |
                +----+------+------------+
                | 1  | Will | null       |
                | 2  | Jane | null       |
                | 3  | Alex | 2          |
                | 4  | Bill | null       |
                | 5  | Zack | 1          |
                | 6  | Mark | 2          |
                +----+------+------------+
--- */

# Write your MySQL query statement below
SELECT name
FROM Customer
WHERE (referee_id != 2) OR (referee_id IS NULL)

/* ---
        Expected Output:
                    +------+
                    | name |
                    +------+
                    | Will |
                    | Jane |
                    | Bill |
                    | Zack |
                    +------+
--- */