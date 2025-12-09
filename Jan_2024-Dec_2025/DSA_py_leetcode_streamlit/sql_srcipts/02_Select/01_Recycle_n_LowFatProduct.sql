/* ---
        References: https://leetcode.com/problems/recyclable-and-low-fat-products/description/

        Input: 
            Products table:
                    +-------------+----------+------------+
                    | product_id  | low_fats | recyclable |
                    +-------------+----------+------------+
                    | 0           | Y        | N          |
                    | 1           | Y        | Y          |
                    | 2           | N        | Y          |
                    | 3           | Y        | Y          |
                    | 4           | N        | N          |
                    +-------------+----------+------------+
--- */

# Write your MySQL query statement below
SELECT product_id
FROM Products
WHERE (low_fats = "Y") AND (recyclable = "Y")

/* ---
    Expected Output: 
            +-------------+
            | product_id  |
            +-------------+
            | 1           |
            | 3           |
            +-------------+
--- */