/* ---
        References: https://leetcode.com/problems/rising-temperature/
        Input:
                Weather table:
                +----+------------+-------------+
                | id | recordDate | temperature |
                +----+------------+-------------+
                | 1  | 2015-01-01 | 10          |
                | 2  | 2015-01-02 | 25          |
                | 3  | 2015-01-03 | 20          |
                | 4  | 2015-01-04 | 30          |
                +----+------------+-------------+
--- */

# Write your MySQL query statement below
SELECT w1.id 
FROM Weather AS w1, Weather AS w2 
WHERE DATEDIFF(w1.recordDate, w2.recordDate) = 1    # diff=1 meant compared to its previous dates (today-yesterday)
        AND w1.temperature > w2.temperature         # with higher temperatures
/* ---
        Expected Output:
                        +----+
                        | id |
                        +----+
                        | 2  |
                        | 4  |
                        +----+
--- */