/* ---
        References: https://leetcode.com/problems/average-time-of-process-per-machine/
        Input:
            Activity table:
                +------------+------------+---------------+-----------+
                | machine_id | process_id | activity_type | timestamp |
                +------------+------------+---------------+-----------+
                | 0          | 0          | start         | 0.712     |
                | 0          | 0          | end           | 1.520     |
                | 0          | 1          | start         | 3.140     |
                | 0          | 1          | end           | 4.120     |
                | 1          | 0          | start         | 0.550     |
                | 1          | 0          | end           | 1.550     |
                | 1          | 1          | start         | 0.430     |
                | 1          | 1          | end           | 1.420     |
                | 2          | 0          | start         | 4.100     |
                | 2          | 0          | end           | 4.512     |
                | 2          | 1          | start         | 2.500     |
                | 2          | 1          | end           | 5.000     |
                +------------+------------+---------------+-----------+
--- */

SELECT 
    start.machine_id, 
    ROUND(AVG(end.timestamp - start.timestamp), 3) AS processing_time 
FROM Activity start

JOIN Activity end 
    ON end.machine_id = start.machine_id

WHERE start.activity_type = 'start' 
    AND end.activity_type='end'

GROUP BY start.machine_id

/* ---
        Expected Output:
                    +------------+-----------------+
                    | machine_id | processing_time |
                    +------------+-----------------+
                    | 0          | 0.894           |
                    | 1          | 0.995           |
                    | 2          | 1.456           |
                    +------------+-----------------+
--- */