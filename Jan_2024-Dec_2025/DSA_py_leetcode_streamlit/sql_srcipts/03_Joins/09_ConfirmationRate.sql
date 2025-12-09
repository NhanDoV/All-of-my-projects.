/* ---
        References: https://leetcode.com/problems/confirmation-rate/description/
        Inputs:
            Signups table:                                          Confirmations table:
                +---------+---------------------+                       +---------+---------------------+-----------+
                | user_id | time_stamp          |                       | user_id | time_stamp          | action    |
                +---------+---------------------+                       +---------+---------------------+-----------+
                | 3       | 2020-03-21 10:16:13 |                       | 3       | 2021-01-06 03:30:46 | timeout   |
                | 7       | 2020-01-04 13:57:59 |                       | 3       | 2021-07-14 14:00:00 | timeout   |
                | 2       | 2020-07-29 23:09:44 |                       | 7       | 2021-06-12 11:57:29 | confirmed |
                | 6       | 2020-12-09 10:39:37 |                       | 7       | 2021-06-13 12:58:28 | confirmed |
                +---------+---------------------+                       | 7       | 2021-06-14 13:59:27 | confirmed |
                                                                        | 2       | 2021-01-22 00:00:00 | confirmed |
                                                                        | 2       | 2021-02-28 23:59:59 | timeout   |
                                                                        +---------+---------------------+-----------+
--- */

SELECT s.user_id, 
       ROUND( AVG(                                          /*-- round the average value of --*/
                    IF( c.action = "confirmed", 1, 0)       /*-- all action which be confirmed (1 if True otherwise 0) --*/
                    ), 2) AS confirmation_rate
FROM Signups AS s 
LEFT JOIN Confirmations AS c 
    ON s.user_id = c.user_id 
GROUP BY user_id;

/* ---
    Expected Output:
        +---------+-------------------+
        | user_id | confirmation_rate |
        +---------+-------------------+
        | 6       | 0.00              |
        | 3       | 0.00              |
        | 7       | 1.00              |
        | 2       | 0.50              |
        +---------+-------------------+
--- */