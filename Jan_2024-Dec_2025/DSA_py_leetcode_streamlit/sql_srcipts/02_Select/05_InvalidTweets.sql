/* ---
        References: https://leetcode.com/problems/invalid-tweets/
        Input:
            Tweets table
                +----------+-----------------------------------+
                | tweet_id | content                           |
                +----------+-----------------------------------+
                | 1        | Let us Code                       |
                | 2        | More than fifteen chars are here! |
                +----------+-----------------------------------+
--- */

SELECT tweet_id FROM Tweets
WHERE LENGTH(content) > 15

/*---
        Expected Output: 
                +----------+
                | tweet_id |
                +----------+
                | 2        |
                +----------+
--- */