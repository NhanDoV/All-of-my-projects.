/* ---
      Reference: https://neetcode.io/problems/sql-capitalization/question
--- */

CREATE TABLE users (
  name TEXT
);

-- Do not modify below this line --
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_name = 'users';

/* ----
        Expected output:
            +-------------+-------------------+-------------+
            | Table_name	|   column_name	    |  data_type  |
            +-------------+-------------------+-------------+
            |  users	    |    name	          |    text     |
            +-------------+-------------------+-------------+
-------- */