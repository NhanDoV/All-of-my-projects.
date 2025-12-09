/* ---
        Reference: https://neetcode.io/problems/sql-create-table/question
--- */

CREATE TABLE videos (
  id INTEGER,
  name TEXT,
  created_at DATE,
  published BOOLEAN
);

-- Do not modify below this line --
SELECT table_name, column_name, data_type
FROM information_schema.columns
WHERE table_name = 'videos';

/* ----
        Expected output:
                +-------------+-------------------+-------------+
                | table_name  |	    column_name	  |  data_type  |
                +-------------+-------------------+-------------+
                | videos      |    id	          |   integer   |
                | videos      |    name	          |     text    |
                | videos      |    created_at	  |     date    |
                | videos      |    published	  |   boolean   |
                +-------------+-------------------+-------------+
--- */