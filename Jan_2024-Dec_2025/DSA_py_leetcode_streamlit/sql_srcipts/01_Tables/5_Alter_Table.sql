/* ---
        References: https://neetcode.io/problems/sql-alter-table/question
--- */

CREATE TABLE books (
  id INTEGER,
  title TEXT,
  author TEXT
);

-- Do not modify above this line --

ALTER TABLE books 
  ADD COLUMN published_year INTEGER;

ALTER TABLE books 
  RENAME COLUMN id TO isbn;

ALTER TABLE books 
  DROP COLUMN author;

-- Do not modify below this line --
SELECT column_name, data_type, column_default
FROM information_schema.columns
WHERE table_name = 'books'
ORDER BY column_name;

/* ---
      Expected output:
          *-----------------*-----------*----------------*
          |   column_name   | data_type | column_default |
          |-----------------|-----------|----------------|
          | isbn            | integer   | null           |
          | published_year  | integer   | null           |
          | title           | text      | null           |
          *-----------------*-----------*----------------*
--- */