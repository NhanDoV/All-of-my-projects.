/* ---
        References: https://neetcode.io/problems/sql-default-values/question
--- */

CREATE TABLE videos (
  id INTEGER,
  name TEXT DEFAULT 'Untitled',
  is_published BOOLEAN DEFAULT false
);

-- Do not modify below this line --
INSERT INTO videos (id, name, is_published) 
VALUES (1, 'My Video', true),
       (2, 'Another Video', false);

INSERT INTO videos (id)
VALUES (3),
       (4);

INSERT INTO videos (name)
VALUES ('Video with no ID');

SELECT * FROM videos;

/* ---
        Expected output:
                *-------*-------------------*--------------*
                | id    |      name         | is_published |
                |-------|-------------------|--------------|
                | 1     | My Video          | true         |
                | 2     | Another Video     | false        |
                | 3     | Untitled          | false        |
                | 4     | Untitled          | false        |
                | null  | Video with no ID  | false        |        
                *-------*-------------------*--------------*
--- */