/* ---
        References: https://leetcode.com/problems/students-and-examinations/
        Input
            Students table:                             Examinations table:
                    +------------+--------------+           +------------+--------------+
                    | student_id | student_name |           | student_id | subject_name |
                    +------------+--------------+           +------------+--------------+
                    | 1          | Alice        |           | 1          | Math         |
                    | 2          | Bob          |           | 1          | Physics      |
                    | 13         | John         |           | 1          | Programming  |
                    | 6          | Alex         |           | 2          | Programming  |
                    +------------+--------------+           | 1          | Physics      |
                                                            | 1          | Math         |
            Subjects table:                                 | 13         | Math         |
                    +--------------+                        | 13         | Programming  |
                    | subject_name |                        | 13         | Physics      |
                    +--------------+                        | 2          | Math         |
                    | Math         |                        | 1          | Math         |
                    | Physics      |                        +------------+--------------+
                    | Programming  |
                    +--------------+
--- */

SELECT st.student_id, st.student_name, su.subject_name, 
        COUNT(ex.subject_name) AS attended_exams
FROM Students AS st
CROSS JOIN Subjects AS su
LEFT JOIN Examinations AS ex ON st.student_id = ex.student_id
                                AND su.subject_name = ex.subject_name
GROUP BY st.student_id, su.subject_name
ORDER BY st.student_id, su.subject_name

/* ---
        Expected Output:
                +------------+--------------+--------------+----------------+
                | student_id | student_name | subject_name | attended_exams |
                +------------+--------------+--------------+----------------+
                | 1          | Alice        | Math         | 3              |
                | 1          | Alice        | Physics      | 2              |
                | 1          | Alice        | Programming  | 1              |
                | 2          | Bob          | Math         | 1              |
                | 2          | Bob          | Physics      | 0              |
                | 2          | Bob          | Programming  | 1              |
                | 6          | Alex         | Math         | 0              |
                | 6          | Alex         | Physics      | 0              |
                | 6          | Alex         | Programming  | 0              |
                | 13         | John         | Math         | 1              |
                | 13         | John         | Physics      | 1              |
                | 13         | John         | Programming  | 1              |
                +------------+--------------+--------------+----------------+        
--- */