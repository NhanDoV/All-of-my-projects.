/* ---
        Reference: https://leetcode.com/problems/replace-employee-id-with-the-unique-identifier/description/
        Input: 
                Employees table:                        EmployeeUNI table:

                    +----+----------+                       +----+-----------+
                    | id | name     |                       | id | unique_id |
                    +----+----------+                       +----+-----------+
                    | 1  | Alice    |                       | 3  | 1         |
                    | 7  | Bob      |                       | 11 | 2         |
                    | 11 | Meir     |                       | 90 | 3         |
                    | 90 | Winston  |                       +----+-----------+
                    | 3  | Jonathan |
                    +----+----------+
--- */

SELECT EmployeeUNI.unique_id, Employees.name
FROM Employees
LEFT JOIN EmployeeUNI ON Employees.id = EmployeeUNI.id

/* ---
        Expected Output: 
                +-----------+----------+
                | unique_id | name     |
                +-----------+----------+
                | null      | Alice    |
                | null      | Bob      |
                | 2         | Meir     |
                | 3         | Winston  |
                | 1         | Jonathan |
                +-----------+----------+
--- */