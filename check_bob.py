import sqlite3
import pathlib

db = pathlib.Path.home() / ".bob" / "db" / "bob.db"

con = sqlite3.connect(db)

rows = con.execute("""
    SELECT project_id, approval_config
    FROM tasks
    WHERE task_type = 'normal'
    ORDER BY updated_at DESC
""").fetchall()

for row in rows[:10]:
    print(row)