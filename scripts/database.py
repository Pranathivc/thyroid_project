import sqlite3

# Connect to SQLite database
conn = sqlite3.connect("users.db")
cursor = conn.cursor()

# Create a Users table with security question support
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    security_question TEXT NOT NULL,
    security_answer TEXT NOT NULL
)
""")

conn.commit()
conn.close()

print("Database setup complete!")