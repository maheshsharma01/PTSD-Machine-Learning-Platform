import sqlite3

def create_connection():
    conn = sqlite3.connect("users.db", check_same_thread=False)
    return conn

def create_table():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        name TEXT,
        password TEXT
    )
    """)

    conn.commit()
    conn.close()

def add_user(username, name, password):
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("INSERT INTO users VALUES (?, ?, ?)", (username, name, password))

    conn.commit()
    conn.close()

def get_all_users():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users")
    users = cursor.fetchall()

    conn.close()
    return users