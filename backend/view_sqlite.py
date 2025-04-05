import sqlite3
import sys

def list_tables(conn):
    """List all tables in the SQLite database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return [table[0] for table in tables]

def view_table(conn, table_name):
    """Fetch and display all rows from the specified table."""
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT * FROM {table_name}")
    except sqlite3.OperationalError as e:
        print(f"Error accessing table {table_name}: {e}")
        return
    rows = cursor.fetchall()
    # Get column names
    columns = [description[0] for description in cursor.description]
    # Print header row
    print(" | ".join(columns))
    print("-" * 50)
    # Print each row in the table
    for row in rows:
        print(" | ".join(str(item) for item in row))

def main():
    if len(sys.argv) < 2:
        print("Usage: python view_sqlite.py <database_file>")
        sys.exit(1)

    db_file = sys.argv[1]
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

    print(f"Connected to {db_file}.")
    tables = list_tables(conn)
    if not tables:
        print("No tables found in the database.")
        sys.exit(0)

    print("Tables in the database:")
    for i, table in enumerate(tables):
        print(f"{i + 1}. {table}")

    try:
        table_choice = int(input("Enter the table number to view its contents: "))
        if table_choice < 1 or table_choice > len(tables):
            raise ValueError
    except ValueError:
        print("Invalid selection. Exiting.")
        sys.exit(1)

    table_name = tables[table_choice - 1]
    print(f"\nViewing contents of table: {table_name}\n")
    view_table(conn, table_name)
    conn.close()

if __name__ == "__main__":
    main()
