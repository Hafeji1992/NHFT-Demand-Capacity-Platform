import sys
import os

# Ensure project root is on the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data_engineering.connect import SQLServerConnection


def test_connection():
    """
    Test SQL Server connectivity using the reusable SQLServerConnection class.
    """

    print("\n=== SQL CONNECTION TEST ===")

    try:
        # Instantiate connection handler
        db = SQLServerConnection()

        # Attempt connection
        conn = db.connect()
        print("[SUCCESS] Connected to SQL Server.")

        # Run a simple metadata query to verify database access
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 5 name FROM sys.objects ORDER BY name;")
        rows = cursor.fetchall()

        print("\nTest query returned:")
        for row in rows:
            print(f" - {row[0]}")

        # Close connection
        db.close()
        print("\nConnection closed successfully.")

    except Exception as e:
        print("\n[ERROR] Connection test failed:")
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    test_connection()
