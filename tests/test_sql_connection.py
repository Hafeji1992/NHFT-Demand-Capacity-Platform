import configparser
import pyodbc
import sys


def load_config(
    path="C:\\Users\\yhafeji\\Desktop\\Level 6 Data Science\\(ZDAT3001 UNUK) (FYR1 25-26) Work-Based Project (Assessment)\\NHFT-Demand-Capacity-Platform\\config.ini",
):
    """Load SQL Server connection details from config.ini."""
    config = configparser.ConfigParser()

    try:
        config.read(path)
        sql_config = config["sqlserver"]

        return {
            "server": sql_config["server"],
            "database": sql_config["database"],
            "driver": sql_config["driver"],
            "trusted": sql_config.get("trusted_connection", "Yes"),
        }
    except Exception as e:
        print(f"[ERROR] Unable to load config.ini: {e}")
        sys.exit(1)


def test_connection(config):
    """Attempt connection to SQL Server and perform a simple test query."""
    print("\n=== SQL CONNECTION TEST ===")
    print(f"Server:   {config['server']}")
    print(f"Database: {config['database']}")
    print(f"Driver:   {config['driver']}")
    print("Attempting connection...\n")

    try:
        conn_str = (
            f"DRIVER={{{config['driver']}}};"
            f"SERVER={config['server']};"
            f"DATABASE={config['database']};"
            f"Trusted_Connection={config['trusted']};"
        )

        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        print("[SUCCESS] Connected to SQL Server.")

        conn.close()
        print("\nConnection closed successfully.")

    except Exception as e:
        print("\n[ERROR] Connection failed:")
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    config = load_config()
    test_connection(config)
