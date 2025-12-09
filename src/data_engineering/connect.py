import os
import configparser
import pyodbc


class SQLServerConnection:
    """
    Reusable SQL Server connection class using Windows Authentication.
    Loads settings from config.ini at the project root.
    """

    def __init__(self, config_path=None):
        self.config = self.load_config(config_path)
        self.conn = None

    @staticmethod
    def load_config(config_path=None):
        """
        Locate and load config.ini from the project root.
        """
        config = configparser.ConfigParser()

        # If no custom path provided, automatically find project root
        if config_path is None:
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            config_path = os.path.join(project_root, "config.ini")

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"config.ini not found at expected location: {config_path}"
            )

        config.read(config_path)

        if "sqlserver" not in config:
            raise KeyError("config.ini is missing the [sqlserver] section")

        return config["sqlserver"]

    def connect(self):
        """
        Create a connection to SQL Server using pyodbc.
        Uses Trusted_Connection=Yes for Windows Authentication.
        """
        try:
            conn_str = (
                f"DRIVER={{{self.config['driver']}}};"
                f"SERVER={self.config['server']};"
                f"DATABASE={self.config['database']};"
                f"Trusted_Connection={self.config.get('trusted_connection', 'Yes')};"
            )

            self.conn = pyodbc.connect(conn_str)
            return self.conn

        except Exception as e:
            raise ConnectionError(f"Failed to connect to SQL Server: {e}")

    def query(self, sql, params=None):
        """
        Execute a SQL query and return the result as a list of rows.
        """
        if self.conn is None:
            self.connect()

        cursor = self.conn.cursor()

        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)

        try:
            rows = cursor.fetchall()
            return rows
        except pyodbc.ProgrammingError:
            # For commands like INSERT/UPDATE/DELETE
            return None

    def close(self):
        """
        Close the DB connection.
        """
        if self.conn:
            self.conn.close()
            self.conn = None
