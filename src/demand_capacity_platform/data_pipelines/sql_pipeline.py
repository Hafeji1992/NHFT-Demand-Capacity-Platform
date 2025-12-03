"""
SQL Data Pipeline Module.

Provides automated SQL data pipelines for extracting healthcare data
from various database sources and preparing it for analysis.
"""

import logging
from typing import Any, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class SQLDataPipeline:
    """
    Automated SQL data pipeline for healthcare data extraction.

    This class provides methods to connect to SQL databases,
    execute queries, and transform data for analysis.

    Attributes:
        connection_string: Database connection string.
        engine: SQLAlchemy engine instance.

    Example:
        >>> pipeline = SQLDataPipeline("sqlite:///data.db")
        >>> df = pipeline.execute_query("SELECT * FROM referrals")
    """

    def __init__(self, connection_string: str):
        """
        Initialize the SQL data pipeline.

        Args:
            connection_string: SQLAlchemy-compatible connection string.
        """
        self.connection_string = connection_string
        self._engine: Optional[Engine] = None
        logger.info("SQLDataPipeline initialized with connection: %s",
                    self._mask_connection_string(connection_string))

    @staticmethod
    def _mask_connection_string(conn_str: str) -> str:
        """Mask sensitive information in connection string for logging."""
        if "@" in conn_str:
            parts = conn_str.split("@")
            return f"***@{parts[-1]}"
        return conn_str

    @property
    def engine(self) -> Engine:
        """Get or create the database engine."""
        if self._engine is None:
            self._engine = create_engine(self.connection_string)
        return self._engine

    def execute_query(
        self,
        query: str,
        params: Optional[dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a DataFrame.

        Args:
            query: SQL query string.
            params: Optional dictionary of query parameters.

        Returns:
            DataFrame containing query results.

        Raises:
            SQLAlchemyError: If query execution fails.
        """
        logger.debug("Executing query: %s", query[:100])
        with self.engine.connect() as conn:
            result = pd.read_sql(text(query), conn, params=params or {})
        logger.info("Query returned %d rows", len(result))
        return result

    def extract_referrals(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        service: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract referral data from the database.

        Args:
            start_date: Start date filter (ISO format).
            end_date: End date filter (ISO format).
            service: Service name filter.

        Returns:
            DataFrame with referral data.
        """
        query = """
            SELECT
                referral_id,
                referral_date,
                service,
                priority,
                wait_days,
                status
            FROM referrals
            WHERE 1=1
        """
        params: dict[str, Any] = {}

        if start_date:
            query += " AND referral_date >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND referral_date <= :end_date"
            params["end_date"] = end_date
        if service:
            query += " AND service = :service"
            params["service"] = service

        return self.execute_query(query, params)

    def extract_activity(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        service: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract activity/appointments data from the database.

        Args:
            start_date: Start date filter (ISO format).
            end_date: End date filter (ISO format).
            service: Service name filter.

        Returns:
            DataFrame with activity data.
        """
        query = """
            SELECT
                activity_id,
                activity_date,
                service,
                activity_type,
                duration_mins,
                attended
            FROM activity
            WHERE 1=1
        """
        params: dict[str, Any] = {}

        if start_date:
            query += " AND activity_date >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND activity_date <= :end_date"
            params["end_date"] = end_date
        if service:
            query += " AND service = :service"
            params["service"] = service

        return self.execute_query(query, params)

    def extract_capacity(self, service: Optional[str] = None) -> pd.DataFrame:
        """
        Extract capacity data from the database.

        Args:
            service: Service name filter.

        Returns:
            DataFrame with capacity data.
        """
        query = """
            SELECT
                service,
                staff_fte,
                available_hours_weekly,
                avg_appointment_mins
            FROM capacity
            WHERE 1=1
        """
        params: dict[str, Any] = {}

        if service:
            query += " AND service = :service"
            params["service"] = service

        return self.execute_query(query, params)

    def aggregate_daily_demand(
        self,
        df: pd.DataFrame,
        date_column: str = "referral_date"
    ) -> pd.DataFrame:
        """
        Aggregate data to daily demand counts.

        Args:
            df: Input DataFrame with date column.
            date_column: Name of the date column.

        Returns:
            DataFrame with daily counts.
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        daily = df.groupby(df[date_column].dt.date).size().reset_index()
        daily.columns = ["date", "count"]
        daily["date"] = pd.to_datetime(daily["date"])
        return daily.sort_values("date").reset_index(drop=True)

    def close(self) -> None:
        """Close the database connection."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            logger.info("Database connection closed")

    def __enter__(self) -> "SQLDataPipeline":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
