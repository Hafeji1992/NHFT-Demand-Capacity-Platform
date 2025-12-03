"""
Capacity Model Module.

Provides capacity modelling and analysis for healthcare services.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CapacityMetrics:
    """
    Container for capacity analysis metrics.

    Attributes:
        service: Service name.
        total_capacity_weekly: Total weekly capacity (appointments).
        average_demand_weekly: Average weekly demand.
        utilisation_rate: Capacity utilisation rate (0-1).
        excess_capacity: Positive if over-capacity, negative if under.
        wait_time_estimate: Estimated wait time in days.
        fte_required: FTE required to meet demand.
    """
    service: str
    total_capacity_weekly: float
    average_demand_weekly: float
    utilisation_rate: float
    excess_capacity: float
    wait_time_estimate: float
    fte_required: float


class CapacityModel:
    """
    Capacity modelling and analysis for healthcare services.

    Provides methods to analyse demand vs capacity, calculate
    utilisation rates, and estimate required resources.

    Attributes:
        working_hours_per_day: Working hours per day per staff.
        working_days_per_week: Working days per week.
        productivity_factor: Factor for non-patient time (0-1).

    Example:
        >>> model = CapacityModel()
        >>> metrics = model.analyse_capacity(demand_df, capacity_df, "Adult Mental Health")
        >>> print(f"Utilisation: {metrics.utilisation_rate:.1%}")
    """

    def __init__(
        self,
        working_hours_per_day: float = 7.5,
        working_days_per_week: int = 5,
        productivity_factor: float = 0.8
    ):
        """
        Initialize the capacity model.

        Args:
            working_hours_per_day: Working hours per day per staff.
            working_days_per_week: Working days per week.
            productivity_factor: Factor for non-patient time.
        """
        self.working_hours_per_day = working_hours_per_day
        self.working_days_per_week = working_days_per_week
        self.productivity_factor = productivity_factor
        logger.info(
            "CapacityModel initialized (hours/day=%s, days/week=%s, productivity=%s)",
            working_hours_per_day, working_days_per_week, productivity_factor
        )

    def calculate_weekly_capacity(
        self,
        staff_fte: float,
        avg_appointment_mins: float
    ) -> float:
        """
        Calculate weekly appointment capacity.

        Args:
            staff_fte: Full-time equivalent staff.
            avg_appointment_mins: Average appointment duration in minutes.

        Returns:
            Weekly appointment capacity.
        """
        available_hours = (
            staff_fte
            * self.working_hours_per_day
            * self.working_days_per_week
            * self.productivity_factor
        )

        capacity = (available_hours * 60) / avg_appointment_mins

        return capacity

    def calculate_fte_required(
        self,
        weekly_demand: float,
        avg_appointment_mins: float,
        target_utilisation: float = 0.85
    ) -> float:
        """
        Calculate FTE required to meet demand.

        Args:
            weekly_demand: Weekly demand (appointments).
            avg_appointment_mins: Average appointment duration in minutes.
            target_utilisation: Target utilisation rate.

        Returns:
            Required FTE.
        """
        required_hours = (weekly_demand * avg_appointment_mins) / 60

        available_hours_per_fte = (
            self.working_hours_per_day
            * self.working_days_per_week
            * self.productivity_factor
        )

        fte = required_hours / (available_hours_per_fte * target_utilisation)

        return fte

    def estimate_wait_time(
        self,
        backlog: float,
        weekly_capacity: float,
        weekly_demand: float
    ) -> float:
        """
        Estimate wait time given backlog and capacity.

        Args:
            backlog: Current number of patients waiting.
            weekly_capacity: Weekly appointment capacity.
            weekly_demand: Weekly demand (new referrals).

        Returns:
            Estimated wait time in days.
        """
        # Net throughput (capacity minus new demand)
        net_throughput = weekly_capacity - weekly_demand

        if net_throughput <= 0:
            # Queue is growing, return infinity (or large number)
            return 365.0  # Cap at 1 year

        weeks_to_clear = backlog / net_throughput
        wait_days = weeks_to_clear * 7

        return min(wait_days, 365.0)

    def analyse_capacity(
        self,
        demand_df: pd.DataFrame,
        capacity_df: pd.DataFrame,
        service: str,
        date_column: str = "referral_date"
    ) -> CapacityMetrics:
        """
        Analyse capacity for a specific service.

        Args:
            demand_df: DataFrame with demand data.
            capacity_df: DataFrame with capacity data.
            service: Service name to analyse.
            date_column: Name of the date column in demand_df.

        Returns:
            CapacityMetrics with analysis results.
        """
        # Filter data for service
        service_demand = demand_df[demand_df["service"] == service].copy()
        service_capacity = capacity_df[capacity_df["service"] == service].iloc[0]

        # Calculate weekly demand
        service_demand[date_column] = pd.to_datetime(service_demand[date_column])
        weekly_demand = (
            service_demand.groupby(pd.Grouper(key=date_column, freq="W"))
            .size()
            .mean()
        )

        # Calculate capacity
        weekly_capacity = self.calculate_weekly_capacity(
            service_capacity["staff_fte"],
            service_capacity["avg_appointment_mins"]
        )

        # Calculate metrics
        utilisation = weekly_demand / weekly_capacity if weekly_capacity > 0 else 1.0
        excess = weekly_capacity - weekly_demand

        # Estimate wait time (assuming current backlog equals 2 weeks of demand)
        estimated_backlog = weekly_demand * 2
        wait_time = self.estimate_wait_time(estimated_backlog, weekly_capacity, weekly_demand)

        # Calculate required FTE
        fte_required = self.calculate_fte_required(
            weekly_demand,
            service_capacity["avg_appointment_mins"]
        )

        return CapacityMetrics(
            service=service,
            total_capacity_weekly=weekly_capacity,
            average_demand_weekly=weekly_demand,
            utilisation_rate=utilisation,
            excess_capacity=excess,
            wait_time_estimate=wait_time,
            fte_required=fte_required
        )

    def scenario_analysis(
        self,
        base_metrics: CapacityMetrics,
        demand_change_pct: float = 0.0,
        capacity_change_pct: float = 0.0
    ) -> CapacityMetrics:
        """
        Perform what-if scenario analysis.

        Args:
            base_metrics: Base capacity metrics.
            demand_change_pct: Percentage change in demand.
            capacity_change_pct: Percentage change in capacity.

        Returns:
            CapacityMetrics with scenario results.
        """
        new_demand = base_metrics.average_demand_weekly * (1 + demand_change_pct / 100)
        new_capacity = base_metrics.total_capacity_weekly * (1 + capacity_change_pct / 100)

        utilisation = new_demand / new_capacity if new_capacity > 0 else 1.0
        excess = new_capacity - new_demand

        # Recalculate wait time
        estimated_backlog = new_demand * 2
        wait_time = self.estimate_wait_time(estimated_backlog, new_capacity, new_demand)

        # Calculate new FTE requirement based on demand change
        new_fte = base_metrics.fte_required * (1 + demand_change_pct / 100)

        return CapacityMetrics(
            service=base_metrics.service,
            total_capacity_weekly=new_capacity,
            average_demand_weekly=new_demand,
            utilisation_rate=utilisation,
            excess_capacity=excess,
            wait_time_estimate=wait_time,
            fte_required=new_fte
        )

    def generate_capacity_report(
        self,
        demand_df: pd.DataFrame,
        capacity_df: pd.DataFrame,
        services: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Generate a capacity report for multiple services.

        Args:
            demand_df: DataFrame with demand data.
            capacity_df: DataFrame with capacity data.
            services: List of services to analyse. If None, analyse all.

        Returns:
            DataFrame with capacity metrics for each service.
        """
        if services is None:
            services = capacity_df["service"].unique().tolist()

        results = []

        for service in services:
            try:
                metrics = self.analyse_capacity(demand_df, capacity_df, service)
                results.append({
                    "service": metrics.service,
                    "weekly_capacity": metrics.total_capacity_weekly,
                    "weekly_demand": metrics.average_demand_weekly,
                    "utilisation_rate": metrics.utilisation_rate,
                    "excess_capacity": metrics.excess_capacity,
                    "wait_time_days": metrics.wait_time_estimate,
                    "fte_required": metrics.fte_required
                })
            except Exception as e:
                logger.warning("Failed to analyse service %s: %s", service, str(e))

        return pd.DataFrame(results)
