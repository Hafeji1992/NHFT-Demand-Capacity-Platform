"""
Queue Simulation Module.

Provides discrete event simulation for healthcare queues.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """
    Container for queue simulation results.

    Attributes:
        service: Service name.
        simulation_days: Number of days simulated.
        average_queue_length: Average queue length over simulation.
        max_queue_length: Maximum queue length.
        average_wait_time: Average wait time in days.
        percentile_95_wait: 95th percentile wait time.
        throughput: Average daily throughput.
        utilisation: Average resource utilisation.
        daily_stats: DataFrame with daily statistics.
    """
    service: str
    simulation_days: int
    average_queue_length: float
    max_queue_length: int
    average_wait_time: float
    percentile_95_wait: float
    throughput: float
    utilisation: float
    daily_stats: Optional[pd.DataFrame] = None


class QueueSimulation:
    """
    Discrete event simulation for healthcare queues.

    Simulates patient flow through a queue with stochastic
    arrival and service rates.

    Attributes:
        daily_capacity: Daily appointment capacity.
        seed: Random seed for reproducibility.

    Example:
        >>> sim = QueueSimulation(daily_capacity=20)
        >>> result = sim.run(days=365, daily_arrivals=18)
        >>> print(f"Average wait: {result.average_wait_time:.1f} days")
    """

    def __init__(
        self,
        daily_capacity: int = 20,
        seed: Optional[int] = None
    ):
        """
        Initialize the queue simulation.

        Args:
            daily_capacity: Maximum daily appointments.
            seed: Random seed for reproducibility.
        """
        self.daily_capacity = daily_capacity
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        logger.info("QueueSimulation initialized with capacity=%d", daily_capacity)

    def run(
        self,
        days: int = 365,
        daily_arrivals: float = 15.0,
        initial_queue: int = 0,
        arrival_distribution: str = "poisson",
        capacity_variation: float = 0.1,
        service_name: str = "Service"
    ) -> SimulationResult:
        """
        Run the queue simulation.

        Args:
            days: Number of days to simulate.
            daily_arrivals: Mean daily arrivals.
            initial_queue: Initial queue length.
            arrival_distribution: Distribution for arrivals ("poisson" or "normal").
            capacity_variation: Coefficient of variation for capacity.
            service_name: Name of the service.

        Returns:
            SimulationResult with simulation outcomes.
        """
        logger.info(
            "Running simulation for %d days (arrivals=%s, capacity=%s)",
            days, daily_arrivals, self.daily_capacity
        )

        # Initialize tracking
        queue = initial_queue
        wait_times: list[float] = []
        daily_stats = []

        # Track patients in queue with their arrival day
        patients_in_queue: list[int] = [0] * initial_queue  # Arrival days

        for day in range(days):
            # Generate arrivals
            if arrival_distribution == "poisson":
                arrivals = self._rng.poisson(daily_arrivals)
            else:
                arrivals = max(0, int(self._rng.normal(daily_arrivals, daily_arrivals * 0.2)))

            # Add new patients to queue
            patients_in_queue.extend([day] * arrivals)

            # Generate daily capacity (with variation)
            if capacity_variation > 0:
                capacity = max(
                    1,
                    int(self._rng.normal(
                        self.daily_capacity,
                        self.daily_capacity * capacity_variation
                    ))
                )
            else:
                capacity = self.daily_capacity

            # Process patients (FIFO)
            served = min(len(patients_in_queue), capacity)

            for _ in range(served):
                if patients_in_queue:
                    arrival_day = patients_in_queue.pop(0)
                    wait_time = day - arrival_day
                    wait_times.append(wait_time)

            queue = len(patients_in_queue)

            # Track daily statistics
            daily_stats.append({
                "day": day,
                "arrivals": arrivals,
                "served": served,
                "capacity": capacity,
                "queue_length": queue,
                "utilisation": served / capacity if capacity > 0 else 0
            })

        # Calculate summary statistics
        daily_df = pd.DataFrame(daily_stats)

        result = SimulationResult(
            service=service_name,
            simulation_days=days,
            average_queue_length=daily_df["queue_length"].mean(),
            max_queue_length=int(daily_df["queue_length"].max()),
            average_wait_time=np.mean(wait_times) if wait_times else 0,
            percentile_95_wait=np.percentile(wait_times, 95) if wait_times else 0,
            throughput=daily_df["served"].mean(),
            utilisation=daily_df["utilisation"].mean(),
            daily_stats=daily_df
        )

        logger.info(
            "Simulation complete. Avg queue: %.1f, Avg wait: %.1f days",
            result.average_queue_length, result.average_wait_time
        )

        return result

    def run_scenarios(
        self,
        base_arrivals: float,
        base_capacity: int,
        days: int = 365,
        demand_scenarios: Optional[list[float]] = None,
        capacity_scenarios: Optional[list[float]] = None,
        service_name: str = "Service"
    ) -> pd.DataFrame:
        """
        Run multiple scenario simulations.

        Args:
            base_arrivals: Base daily arrival rate.
            base_capacity: Base daily capacity.
            days: Simulation days per scenario.
            demand_scenarios: List of demand multipliers (e.g., [0.9, 1.0, 1.1]).
            capacity_scenarios: List of capacity multipliers.
            service_name: Name of the service.

        Returns:
            DataFrame with scenario results.
        """
        if demand_scenarios is None:
            demand_scenarios = [0.8, 0.9, 1.0, 1.1, 1.2]
        if capacity_scenarios is None:
            capacity_scenarios = [0.8, 0.9, 1.0, 1.1, 1.2]

        results = []

        for demand_mult in demand_scenarios:
            for cap_mult in capacity_scenarios:
                arrivals = base_arrivals * demand_mult
                capacity = int(base_capacity * cap_mult)

                self.daily_capacity = capacity
                result = self.run(
                    days=days,
                    daily_arrivals=arrivals,
                    service_name=service_name
                )

                results.append({
                    "demand_multiplier": demand_mult,
                    "capacity_multiplier": cap_mult,
                    "daily_arrivals": arrivals,
                    "daily_capacity": capacity,
                    "avg_queue_length": result.average_queue_length,
                    "avg_wait_time": result.average_wait_time,
                    "wait_time_95th": result.percentile_95_wait,
                    "utilisation": result.utilisation
                })

        return pd.DataFrame(results)

    def monte_carlo(
        self,
        days: int = 365,
        daily_arrivals: float = 15.0,
        n_simulations: int = 100,
        service_name: str = "Service"
    ) -> pd.DataFrame:
        """
        Run Monte Carlo simulation to estimate uncertainty.

        Args:
            days: Simulation days per run.
            daily_arrivals: Mean daily arrivals.
            n_simulations: Number of simulation runs.
            service_name: Name of the service.

        Returns:
            DataFrame with results from each simulation run.
        """
        logger.info("Running Monte Carlo with %d simulations", n_simulations)

        results = []

        for i in range(n_simulations):
            # Use different seed for each run
            self._rng = np.random.default_rng(self.seed + i if self.seed else None)

            result = self.run(
                days=days,
                daily_arrivals=daily_arrivals,
                service_name=service_name
            )

            results.append({
                "run": i + 1,
                "avg_queue_length": result.average_queue_length,
                "max_queue_length": result.max_queue_length,
                "avg_wait_time": result.average_wait_time,
                "wait_time_95th": result.percentile_95_wait,
                "throughput": result.throughput,
                "utilisation": result.utilisation
            })

        results_df = pd.DataFrame(results)

        # Add summary statistics
        logger.info(
            "Monte Carlo complete. Wait time: %.1f ± %.1f days (95%% CI: %.1f - %.1f)",
            results_df["avg_wait_time"].mean(),
            results_df["avg_wait_time"].std(),
            results_df["avg_wait_time"].quantile(0.025),
            results_df["avg_wait_time"].quantile(0.975)
        )

        return results_df

    @staticmethod
    def summarize_monte_carlo(results_df: pd.DataFrame) -> dict:
        """
        Summarize Monte Carlo simulation results.

        Args:
            results_df: DataFrame from monte_carlo method.

        Returns:
            Dictionary with summary statistics.
        """
        metrics = ["avg_queue_length", "avg_wait_time", "utilisation"]
        summary = {}

        for metric in metrics:
            if metric in results_df.columns:
                summary[metric] = {
                    "mean": results_df[metric].mean(),
                    "std": results_df[metric].std(),
                    "ci_lower": results_df[metric].quantile(0.025),
                    "ci_upper": results_df[metric].quantile(0.975),
                    "min": results_df[metric].min(),
                    "max": results_df[metric].max()
                }

        return summary
