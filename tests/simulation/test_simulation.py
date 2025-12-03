"""
Tests for the simulation module.
"""

import pandas as pd

from demand_capacity_platform.simulation import CapacityModel, QueueSimulation
from demand_capacity_platform.simulation.capacity_model import CapacityMetrics


class TestCapacityModel:
    """Tests for CapacityModel class."""

    def test_init(self):
        """Test CapacityModel initialization."""
        model = CapacityModel(
            working_hours_per_day=8,
            working_days_per_week=5,
            productivity_factor=0.75
        )

        assert model.working_hours_per_day == 8
        assert model.working_days_per_week == 5
        assert model.productivity_factor == 0.75

    def test_calculate_weekly_capacity(self):
        """Test weekly capacity calculation."""
        model = CapacityModel(
            working_hours_per_day=7.5,
            working_days_per_week=5,
            productivity_factor=0.8
        )

        # 10 FTE * 7.5 hours * 5 days * 0.8 productivity = 300 hours
        # 300 hours * 60 mins / 60 min appointment = 300 appointments
        capacity = model.calculate_weekly_capacity(
            staff_fte=10.0,
            avg_appointment_mins=60
        )

        assert capacity == 300.0

    def test_calculate_fte_required(self):
        """Test FTE requirement calculation."""
        model = CapacityModel(
            working_hours_per_day=7.5,
            working_days_per_week=5,
            productivity_factor=0.8
        )

        fte = model.calculate_fte_required(
            weekly_demand=100,
            avg_appointment_mins=60,
            target_utilisation=0.85
        )

        assert fte > 0
        assert isinstance(fte, float)

    def test_estimate_wait_time_positive_throughput(self):
        """Test wait time estimation with positive net throughput."""
        model = CapacityModel()

        wait_time = model.estimate_wait_time(
            backlog=100,
            weekly_capacity=50,
            weekly_demand=40
        )

        # Net throughput = 50 - 40 = 10 per week
        # 100 / 10 = 10 weeks = 70 days
        assert wait_time == 70.0

    def test_estimate_wait_time_zero_throughput(self):
        """Test wait time estimation with zero net throughput."""
        model = CapacityModel()

        wait_time = model.estimate_wait_time(
            backlog=100,
            weekly_capacity=50,
            weekly_demand=50
        )

        # Queue is not shrinking, should return max (365 days)
        assert wait_time == 365.0

    def test_analyse_capacity(self, sample_referrals, sample_capacity):
        """Test capacity analysis for a service."""
        model = CapacityModel()

        metrics = model.analyse_capacity(
            sample_referrals,
            sample_capacity,
            "Adult Mental Health"
        )

        assert isinstance(metrics, CapacityMetrics)
        assert metrics.service == "Adult Mental Health"
        assert metrics.total_capacity_weekly > 0
        assert metrics.average_demand_weekly > 0
        assert 0 <= metrics.utilisation_rate <= 2  # Can be > 1 if over capacity

    def test_scenario_analysis(self, sample_referrals, sample_capacity):
        """Test scenario analysis."""
        model = CapacityModel()

        base_metrics = model.analyse_capacity(
            sample_referrals,
            sample_capacity,
            "Adult Mental Health"
        )

        # Increase demand by 20%
        scenario = model.scenario_analysis(
            base_metrics,
            demand_change_pct=20.0,
            capacity_change_pct=0.0
        )

        assert scenario.average_demand_weekly > base_metrics.average_demand_weekly
        assert scenario.utilisation_rate > base_metrics.utilisation_rate

    def test_generate_capacity_report(self, sample_referrals, sample_capacity):
        """Test capacity report generation."""
        model = CapacityModel()

        report = model.generate_capacity_report(
            sample_referrals,
            sample_capacity
        )

        assert isinstance(report, pd.DataFrame)
        assert len(report) > 0
        assert "service" in report.columns
        assert "utilisation_rate" in report.columns


class TestQueueSimulation:
    """Tests for QueueSimulation class."""

    def test_init(self):
        """Test QueueSimulation initialization."""
        sim = QueueSimulation(daily_capacity=20, seed=42)

        assert sim.daily_capacity == 20
        assert sim.seed == 42

    def test_run_basic(self):
        """Test basic simulation run."""
        sim = QueueSimulation(daily_capacity=20, seed=42)
        result = sim.run(days=100, daily_arrivals=18)

        assert result.simulation_days == 100
        assert result.average_queue_length >= 0
        assert result.average_wait_time >= 0
        assert result.throughput > 0
        assert 0 <= result.utilisation <= 1.5

    def test_run_with_initial_queue(self):
        """Test simulation with initial queue."""
        sim = QueueSimulation(daily_capacity=20, seed=42)
        result = sim.run(days=100, daily_arrivals=18, initial_queue=50)

        assert result.average_queue_length > 0

    def test_daily_stats_returned(self):
        """Test that daily statistics are returned."""
        sim = QueueSimulation(daily_capacity=20, seed=42)
        result = sim.run(days=30, daily_arrivals=15)

        assert result.daily_stats is not None
        assert len(result.daily_stats) == 30
        assert "queue_length" in result.daily_stats.columns
        assert "arrivals" in result.daily_stats.columns
        assert "served" in result.daily_stats.columns

    def test_run_scenarios(self):
        """Test multiple scenario simulation."""
        sim = QueueSimulation(daily_capacity=20, seed=42)

        scenarios = sim.run_scenarios(
            base_arrivals=15,
            base_capacity=20,
            days=100,
            demand_scenarios=[0.9, 1.0, 1.1],
            capacity_scenarios=[1.0]
        )

        assert isinstance(scenarios, pd.DataFrame)
        assert len(scenarios) == 3  # 3 demand scenarios * 1 capacity scenario
        assert "avg_wait_time" in scenarios.columns

    def test_monte_carlo(self):
        """Test Monte Carlo simulation."""
        sim = QueueSimulation(daily_capacity=20, seed=42)

        results = sim.monte_carlo(
            days=50,
            daily_arrivals=18,
            n_simulations=10
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 10
        assert "avg_wait_time" in results.columns

    def test_summarize_monte_carlo(self):
        """Test Monte Carlo summary statistics."""
        sim = QueueSimulation(daily_capacity=20, seed=42)

        results = sim.monte_carlo(
            days=50,
            daily_arrivals=18,
            n_simulations=20
        )

        summary = QueueSimulation.summarize_monte_carlo(results)

        assert "avg_wait_time" in summary
        assert "mean" in summary["avg_wait_time"]
        assert "std" in summary["avg_wait_time"]
        assert "ci_lower" in summary["avg_wait_time"]
        assert "ci_upper" in summary["avg_wait_time"]

    def test_reproducibility_with_seed(self):
        """Test that simulation is reproducible with same seed."""
        sim1 = QueueSimulation(daily_capacity=20, seed=42)
        sim2 = QueueSimulation(daily_capacity=20, seed=42)

        result1 = sim1.run(days=100, daily_arrivals=18)
        result2 = sim2.run(days=100, daily_arrivals=18)

        assert result1.average_queue_length == result2.average_queue_length
        assert result1.average_wait_time == result2.average_wait_time
