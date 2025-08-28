"""
Wind Farm GNN Analysis Pipeline
===============================

A comprehensive pipeline for wind farm layout generation, flow simulation,
and Graph Neural Network training for turbine power prediction.

Dependencies:
pip install numpy pandas matplotlib torch torch-geometric py_wake requests scipy networkx

For Global Wind Atlas data, you may need to register and get API access.
This implementation includes a fallback synthetic wind generator.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy import stats
# import requests
# import json
from typing import List, Dict
# import warnings

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv  # , global_mean_pool
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# PyWake (optional - fallback to simplified wake model if not available)
try:
    from py_wake import BastankhahGaussian
    from py_wake.site import UniformSite
    from py_wake.wind_turbines import OneTypeWindTurbines
    PYWAKE_AVAILABLE = True
except ImportError:
    print("PyWake not available. Using simplified wake model.")
    PYWAKE_AVAILABLE = False


class LayoutGenerator:
    """Generate regular grid layouts for wind farms."""

    def __init__(self, spacing_x: float = 800, spacing_y: float = 600):
        """
        Initialize layout generator.

        Args:
            spacing_x: Spacing between turbines in x direction (meters)
            spacing_y: Spacing between turbines in y direction (meters)
        """
        self.spacing_x = spacing_x
        self.spacing_y = spacing_y

    def generate_grid(self, n_turbines: int) -> pd.DataFrame:
        """
        Generate a regular grid layout.

        Args:
            n_turbines: Total number of turbines

        Returns:
            DataFrame with columns: turbine_id, x, y
        """
        # Calculate grid dimensions
        n_cols = int(np.ceil(np.sqrt(n_turbines)))
        n_rows = int(np.ceil(n_turbines / n_cols))

        turbines = []
        turbine_id = 0

        for row in range(n_rows):
            for col in range(n_cols):
                if turbine_id >= n_turbines:
                    break

                x = col * self.spacing_x
                y = row * self.spacing_y

                turbines.append({
                    'turbine_id': turbine_id,
                    'x': x,
                    'y': y
                })
                turbine_id += 1

        return pd.DataFrame(turbines)

    def visualize_layout(self, layout_df: pd.DataFrame):
        """Visualize the wind farm layout."""
        plt.figure(figsize=(10, 8))
        plt.scatter(layout_df['x'], layout_df['y'], s=100, alpha=0.7)

        for _, row in layout_df.iterrows():
            plt.annotate(
                f"T{int(row['turbine_id'])}",
                (row['x'], row['y']),
                xytext=(5, 5),
                textcoords='offset points'
            )

        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Wind Farm Layout')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()


class SyntheticWindSimulator:
    """Generate synthetic wind conditions based on location and wind statistics."""

    def __init__(self, lat: float, lon: float):
        """
        Initialize wind simulator for a specific location.

        Args:
            lat: Latitude
            lon: Longitude
        """
        self.lat = lat
        self.lon = lon
        self.wind_data = None

    def fetch_wind_atlas_data(self) -> Dict:
        """
        Attempt to fetch data from Global Wind Atlas.
        Falls back to synthetic data if API not available.
        """
        # Note: Global Wind Atlas API requires registration
        # This is a placeholder implementation
        print(f"Fetching wind data for coordinates: {self.lat}, {self.lon}")

        # Fallback to synthetic wind statistics (representative of good wind site)
        synthetic_data = {
            'mean_wind_speed': 8.5,  # m/s
            'weibull_A': 9.2,       # Scale parameter
            'weibull_k': 2.1,       # Shape parameter
            'sector_frequencies': np.ones(12) / 12,  # Equal probability for 12 sectors
            'sector_speeds': np.array([
                8.0, 8.5, 9.0, 9.2, 8.8, 8.3,
                7.8, 7.5, 7.9, 8.1, 8.4, 8.2]
            )  # Mean speeds per sector
        }

        self.wind_data = synthetic_data
        print("Using synthetic wind data (Global Wind Atlas API not configured)")
        return synthetic_data

    def generate_wind_timeseries(self, n_samples: int = 8760) -> pd.DataFrame:
        """
        Generate wind speed and direction time series.

        Args:
            n_samples: Number of time samples (default 8760 for 1 year hourly)

        Returns:
            DataFrame with columns: timestamp, wind_speed, wind_direction
        """
        if self.wind_data is None:
            self.fetch_wind_atlas_data()

        # Generate wind directions (12 sectors = 30° each)
        sector_probs = self.wind_data['sector_frequencies']
        sectors = np.random.choice(12, size=n_samples, p=sector_probs)
        wind_directions = sectors * 30 + np.random.uniform(-15, 15, n_samples)
        wind_directions = wind_directions % 360

        # Generate wind speeds using Weibull distribution per sector
        wind_speeds = np.zeros(n_samples)
        for i, sector in enumerate(sectors):
            # Adjust Weibull parameters slightly per sector
            sector_A = (
                self.wind_data['weibull_A']
                * (self.wind_data['sector_speeds'][sector] / 8.5)
            )
            wind_speeds[i] = np.random.weibull(self.wind_data['weibull_k']) * sector_A

        # Create timestamp index
        timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='H')

        return pd.DataFrame({
            'timestamp': timestamps,
            'wind_speed': np.clip(wind_speeds, 3, 25),  # Realistic wind speed range
            'wind_direction': wind_directions
        })


class FarmFlowSimulator:
    """Simulate wind farm flow and wake interactions."""

    def __init__(self, layout_df: pd.DataFrame):
        """
        Initialize farm flow simulator.

        Args:
            layout_df: DataFrame with turbine positions (turbine_id, x, y)
        """
        self.layout_df = layout_df
        self.turbine_positions = layout_df[['x', 'y']].values

        # Default turbine characteristics (representative of modern 2-3 MW turbine)
        self.rotor_diameter = 110  # meters
        self.hub_height = 80       # meters
        self.rated_power = 2000    # kW

    def simplified_wake_model(self, wind_speed: float, wind_direction: float) -> np.ndarray:
        """
        Simplified wake model when PyWake is not available.

        Args:
            wind_speed: Free-stream wind speed (m/s)
            wind_direction: Wind direction (degrees)

        Returns:
            Array of effective wind speeds at each turbine
        """
        n_turbines = len(self.turbine_positions)
        effective_speeds = np.full(n_turbines, wind_speed)

        # Convert wind direction to radians
        wind_dir_rad = np.radians(wind_direction)
        wind_vector = np.array([np.cos(wind_dir_rad), np.sin(wind_dir_rad)])

        for i, pos_i in enumerate(self.turbine_positions):
            for j, pos_j in enumerate(self.turbine_positions):
                if i == j:
                    continue

                # Vector from turbine j to turbine i
                distance_vector = pos_i - pos_j
                distance = np.linalg.norm(distance_vector)

                # Check if turbine i is downstream of turbine j
                downstream_distance = np.dot(distance_vector, wind_vector)

                if downstream_distance > 0 and distance < 8 * self.rotor_diameter:
                    # Simple wake deficit model
                    wake_deficit = 0.5 * (self.rotor_diameter / distance) ** 2
                    wake_deficit *= np.exp(
                        -0.5 * (downstream_distance / (3 * self.rotor_diameter)) ** 2)
                    effective_speeds[i] *= (1 - wake_deficit)

        return effective_speeds

    def power_curve(self, wind_speed: np.ndarray) -> np.ndarray:
        """
        Simple power curve model.

        Args:
            wind_speed: Array of wind speeds

        Returns:
            Array of power outputs (kW)
        """
        # Simplified power curve
        power = np.zeros_like(wind_speed)

        # Cut-in: 3 m/s, Rated: 12 m/s, Cut-out: 25 m/s
        mask1 = (wind_speed >= 3) & (wind_speed < 12)
        mask2 = (wind_speed >= 12) & (wind_speed <= 25)

        # Cubic relationship up to rated speed
        power[mask1] = self.rated_power * ((wind_speed[mask1] - 3) / 9) ** 3
        power[mask2] = self.rated_power

        return power

    def simulate_timestep(self, wind_speed: float, wind_direction: float) -> pd.DataFrame:
        """
        Simulate one timestep for all turbines.

        Args:
            wind_speed: Wind speed (m/s)
            wind_direction: Wind direction (degrees)

        Returns:
            DataFrame with turbine conditions
        """
        if PYWAKE_AVAILABLE:
            # Use PyWake if available (more accurate)
            try:
                site = UniformSite([1, 0, 0, 0], ti=0.1)
                windTurbines = OneTypeWindTurbines.from_tabular(
                    ws=np.arange(3, 26), power=self.power_curve(np.arange(3, 26)) * 1000
                )

                wf_model = BastankhahGaussian(site, windTurbines)
                x, y = self.turbine_positions[:, 0], self.turbine_positions[:, 1]

                sim_res = wf_model(x, y, wd=wind_direction, ws=wind_speed)
                effective_speeds = sim_res.WS_eff.values.flatten()
                powers = sim_res.P.values.flatten() / 1000  # Convert to kW

            except Exception as e:
                print(f"PyWake simulation failed: {e}. Using simplified model.")
                effective_speeds = self.simplified_wake_model(wind_speed, wind_direction)
                powers = self.power_curve(effective_speeds)
        else:
            # Use simplified model
            effective_speeds = self.simplified_wake_model(wind_speed, wind_direction)
            powers = self.power_curve(effective_speeds)

        # Create result DataFrame
        results = self.layout_df.copy()
        results['wind_speed_eff'] = effective_speeds
        results['wind_direction'] = wind_direction
        results['power'] = powers
        results['wind_speed_free'] = wind_speed

        return results

    def simulate_campaign(self, wind_timeseries: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate entire wind campaign.

        Args:
            wind_timeseries: DataFrame with timestamp, wind_speed, wind_direction

        Returns:
            DataFrame with all turbine results over time
        """
        all_results = []

        print(f"Simulating {len(wind_timeseries)} timesteps...")

        for idx, row in wind_timeseries.iterrows():
            if idx % 1000 == 0:
                print(f"Progress: {idx}/{len(wind_timeseries)}")

            timestep_result = self.simulate_timestep(row['wind_speed'], row['wind_direction'])
            timestep_result['timestamp'] = row['timestamp']
            all_results.append(timestep_result)

        return pd.concat(all_results, ignore_index=True)


class TurbineGNN(nn.Module):
    """Graph Neural Network for turbine power prediction."""

    def __init__(self, input_features: int, hidden_dim: int = 64):
        super(TurbineGNN, self).__init__()
        self.conv1 = GCNConv(input_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, 32)
        self.predictor = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        x, edge_index, _ = data.x, data.edge_index, data.batch

        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))

        # Final prediction
        x = self.predictor(x)
        return x


class GNNModelTrainer:
    """Train GNN model for turbine power prediction."""

    def __init__(self, layout_df: pd.DataFrame):
        self.layout_df = layout_df
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        self.model = None

    def create_graph_structure(self, distance_threshold: float = 1000) -> torch.Tensor:
        """
        Create graph edges based on distance threshold.

        Args:
            distance_threshold: Maximum distance for edge connection (meters)

        Returns:
            Edge index tensor for PyTorch Geometric
        """
        positions = self.layout_df[['x', 'y']].values
        n_turbines = len(positions)

        edges = []
        for i in range(n_turbines):
            for j in range(i + 1, n_turbines):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance <= distance_threshold:
                    edges.append([i, j])
                    edges.append([j, i])  # Undirected graph

        if len(edges) == 0:
            # Fallback: connect all nodes
            edges = [[i, j] for i in range(n_turbines) for j in range(n_turbines) if i != j]

        return torch.tensor(edges).t().contiguous()

    def prepare_data(self, simulation_results: pd.DataFrame, target_turbine_id: int):
        """
        Prepare data for GNN training.

        Args:
            simulation_results: Results from farm simulation
            target_turbine_id: ID of turbine to predict

        Returns:
            List of PyTorch Geometric Data objects
        """
        # Get unique timestamps
        timestamps = simulation_results['timestamp'].unique()

        # Create edge index (same for all timestamps)
        edge_index = self.create_graph_structure()

        data_list = []

        for timestamp in timestamps:
            # Get data for this timestamp
            ts_data = simulation_results[simulation_results['timestamp'] == timestamp].copy()
            ts_data = ts_data.sort_values('turbine_id').reset_index(drop=True)

            # Prepare features (exclude target turbine's power for features)
            features = []
            targets = []

            for turbine_id in ts_data['turbine_id']:
                row = ts_data[ts_data['turbine_id'] == turbine_id].iloc[0]

                # Features: position, wind conditions, but not power of target turbine
                feature_vector = [
                    row['x'] / 1000,  # Normalize position
                    row['y'] / 1000,
                    row['wind_speed_free'],
                    row['wind_direction'] / 360,  # Normalize direction
                    np.sin(np.radians(row['wind_direction'])),  # Cyclical encoding
                    np.cos(np.radians(row['wind_direction'])),
                ]

                # Add neighbor power information (if not target turbine)
                if turbine_id != target_turbine_id:
                    feature_vector.append(row['power'] / 2000)  # Normalize power
                else:
                    feature_vector.append(0)  # Unknown for target

                features.append(feature_vector)
                targets.append(row['power'])

            # Convert to tensors
            x = torch.tensor(features, dtype=torch.float)
            y = torch.tensor(targets, dtype=torch.float)

            # Create data object
            data = Data(x=x, edge_index=edge_index, y=y)
            data.target_turbine_id = target_turbine_id
            data_list.append(data)

        return data_list

    def train_model(
        self,
        simulation_results: pd.DataFrame,
        target_turbine_id: int,
        epochs: int = 200
    ) -> Dict:
        """
        Train GNN model.

        Args:
            simulation_results: Simulation results DataFrame
            target_turbine_id: ID of target turbine
            epochs: Number of training epochs

        Returns:
            Training history dictionary
        """
        # Prepare data
        print(f"Preparing data for target turbine {target_turbine_id}...")
        data_list = self.prepare_data(simulation_results, target_turbine_id)

        # Split data
        train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

        # Initialize model
        input_features = data_list[0].x.shape[1]
        self.model = TurbineGNN(input_features)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Training loop
        history = {'train_loss': [], 'val_loss': []}

        print(f"Training GNN model for {epochs} epochs...")

        for epoch in range(epochs):
            # Training
            self.model.train()
            total_train_loss = 0

            for data in train_data:
                optimizer.zero_grad()
                out = self.model(data)
                target_mask = torch.zeros(len(data.y), dtype=torch.bool)
                target_mask[target_turbine_id] = True

                loss = F.mse_loss(out[target_mask], data.y[target_mask].unsqueeze(-1))
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            # Validation
            self.model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for data in test_data:
                    out = self.model(data)
                    target_mask = torch.zeros(len(data.y), dtype=torch.bool)
                    target_mask[target_turbine_id] = True

                    loss = F.mse_loss(out[target_mask], data.y[target_mask].unsqueeze(-1))
                    total_val_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_data)
            avg_val_loss = total_val_loss / len(test_data)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)

            if epoch % 50 == 0:
                print(
                    f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, '
                    f'Val Loss = {avg_val_loss:.4f}'
                )

        # Final evaluation
        print("\nFinal evaluation...")
        self.evaluate_model(test_data, target_turbine_id)

        return history

    def evaluate_model(self, test_data: List, target_turbine_id: int):
        """Evaluate trained model."""
        if self.model is None:
            print("No model trained yet!")
            return

        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for data in test_data:
                out = self.model(data)
                target_mask = torch.zeros(len(data.y), dtype=torch.bool)
                target_mask[target_turbine_id] = True

                pred = out[target_mask].item()
                actual = data.y[target_mask].item()

                predictions.append(pred)
                actuals.append(actual)

        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)

        print(f"Test MSE: {mse:.2f}")
        print(f"Test R²: {r2:.4f}")
        print(f"RMSE: {np.sqrt(mse):.2f} kW")

        # Plot predictions vs actuals
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(actuals, predictions, alpha=0.6)
        plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
        plt.xlabel('Actual Power (kW)')
        plt.ylabel('Predicted Power (kW)')
        plt.title(f'GNN Predictions vs Actual\nTurbine {target_turbine_id}')

        plt.subplot(1, 2, 2)
        residuals = np.array(predictions) - np.array(actuals)
        plt.scatter(actuals, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Actual Power (kW)')
        plt.ylabel('Residuals (kW)')
        plt.title('Residuals Plot')

        plt.tight_layout()
        plt.show()


def main():
    """Main pipeline execution."""
    print("=== Wind Farm GNN Analysis Pipeline ===\n")

    # 1. Generate Layout
    print("1. Generating wind farm layout...")
    layout_gen = LayoutGenerator(spacing_x=600, spacing_y=500)
    layout = layout_gen.generate_grid(n_turbines=9)  # 3x3 grid
    print(f"Generated layout with {len(layout)} turbines")
    layout_gen.visualize_layout(layout)

    # 2. Generate Wind Data
    print("\n2. Generating wind conditions...")
    wind_sim = SyntheticWindSimulator(lat=55.0, lon=8.0)  # North Sea location
    wind_timeseries = wind_sim.generate_wind_timeseries(n_samples=2000)  # 2000 hours
    print(f"Generated {len(wind_timeseries)} wind conditions")

    # 3. Farm Flow Simulation
    print("\n3. Running farm flow simulation...")
    farm_sim = FarmFlowSimulator(layout)
    simulation_results = farm_sim.simulate_campaign(wind_timeseries)
    print(f"Simulation complete. Generated {len(simulation_results)} turbine-time records")

    # 4. Train GNN Model
    print("\n4. Training GNN model...")
    target_turbine = 4  # Center turbine in 3x3 grid
    trainer = GNNModelTrainer(layout)
    history = trainer.train_model(simulation_results, target_turbine, epochs=100)

    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('GNN Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\n=== Pipeline Complete ===")
    print(f"Successfully trained GNN model for turbine {target_turbine}")
    print("Model can now predict turbine power using neighbor information!")


if __name__ == "__main__":
    main()
