# -*- coding: utf-8 -*-
# %% Python 3.10+
METHOD = 'Proposed'
FAMILY = 'Proposed'
USE_DTR = True
VARIANT = 'full'


SOURCES = {
    'main': r'''
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import glob
import time
import random
from math import atan2, degrees, radians, cos, sin, sqrt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler

torch._dynamo.config.suppress_errors = True


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

QUICK_TEST_MODE = False

DATASET_ROOT = "ais_dataset"
REFERENCE_DIR = os.path.join(DATASET_ROOT, "reference_trajectories")
TEST_DIR = os.path.join(DATASET_ROOT, "test_trajectories")
OUTPUT_ROOT = "prediction_results"
SAMPLE_PLOT_DIR = os.path.join(OUTPUT_ROOT, "sample_detail_plots")

PREDICTION_STEPS = 10
OBS_WINDOW_LENGTH = 20
SLIDING_STRIDE = 10
MAX_EVAL_SAMPLES = None
LIMIT_ONE_SAMPLE_PER_MMSI = True
sequence_length = 5
TIME_INTERVAL_MIN = 5
SKIP_SAMPLE_IDS = {}
SKIP_SOURCE_FILES = set()
SKIP_MMSI = set()
MAX_SPEED_TRAIN_SEC = 45
MAX_ANGLE_TRAIN_SEC = 120
ANGLE_EARLY_STOP_PATIENCE = 30

if QUICK_TEST_MODE:
    ANGLE_EPOCHS = 20
    SPEED_EPOCHS = 20
    N_TRAJECTORIES = 100
    N_MIXTURES = 5
    MAX_SIMILAR_TRAJ = 10
    MAX_EVAL_SAMPLES = 6
else:
    ANGLE_EPOCHS = 300
    SPEED_EPOCHS = 300
    N_TRAJECTORIES = 1000
    N_MIXTURES = 11
    MAX_SIMILAR_TRAJ = 20

DIRECTION_CHECK_LOOKBACK = 3
DIRECTION_CHECK_THRESHOLD = 20
DISTANCE_CHECK_THRESHOLD = 0.2
SPATIAL_HARD_THRESHOLD_KM = 50.0
SIMILAR_TOP_RATIO = 0.2

TURN_THRESHOLD_DEG = 15.0
SPEED_CV_THRESHOLD = 0.15

USE_SIMILARITY_RETRIEVAL = True

N_DETAILED_VIS = 6

EARTH_RADIUS_KM = 6371.0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(SAMPLE_PLOT_DIR, exist_ok=True)


class FeatureEngineer:
    def __init__(self):
        self.earth_radius = EARTH_RADIUS_KM

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return self.earth_radius * c

    def calculate_heading(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        x = sin(dlon) * cos(lat2)
        y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
        initial_bearing = atan2(x, y)
        return (degrees(initial_bearing) + 360) % 360

    def calculate_features(self, lats, lons, times=None):
        n_points = len(lats)
        parsed_times = pd.to_datetime(times) if times is not None else None
        speeds = np.zeros(n_points)
        accelerations = np.zeros(n_points)
        headings = np.zeros(n_points)
        turning_angles = np.zeros(n_points)
        curvatures = np.zeros(n_points)
        sin_headings = np.zeros(n_points)
        cos_headings = np.zeros(n_points)

        for i in range(1, n_points):
            distance = self.haversine_distance(lats[i - 1], lons[i - 1], lats[i], lons[i])
            if times is not None and i > 0:
                time_diff = (parsed_times[i] - parsed_times[i - 1]).total_seconds() / 3600.0
                speeds[i] = distance / time_diff if time_diff > 0 else (speeds[i - 1] if i > 1 else 0)
            else:
                speeds[i] = distance * 12
        if n_points > 1:
            speeds[0] = speeds[1]

        for i in range(1, n_points):
            headings[i] = self.calculate_heading(lats[i - 1], lons[i - 1], lats[i], lons[i])
            sin_headings[i] = sin(radians(headings[i]))
            cos_headings[i] = cos(radians(headings[i]))
        if n_points > 1:
            headings[0] = headings[1]
            sin_headings[0] = sin_headings[1]
            cos_headings[0] = cos_headings[1]

        for i in range(2, n_points):
            turning_angle = headings[i] - headings[i - 1]
            if turning_angle > 180:
                turning_angle -= 360
            elif turning_angle < -180:
                turning_angle += 360
            turning_angles[i] = turning_angle

        time_interval = TIME_INTERVAL_MIN / 60.0
        for i in range(2, n_points):
            accelerations[i] = (speeds[i] - speeds[i - 1]) / time_interval

        for i in range(2, n_points):
            if speeds[i] > 0.1:
                curvatures[i] = abs(radians(turning_angles[i])) / (speeds[i] * time_interval)
            else:
                curvatures[i] = 0

        features = np.column_stack([
            lons, lats, sin_headings, cos_headings, speeds, accelerations, curvatures
        ])
        return features, headings, turning_angles


fe_global = FeatureEngineer()


def haversine_km_vec(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class SimilarTrajectoryFilter:
    def __init__(self, feature_engineer):
        self.feature_engineer = feature_engineer
        self.historical_trajectories = {}
        self._loaded = False
        self._segment_cache = {}

    def load_historical_trajectories(self, data_dir, force_reload=False):
        if self._loaded and not force_reload:
            return
        files = sorted(glob.glob(os.path.join(data_dir, '*_merged_resampled_5min.csv')))
        self.historical_trajectories = {}
        for f in files:
            df = pd.read_csv(f)
            filename = os.path.basename(f)
            self.historical_trajectories[filename] = {
                'df': df,
                'lons': df['LONGITUDE'].values,
                'lats': df['LATITUDE'].values,
                'times': df['TIME'].values if 'TIME' in df.columns else None
            }
        self._loaded = True
        self._segment_cache = {}

    def extract_trajectory_segments(self, trajectory_length):
        if trajectory_length in self._segment_cache:
            return self._segment_cache[trajectory_length]

        segments = []
        for filename, traj_data in self.historical_trajectories.items():
            lons, lats, times = traj_data['lons'], traj_data['lats'], traj_data['times']
            total_points = len(lons)
            if total_points >= trajectory_length:
                for start_idx in range(0, total_points - trajectory_length + 1, 1):
                    end_idx = start_idx + trajectory_length
                    segments.append({
                        'filename': filename, 'start_idx': start_idx, 'end_idx': end_idx,
                        'lons': lons[start_idx:end_idx], 'lats': lats[start_idx:end_idx],
                        'times': times[start_idx:end_idx] if times is not None else None
                    })

        for seg in segments:
            feats, _, turns = self.feature_engineer.calculate_features(seg['lats'], seg['lons'], seg['times'])
            seg['features'] = feats
            seg['turning_angles'] = turns

        self._segment_cache[trajectory_length] = segments
        return segments

    def calculate_similarity_score(self, current_segment, historical_segment,
                                    current_features=None, current_turning_angles=None, weights=None):
        curr_lons, curr_lats = current_segment['lons'], current_segment['lats']
        hist_lons, hist_lats = historical_segment['lons'], historical_segment['lats']

        curr_pos = (curr_lats[-1], curr_lons[-1])
        hist_pos = (hist_lats[-1], hist_lons[-1])
        actual_spatial_dist = self.feature_engineer.haversine_distance(
            curr_pos[0], curr_pos[1], hist_pos[0], hist_pos[1])

        zero_scores = {k: 0.0 for k in ['angular_velocity', 'angular_direction', 'speed', 'curvature',
                                         'geometry', 'proximity']}
        if actual_spatial_dist > SPATIAL_HARD_THRESHOLD_KM:
            return -3.0, zero_scores

        dist_curr = self.feature_engineer.haversine_distance(curr_lats[0], curr_lons[0], curr_lats[-1], curr_lons[-1])
        dist_hist = self.feature_engineer.haversine_distance(hist_lats[0], hist_lons[0], hist_lats[-1], hist_lons[-1])
        rel_diff = abs(dist_curr - dist_hist) / dist_curr if dist_curr > 1e-6 else (0.0 if dist_hist < 0.1 else 1.0)
        if rel_diff > DISTANCE_CHECK_THRESHOLD:
            return -2.0, zero_scores

        if len(curr_lons) >= DIRECTION_CHECK_LOOKBACK and len(hist_lons) >= DIRECTION_CHECK_LOOKBACK:
            curr_dir = self.feature_engineer.calculate_heading(
                curr_lats[-DIRECTION_CHECK_LOOKBACK], curr_lons[-DIRECTION_CHECK_LOOKBACK], curr_lats[-1], curr_lons[-1])
            hist_dir = self.feature_engineer.calculate_heading(
                hist_lats[-DIRECTION_CHECK_LOOKBACK], hist_lons[-DIRECTION_CHECK_LOOKBACK], hist_lats[-1], hist_lons[-1])
            diff = abs(curr_dir - hist_dir)
            if diff > 180:
                diff = 360 - diff
            if diff > DIRECTION_CHECK_THRESHOLD:
                return -1.0, zero_scores

        if weights is None:
            weights = {'proximity': 0.3, 'geometry': 0.2, 'speed': 0.15,
                       'angular_velocity': 0.1, 'angular_direction': 0.1, 'curvature': 0.15}

        similarity_scores = {}
        decay_constant = 15.0
        similarity_scores['proximity'] = np.exp(-actual_spatial_dist / decay_constant)

        if current_features is None or current_turning_angles is None:
            current_features, _, current_turning_angles = self.feature_engineer.calculate_features(
                current_segment['lats'], current_segment['lons'], current_segment['times'])
        if 'features' in historical_segment:
            historical_features = historical_segment['features']
            historical_turning_angles = historical_segment['turning_angles']
        else:
            historical_features, _, historical_turning_angles = self.feature_engineer.calculate_features(
                historical_segment['lats'], historical_segment['lons'], historical_segment['times'])

        min_len = min(len(current_turning_angles), len(historical_turning_angles)) - 2
        if min_len > 0:
            c_ang_vel = np.abs(current_turning_angles[2:2 + min_len])
            h_ang_vel = np.abs(historical_turning_angles[2:2 + min_len])
            similarity_scores['angular_velocity'] = max(0, 1 - (np.abs(c_ang_vel - h_ang_vel).mean() / 180.0))

            c_ang_dir = np.sign(current_turning_angles[2:2 + min_len])
            h_ang_dir = np.sign(historical_turning_angles[2:2 + min_len])
            similarity_scores['angular_direction'] = (c_ang_dir == h_ang_dir).mean()

            c_speeds = current_features[2:2 + min_len, 4]
            h_speeds = historical_features[2:2 + min_len, 4]
            similarity_scores['speed'] = max(0, 1 - (np.abs(c_speeds - h_speeds).mean() / 50.0))

            c_curv = current_features[2:2 + min_len, 6]
            h_curv = historical_features[2:2 + min_len, 6]
            similarity_scores['curvature'] = max(0, 1 - (np.abs(c_curv - h_curv).mean() / 0.1))
        else:
            for k in ['angular_velocity', 'angular_direction', 'speed', 'curvature']:
                similarity_scores[k] = 0.5

        curr_centered_lons = curr_lons - np.mean(curr_lons)
        curr_centered_lats = curr_lats - np.mean(curr_lats)
        hist_centered_lons = hist_lons - np.mean(hist_lons)
        hist_centered_lats = hist_lats - np.mean(hist_lats)
        if len(curr_centered_lons) == len(hist_centered_lons):
            lon_corr = np.corrcoef(curr_centered_lons, hist_centered_lons)[0, 1]
            lat_corr = np.corrcoef(curr_centered_lats, hist_centered_lats)[0, 1]
            geometry_sim = (np.nan_to_num(lon_corr, nan=0.5) + np.nan_to_num(lat_corr, nan=0.5)) / 2
            similarity_scores['geometry'] = max(0, geometry_sim)
        else:
            similarity_scores['geometry'] = 0.5

        total_score = sum(similarity_scores.get(k, 0.5) * w for k, w in weights.items())
        return total_score, similarity_scores

    def find_similar_trajectories(self, current_trajectory, top_ratio=SIMILAR_TOP_RATIO, max_count=MAX_SIMILAR_TRAJ):
        current_length = len(current_trajectory['lons'])
        current_segment = {
            'lons': current_trajectory['lons'], 'lats': current_trajectory['lats'],
            'times': current_trajectory['times']
        }
        historical_segments = self.extract_trajectory_segments(current_length)

        current_features, _, current_turning_angles = self.feature_engineer.calculate_features(
            current_segment['lats'], current_segment['lons'], current_segment['times'])

        similarity_results = []
        for segment in historical_segments:
            total_score, detail_scores = self.calculate_similarity_score(
                current_segment, segment,
                current_features=current_features, current_turning_angles=current_turning_angles)
            similarity_results.append({'segment': segment, 'total_score': total_score, 'detail_scores': detail_scores})

        similarity_results.sort(key=lambda x: x['total_score'], reverse=True)
        top_count = max(1, int(len(similarity_results) * top_ratio))
        top_segments = similarity_results[:top_count]

        selected = {}
        for result in top_segments:
            filename = result['segment']['filename']
            if filename not in selected or result['total_score'] > selected[filename]['total_score']:
                selected[filename] = result

        final_results = [r for r in selected.values() if r['total_score'] > 0]
        final_results.sort(key=lambda x: x['total_score'], reverse=True)
        if len(final_results) > max_count:
            final_results = final_results[:max_count]

        return final_results


class AngleDataset(Dataset):
    def __init__(self, features, turning_angles, seq_len):
        min_len = min(len(features), len(turning_angles))
        self.features = features[:min_len]
        self.turning_angles = turning_angles[:min_len]
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.features) - self.seq_len)

    def __getitem__(self, idx):
        X_features = self.features[idx:idx + self.seq_len]
        X_angle = self.turning_angles[idx:idx + self.seq_len]
        y_angle = self.turning_angles[idx + self.seq_len]
        return (torch.FloatTensor(X_features), torch.FloatTensor(X_angle).unsqueeze(-1),
                torch.FloatTensor([y_angle]))


class SpeedDataset(Dataset):
    def __init__(self, speeds, seq_len):
        self.speeds = speeds
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.speeds) - self.seq_len)

    def __getitem__(self, idx):
        x = self.speeds[idx:idx + self.seq_len]
        y = self.speeds[idx + self.seq_len]
        return torch.FloatTensor(x).unsqueeze(-1), torch.FloatTensor([y])


class AttentionMechanism(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        attention_scores = self.attention_weights(lstm_output).squeeze(-1)
        max_scores = torch.max(attention_scores, dim=1, keepdim=True)[0]
        stable_scores = attention_scores - max_scores
        attention_weights = torch.softmax(stable_scores, dim=1)
        weighted_output = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        return weighted_output, attention_weights


class SpeedLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class AngleMDN(nn.Module):
    def __init__(self, feature_dim=7, angle_dim=1, hidden_dim=256, n_layers=3, n_mixtures=N_MIXTURES):
        super().__init__()
        self.n_mixtures = n_mixtures
        self.feature_lstm = nn.LSTM(feature_dim, hidden_dim, n_layers, batch_first=True, dropout=0)
        self.angle_lstm = nn.LSTM(angle_dim, hidden_dim // 2, n_layers, batch_first=True, dropout=0)
        self.feature_attention = AttentionMechanism(hidden_dim)
        self.angle_attention = AttentionMechanism(hidden_dim // 2)
        total_hidden = hidden_dim + hidden_dim // 2
        self.batch_norm = nn.BatchNorm1d(total_hidden)
        self.mdn_output = nn.Sequential(
            nn.Linear(total_hidden, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(1e-38),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(1e-38),
            nn.Linear(128, n_mixtures * 3)
        )

    def forward(self, x_features, x_angle):
        feature_out, _ = self.feature_lstm(x_features)
        feature_attended, _ = self.feature_attention(feature_out)
        angle_out, _ = self.angle_lstm(x_angle)
        angle_attended, _ = self.angle_attention(angle_out)
        combined = torch.cat([feature_attended, angle_attended], dim=1)
        if combined.size(0) > 1:
            combined = self.batch_norm(combined)
        return self.mdn_output(combined)


class TrajectoryModelCore:
    def __init__(self, reference_dir, n_mixtures=N_MIXTURES, seq_len=sequence_length,
                 prediction_steps=PREDICTION_STEPS):
        self.reference_dir = reference_dir
        self.n_mixtures = n_mixtures
        self.sequence_length = seq_len
        self.prediction_steps = prediction_steps
        self.angle_model = None
        self.speed_model = None
        self.device = device
        self.feature_engineer = fe_global
        self.feature_scaler = StandardScaler()
        self.angle_scaler = StandardScaler()
        self.speed_scaler = StandardScaler()
        self.similar_trajectory_filter = SimilarTrajectoryFilter(self.feature_engineer)
        self.pooled_features_scaled = None
        self.pooled_angles_scaled = None

    def filter_trajectory(self, lons, lats, sigma=1.0):
        return gaussian_filter1d(lons, sigma=sigma), gaussian_filter1d(lats, sigma=sigma)

    def correct_turning_angles(self, features, turning_angles):
        speeds = features[:, 4]
        corrected = turning_angles.copy()
        for i in range(2, len(speeds)):
            if speeds[i] < 2.5 or abs(turning_angles[i]) < 0.01:
                corrected[i] = 0.0
        return corrected

    def calculate_raw_speeds(self, lons, lats, times=None):
        n = len(lons)
        raw_speeds = np.zeros(n)
        for i in range(1, n):
            dist = self.feature_engineer.haversine_distance(lats[i - 1], lons[i - 1], lats[i], lons[i])
            if times is not None:
                try:
                    t_diff = (parsed_times[i] - parsed_times[i - 1]).total_seconds() / 3600.0
                except Exception:
                    t_diff = TIME_INTERVAL_MIN / 60.0
            else:
                t_diff = TIME_INTERVAL_MIN / 60.0
            raw_speeds[i] = dist / t_diff if t_diff > 0 else 0
        raw_speeds[0] = raw_speeds[1] if n > 1 else 0
        return raw_speeds

    def fit_scalers_from_reference(self):
        files = sorted(glob.glob(os.path.join(self.reference_dir, '*_merged_resampled_5min.csv')))
        if len(files) == 0:
            raise FileNotFoundError(f'No data files found in reference directory: {self.reference_dir}')

        all_features, all_turning_angles, all_speeds = [], [], []
        for f in files:
            df = pd.read_csv(f)
            lons, lats = df['LONGITUDE'].values, df['LATITUDE'].values
            times = df['TIME'].values if 'TIME' in df.columns else None
            filtered_lons, filtered_lats = self.filter_trajectory(lons, lats)
            features, headings, turning_angles = self.feature_engineer.calculate_features(
                filtered_lats, filtered_lons, times)
            turning_angles_corrected = self.correct_turning_angles(features, turning_angles)
            all_features.extend(features[2:])
            all_turning_angles.extend(turning_angles_corrected[2:])
            all_speeds.extend(features[2:, 4])

        all_features = np.array(all_features)
        all_turning_angles = np.array(all_turning_angles)
        all_speeds = np.array(all_speeds)

        self.feature_scaler.fit(all_features)
        self.angle_scaler.fit(all_turning_angles.reshape(-1, 1))
        self.speed_scaler.fit(all_speeds.reshape(-1, 1))

        self.pooled_features_scaled = self.feature_scaler.transform(all_features)
        self.pooled_angles_scaled = self.angle_scaler.transform(all_turning_angles.reshape(-1, 1)).flatten()

    def prepare_similar_trajectory_training_data(self, current_trajectory):
        self.similar_trajectory_filter.load_historical_trajectories(self.reference_dir)
        similar_trajectories = self.similar_trajectory_filter.find_similar_trajectories(
            current_trajectory, top_ratio=SIMILAR_TOP_RATIO, max_count=MAX_SIMILAR_TRAJ)

        all_features, all_turning_angles, all_speeds, all_headings = [], [], [], []
        for result in similar_trajectories:
            filename = result['segment']['filename']
            traj_data = self.similar_trajectory_filter.historical_trajectories[filename]
            segment_end_idx = result['segment']['end_idx']
            local_start_idx = max(0, segment_end_idx - self.sequence_length)
            local_end_idx = min(len(traj_data['lons']), segment_end_idx + self.prediction_steps)

            if local_end_idx - local_start_idx >= self.sequence_length:
                lons_segment = traj_data['lons'][local_start_idx:local_end_idx]
                lats_segment = traj_data['lats'][local_start_idx:local_end_idx]
                times_segment = (traj_data['times'][local_start_idx:local_end_idx]
                                 if traj_data['times'] is not None else None)

                filtered_lons, filtered_lats = self.filter_trajectory(lons_segment, lats_segment)
                features, headings, turning_angles = self.feature_engineer.calculate_features(
                    filtered_lats, filtered_lons, times_segment)
                turning_angles_corrected = self.correct_turning_angles(features, turning_angles)

                all_features.extend(features)
                all_headings.extend(headings)
                all_turning_angles.extend(turning_angles_corrected)
                all_speeds.extend(features[:, 4])

        if len(all_features) == 0:
            return None, None, None, None, similar_trajectories

        all_features = np.array(all_features)
        all_turning_angles = np.array(all_turning_angles)
        features_scaled = self.feature_scaler.transform(all_features)
        angles_scaled = self.angle_scaler.transform(all_turning_angles.reshape(-1, 1)).flatten()

        return features_scaled, angles_scaled, np.array(all_speeds), all_headings, similar_trajectories

    def train_speed_model(self, raw_speeds, epochs=SPEED_EPOCHS, lr=0.001, max_train_sec=MAX_SPEED_TRAIN_SEC):
        local_speed_scaler = StandardScaler()
        speeds_scaled = local_speed_scaler.fit_transform(raw_speeds.reshape(-1, 1)).flatten()
        dataset = SpeedDataset(speeds_scaled, self.sequence_length)
        if len(dataset) < 2:
            return None, local_speed_scaler
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        model = SpeedLSTM().to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()
        train_t0 = time.time()

        for epoch in range(epochs):
            if max_train_sec is not None and time.time() - train_t0 > max_train_sec:
                break

            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

        self.speed_model = model
        return model, local_speed_scaler

    def angle_mdn_loss(self, mdn_params, y_angle_true):
        n_mix = self.n_mixtures
        mix_probs = torch.softmax(mdn_params[:, :n_mix], dim=-1)
        means_angle = torch.clamp(mdn_params[:, n_mix:2 * n_mix], min=-1e2, max=1e2)
        stds_angle = torch.clamp(torch.nn.functional.softplus(mdn_params[:, 2 * n_mix:3 * n_mix]),
                                  min=1e-4, max=1e2)
        try:
            mixture_dist = dist.MixtureSameFamily(
                mixture_distribution=dist.Categorical(probs=mix_probs),
                component_distribution=dist.Normal(loc=means_angle, scale=stds_angle))
            log_probs = mixture_dist.log_prob(y_angle_true.squeeze())
            if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                valid_mask = ~(torch.isnan(log_probs) | torch.isinf(log_probs))
                angle_loss = -log_probs[valid_mask].mean() if valid_mask.any() else torch.tensor(10.0, device=mdn_params.device)
            else:
                angle_loss = -log_probs.mean()
        except Exception:
            angle_loss = torch.tensor(10.0, device=mdn_params.device)
        return angle_loss

    def train_model(self, features, angles, epochs=ANGLE_EPOCHS, batch_size=64,
                learning_rate=0.001, max_train_sec=MAX_ANGLE_TRAIN_SEC,
                early_stop_patience=ANGLE_EARLY_STOP_PATIENCE):

        angle_dataset = AngleDataset(features, angles, self.sequence_length)
        if len(angle_dataset) < 4:
            train_ds, val_ds = angle_dataset, angle_dataset
        else:
            train_size = int(0.8 * len(angle_dataset))
            val_size = len(angle_dataset) - train_size
            train_ds, val_ds = torch.utils.data.random_split(angle_dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=min(batch_size, max(1, len(train_ds))),
                                   shuffle=True, num_workers=0, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=min(batch_size, max(1, len(val_ds))),
                                 shuffle=False, num_workers=0)

        self.angle_model = AngleMDN(feature_dim=features.shape[1], n_mixtures=self.n_mixtures).to(self.device)
        optimizer = optim.Adam(self.angle_model.parameters(), lr=learning_rate * 0.1, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        train_losses, val_losses = [], []
        train_t0 = time.time()
        best_val = np.inf
        bad_epochs = 0

        for epoch in range(epochs):
            if max_train_sec is not None and time.time() - train_t0 > max_train_sec:
                break

            self.angle_model.train()
            train_loss, n_batches = 0.0, 0
            for x_feat, x_ang, y_angle in train_loader:
                x_feat, x_ang, y_angle = x_feat.to(self.device), x_ang.to(self.device), y_angle.to(self.device)
                if torch.isnan(x_feat).any() or torch.isnan(x_ang).any() or torch.isnan(y_angle).any():
                    continue
                optimizer.zero_grad()
                mdn_params = self.angle_model(x_feat, x_ang)
                if torch.isnan(mdn_params).any() or torch.isinf(mdn_params).any():
                    continue
                loss = self.angle_mdn_loss(mdn_params, y_angle)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.angle_model.parameters(), max_norm=0.5)
                bad_grad = any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                               for p in self.angle_model.parameters())
                if not bad_grad:
                    optimizer.step()
                    train_loss += loss.item()
                    n_batches += 1
            if n_batches == 0:
                continue

            self.angle_model.eval()
            val_loss, val_batches = 0.0, 0
            with torch.no_grad():
                for x_feat, x_ang, y_angle in val_loader:
                    x_feat, x_ang, y_angle = x_feat.to(self.device), x_ang.to(self.device), y_angle.to(self.device)
                    if torch.isnan(x_feat).any() or torch.isnan(x_ang).any():
                        continue
                    mdn_params = self.angle_model(x_feat, x_ang)
                    if torch.isnan(mdn_params).any() or torch.isinf(mdn_params).any():
                        continue
                    l = self.angle_mdn_loss(mdn_params, y_angle)
                    if torch.isnan(l) or torch.isinf(l):
                        continue
                    val_loss += l.item()
                    val_batches += 1
            if val_batches == 0:
                continue

            avg_train, avg_val = train_loss / n_batches, val_loss / val_batches
            train_losses.append(avg_train)
            val_losses.append(avg_val)
            scheduler.step(avg_val)

            if avg_val < best_val - 1e-4:
                best_val = avg_val
                bad_epochs = 0
            else:
                bad_epochs += 1

            if early_stop_patience is not None and bad_epochs >= early_stop_patience:
                break

        return train_losses, val_losses

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def predict_trajectories(self, initial_features, initial_angles, initial_speeds, initial_headings,
                              initial_lons, initial_lats, n_trajectories=N_TRAJECTORIES,
                              speed_model_components=None):
        self.angle_model.eval()
        speed_lstm, local_speed_scaler = (speed_model_components if speed_model_components else (None, None))
        if speed_lstm:
            speed_lstm.eval()

        raw_hist_speeds = self.speed_scaler.inverse_transform(initial_speeds.reshape(-1, 1)).flatten()
        avg_speed = np.mean(raw_hist_speeds)

        if local_speed_scaler is not None:
            current_speed_seq_scaled = local_speed_scaler.transform(raw_hist_speeds.reshape(-1, 1)).flatten()

        all_trajs_angle, all_trajs_speed, all_probs = [], [], []
        time_interval = TIME_INTERVAL_MIN / 60.0

        with torch.no_grad():
            for _ in range(n_trajectories):
                traj_angles, traj_speeds = [], []
                log_prob = 0.0
                feat_seq = initial_features.tolist()
                angle_seq = initial_angles.tolist()
                speed_seq_local = current_speed_seq_scaled.tolist() if local_speed_scaler is not None else None
                current_lons = initial_lons.tolist()
                current_lats = initial_lats.tolist()
                current_headings = initial_headings.tolist()

                for step in range(self.prediction_steps):
                    if speed_lstm is not None:
                        x_spd = np.array(speed_seq_local[-self.sequence_length:]).reshape(1, self.sequence_length, 1)
                        pred_spd_scaled = speed_lstm(torch.FloatTensor(x_spd).to(self.device)).item()
                        speed_original = max(0.1, local_speed_scaler.inverse_transform([[pred_spd_scaled]])[0, 0])
                        speed_seq_local.append(pred_spd_scaled)
                    else:
                        speed_original = avg_speed

                    x_feat = np.array(feat_seq[-self.sequence_length:]).reshape(1, self.sequence_length, -1)
                    x_ang = np.array(angle_seq[-self.sequence_length:]).reshape(1, self.sequence_length, 1)
                    mdn_params = self.angle_model(torch.FloatTensor(x_feat).to(self.device),
                                                   torch.FloatTensor(x_ang).to(self.device)).cpu().numpy()[0]

                    n_mix = self.n_mixtures
                    pis = self.softmax(mdn_params[:n_mix])
                    mus_angle = np.clip(mdn_params[n_mix:2 * n_mix], -1e2, 1e2)
                    raw_std = np.clip(mdn_params[2 * n_mix:3 * n_mix], -30, 30)
                    stds_angle = np.clip(np.log1p(np.exp(raw_std)), 1e-4, 1e2)

                    k = np.random.choice(n_mix, p=pis)
                    angle_sample = np.random.normal(mus_angle[k], stds_angle[k])
                    angle_sample = np.clip(angle_sample, -1e2, 1e2)
                    angle_original = self.angle_scaler.inverse_transform([[angle_sample]])[0, 0]
                    log_prob += np.log(pis[k] + 1e-10)

                    traj_angles.append(angle_original)
                    traj_speeds.append(speed_original)

                    current_heading_rad = radians(current_headings[-1])
                    distance = speed_original * time_interval
                    delta_lat = (distance * cos(current_heading_rad)) / 111.0
                    delta_lon = (distance * sin(current_heading_rad)) / (111.0 * cos(radians(current_lats[-1])))
                    new_lon = current_lons[-1] + delta_lon
                    new_lat = current_lats[-1] + delta_lat
                    current_lons.append(new_lon)
                    current_lats.append(new_lat)
                    new_heading = (current_headings[-1] + angle_original) % 360
                    current_headings.append(new_heading)

                    prev_speed = traj_speeds[-2] if len(traj_speeds) >= 2 else raw_hist_speeds[-1]
                    new_acc = (speed_original - prev_speed) * 12
                    new_curvature = abs(radians(angle_original)) / (speed_original * time_interval)
                    new_feat = np.array([new_lon, new_lat, sin(radians(new_heading)), cos(radians(new_heading)),
                                         speed_original, new_acc, new_curvature])
                    new_feat_scaled = self.feature_scaler.transform([new_feat])[0]
                    feat_seq.append(new_feat_scaled)
                    angle_seq.append(angle_sample)

                all_trajs_angle.append(traj_angles)
                all_trajs_speed.append(traj_speeds)
                all_probs.append(np.exp(log_prob))

        return all_trajs_angle, all_trajs_speed, all_probs

    def reconstruct_trajectory(self, initial_lons, initial_lats, initial_heading, angle_predictions,
                                speed_predictions, time_interval=TIME_INTERVAL_MIN / 60.0):
        last_lon, last_lat = initial_lons[-1], initial_lats[-1]
        current_heading = radians(initial_heading)
        pred_lons, pred_lats = [last_lon], [last_lat]
        for angle, speed in zip(angle_predictions, speed_predictions):
            current_heading += radians(angle)
            distance = speed * time_interval
            delta_lat = (distance * cos(current_heading)) / 111.0
            delta_lon = (distance * sin(current_heading)) / (111.0 * cos(radians(last_lat)))
            last_lon, last_lat = last_lon + delta_lon, last_lat + delta_lat
            pred_lons.append(last_lon)
            pred_lats.append(last_lat)
        return pred_lons, pred_lats


def parse_mmsi_from_filename(filename):
    suffix = '_merged_resampled_5min.csv'
    base = filename[:-len(suffix)] if filename.endswith(suffix) else filename
    parts = base.split('_')
    if len(parts) >= 3:
        return parts[-2]
    return f"unknown_{filename}"


def build_evaluation_samples(test_dir, obs_len=OBS_WINDOW_LENGTH, pred_len=PREDICTION_STEPS,
                              stride=SLIDING_STRIDE, one_per_mmsi=LIMIT_ONE_SAMPLE_PER_MMSI):
    files = sorted(glob.glob(os.path.join(test_dir, '*_merged_resampled_5min.csv')))
    if len(files) == 0:
        raise FileNotFoundError(f'No data files found in test directory: {test_dir}')

    samples = []
    sample_id = 0
    window = obs_len + pred_len
    for f in files:
        df = pd.read_csv(f)
        lons, lats = df['LONGITUDE'].values, df['LATITUDE'].values
        times = df['TIME'].values if 'TIME' in df.columns else None
        n = len(lons)
        if n < window:
            continue
        mmsi = parse_mmsi_from_filename(os.path.basename(f))
        for s in range(0, n - window + 1, stride):
            samples.append({
                'sample_id': sample_id, 'source_file': os.path.basename(f),
                'mmsi': mmsi,
                'window_start': s,
                'obs_lons': lons[s:s + obs_len], 'obs_lats': lats[s:s + obs_len],
                'obs_times': times[s:s + obs_len] if times is not None else None,
                'true_future_lons': lons[s + obs_len:s + window],
                'true_future_lats': lats[s + obs_len:s + window],
                'true_future_times': times[s + obs_len:s + window] if times is not None else None,
            })
            sample_id += 1

    if one_per_mmsi:
        rng = np.random.RandomState(42)
        by_mmsi = {}
        for s in samples:
            by_mmsi.setdefault(s['mmsi'], []).append(s)
        dedup = [group[rng.randint(len(group))] for group in by_mmsi.values()]
        dedup.sort(key=lambda x: x['sample_id'])
        for new_id, s in enumerate(dedup):
            s['sample_id'] = new_id
        samples = dedup

    if MAX_EVAL_SAMPLES is not None and len(samples) > MAX_EVAL_SAMPLES:
        rng2 = np.random.RandomState(42)
        idx = rng2.choice(len(samples), size=MAX_EVAL_SAMPLES, replace=False)
        idx.sort()
        samples = [samples[i] for i in idx]

    return samples


def compute_displacement_errors(pred_lons, pred_lats, true_lons, true_lats):
    p_lons, p_lats = np.array(pred_lons[1:]), np.array(pred_lats[1:])
    t_lons, t_lats = np.array(true_lons), np.array(true_lats)
    n = min(len(p_lons), len(t_lons))
    dists = haversine_km_vec(p_lats[:n], p_lons[:n], t_lats[:n], t_lons[:n])
    return dists.mean(), dists[-1], dists


def compute_ensemble_metrics(all_pred_coords, true_lons, true_lats):
    all_ade, all_fde = [], []
    for pred_lons, pred_lats in all_pred_coords:
        ade, fde, _ = compute_displacement_errors(pred_lons, pred_lats, true_lons, true_lats)
        all_ade.append(ade)
        all_fde.append(fde)
    all_ade, all_fde = np.array(all_ade), np.array(all_fde)
    return {
        'min_ade': all_ade.min(), 'min_fde': all_fde.min(),
        'mean_ade': all_ade.mean(), 'mean_fde': all_fde.mean(),
        'all_ade': all_ade, 'all_fde': all_fde,
    }


def compute_mpqr(all_pred_coords, true_lons, true_lats, confidence=0.95):
    N = len(all_pred_coords)
    T = len(true_lons)
    percentile = confidence * 100
    lons_mat = np.array([p[0][1:1 + T] for p in all_pred_coords])
    lats_mat = np.array([p[1][1:1 + T] for p in all_pred_coords])

    widths = np.zeros(T)
    for t in range(T):
        lons_t, lats_t = lons_mat[:, t], lats_mat[:, t]
        c_lon, c_lat = lons_t.mean(), lats_t.mean()
        d = haversine_km_vec(lats_t, lons_t, np.full(N, c_lat), np.full(N, c_lon))
        widths[t] = np.percentile(d, percentile)
    return widths.mean(), widths


def compute_heading_speed_error(pred_lons, pred_lats, true_lons, true_lats):
    anchor_lon, anchor_lat = pred_lons[0], pred_lats[0]

    def headings_and_speeds(lons, lats):
        full_lons = np.concatenate([[anchor_lon], lons])
        full_lats = np.concatenate([[anchor_lat], lats])
        heads, spds = [], []
        for i in range(1, len(full_lons)):
            heads.append(fe_global.calculate_heading(full_lats[i - 1], full_lons[i - 1], full_lats[i], full_lons[i]))
            d = fe_global.haversine_distance(full_lats[i - 1], full_lons[i - 1], full_lats[i], full_lons[i])
            spds.append(d / (TIME_INTERVAL_MIN / 60.0))
        return np.array(heads), np.array(spds)

    p_heads, p_speeds = headings_and_speeds(np.array(pred_lons[1:]), np.array(pred_lats[1:]))
    t_heads, t_speeds = headings_and_speeds(np.array(true_lons), np.array(true_lats))
    n = min(len(p_heads), len(t_heads))
    heading_diff = np.abs(((p_heads[:n] - t_heads[:n]) + 180) % 360 - 180)
    return heading_diff.mean(), np.abs(p_speeds[:n] - t_speeds[:n]).mean()


def classify_sample(true_lons, true_lats, anchor_lon, anchor_lat):
    full_lons = np.concatenate([[anchor_lon], true_lons])
    full_lats = np.concatenate([[anchor_lat], true_lats])
    features, headings, turning_angles = fe_global.calculate_features(full_lats, full_lons, None)
    cum_turn = float(np.sum(np.abs(turning_angles[2:]))) if len(turning_angles) > 2 else 0.0
    speeds = features[2:, 4] if features.shape[0] > 2 else features[:, 4]
    speed_cv = float(speeds.std() / max(speeds.mean(), 1e-6))
    return {
        'cum_turn_deg': cum_turn, 'speed_cv': speed_cv,
        'turn_category': 'turning' if cum_turn >= TURN_THRESHOLD_DEG else 'straight',
        'speed_category': 'varying_speed' if speed_cv >= SPEED_CV_THRESHOLD else 'constant_speed',
    }


def evaluate_sample(predictor, sample):
    t0 = time.time()
    obs_lons_raw, obs_lats_raw, obs_times = sample['obs_lons'], sample['obs_lats'], sample['obs_times']

    filtered_lons, filtered_lats = predictor.filter_trajectory(obs_lons_raw, obs_lats_raw)
    raw_speeds = predictor.calculate_raw_speeds(obs_lons_raw, obs_lats_raw, obs_times)

    features, headings, turning_angles = predictor.feature_engineer.calculate_features(
        filtered_lats, filtered_lons, obs_times)
    turning_angles_corrected = predictor.correct_turning_angles(features, turning_angles)

    features_scaled = predictor.feature_scaler.transform(features)
    angles_scaled = predictor.angle_scaler.transform(turning_angles_corrected.reshape(-1, 1)).flatten()
    speeds_scaled = predictor.speed_scaler.transform(features[:, 4].reshape(-1, 1)).flatten()

    v_max, v_min, v_mean = raw_speeds.max(), raw_speeds.min(), raw_speeds.mean()
    v_range = v_max - v_min
    rel_error = v_range / v_mean if v_mean > 1e-5 else 0.0
    speed_stable = (rel_error <= 0.10) or (v_range <= 5.0)
    predictor.recent_speed_mean = float(np.mean(raw_speeds[-5:]))
    predictor.stable_rollout_speed = predictor.recent_speed_mean if speed_stable else None
    predictor.adaptive_weight = prepare(predictor, speed_stable, raw_speeds, obs_lons_raw, obs_lats_raw, obs_times, SPEED_EPOCHS, OUTPUT_ROOT, sample["mmsi"])
    if speed_stable:
        speed_model_components = (None, None)
    else:
        speed_model_components = predictor.train_speed_model(raw_speeds, epochs=SPEED_EPOCHS)

    current_trajectory = {'lons': filtered_lons, 'lats': filtered_lats, 'times': obs_times}
    similar_trajectories = []

    if USE_SIMILARITY_RETRIEVAL:
        feat_sim, ang_sim, spd_sim, head_sim, similar_trajectories = predictor.prepare_similar_trajectory_training_data(
            current_trajectory)
        if feat_sim is None:
            feat_sim, ang_sim = predictor.pooled_features_scaled, predictor.pooled_angles_scaled
        n_similar = len(similar_trajectories)
    else:
        feat_sim, ang_sim = predictor.pooled_features_scaled, predictor.pooled_angles_scaled
        n_similar = -1

    train_losses, val_losses = predictor.train_model(feat_sim, ang_sim, epochs=ANGLE_EPOCHS, batch_size=64)

    seq_len = predictor.sequence_length
    start_idx = max(0, len(filtered_lons) - seq_len)
    init_feat = features_scaled[start_idx:]
    init_ang = angles_scaled[start_idx:]
    init_spd = speeds_scaled[start_idx:]
    init_head = headings[start_idx:]
    init_lon = filtered_lons[start_idx:]
    init_lat = filtered_lats[start_idx:]

    predictor.prediction_steps = PREDICTION_STEPS
    t_train_end = time.time()

    all_trajs_angle, all_trajs_speed, all_probs = predictor.predict_trajectories(
        init_feat, init_ang, init_spd, init_head, init_lon, init_lat,
        n_trajectories=N_TRAJECTORIES, speed_model_components=speed_model_components)

    all_pred_coords = [predictor.reconstruct_trajectory(init_lon, init_lat, init_head[-1], a, s)
                       for a, s in zip(all_trajs_angle, all_trajs_speed)]
    t_infer_end = time.time()

    top_idx = np.argsort(all_probs)[-5:][::-1]
    top5_coords = [all_pred_coords[i] for i in top_idx]
    top5_probs = [all_probs[i] for i in top_idx]

    true_lons, true_lats = sample['true_future_lons'], sample['true_future_lats']

    ens_metrics = compute_ensemble_metrics(all_pred_coords, true_lons, true_lats)
    top1_ade, top1_fde, _ = compute_displacement_errors(top5_coords[0][0], top5_coords[0][1], true_lons, true_lats)
    mpqr95, _ = compute_mpqr(all_pred_coords, true_lons, true_lats, confidence=0.95)
    heading_mae, speed_mae = compute_heading_speed_error(top5_coords[0][0], top5_coords[0][1], true_lons, true_lats)
    cls_info = classify_sample(true_lons, true_lats, init_lon[-1], init_lat[-1])

    result_row = {
        'sample_id': sample['sample_id'], 'source_file': sample['source_file'],
        'mmsi': sample['mmsi'],
        'window_start': sample['window_start'], 'n_similar_trajectories': n_similar,
        'ADE_top1_km': top1_ade, 'FDE_top1_km': top1_fde,
        'minADE_km': ens_metrics['min_ade'], 'minFDE_km': ens_metrics['min_fde'],
        'meanADE_km': ens_metrics['mean_ade'], 'meanFDE_km': ens_metrics['mean_fde'],
        'MPQR95_km': mpqr95,
        'heading_MAE_deg': heading_mae, 'speed_MAE_kmh': speed_mae,
        'cum_turn_deg': cls_info['cum_turn_deg'], 'speed_cv': cls_info['speed_cv'],
        'turn_category': cls_info['turn_category'], 'speed_category': cls_info['speed_category'],
        'train_time_sec': t_train_end - t0, 'infer_time_sec': t_infer_end - t_train_end,
    }

    extra = {
        'init_lon': init_lon, 'init_lat': init_lat, 'all_pred_coords': all_pred_coords,
        'top5_coords': top5_coords, 'top5_probs': top5_probs,
        'true_lons': true_lons, 'true_lats': true_lats,
        'obs_lons': obs_lons_raw, 'obs_lats': obs_lats_raw,
        'current_trajectory': current_trajectory, 'similar_trajectories': similar_trajectories,
        'train_losses': train_losses, 'val_losses': val_losses,
    }
    return result_row, extra


from pathlib import Path
import json
from time import perf_counter

def estimate(core, raw_speeds, lons, lats, times, epochs):
    n = len(raw_speeds)
    horizon = core.prediction_steps
    seq_length = core.sequence_length


    origins = np.unique(np.linspace(max(seq_length + 2, n // 2), n - 4, 3, dtype=int))
    comparison = []
    devices = list(range(torch.cuda.device_count())) if core.device.type == 'cuda' else []
    prior_speed_model = core.speed_model
    try:
        with torch.random.fork_rng(devices=devices):
            for origin in origins:
                origin = int(origin)
                if origin < seq_length + 2 or origin >= n:
                    continue
                torch.manual_seed(1907 + origin)
                historical = raw_speeds[:origin]
                net, scaler = core.train_speed_model(historical, epochs=epochs)
                if net is None:
                    continue
                net.eval()
                prefix_lon, prefix_lat = core.filter_trajectory(lons[:origin], lats[:origin])
                features, _, _ = core.feature_engineer.calculate_features(prefix_lat, prefix_lon, times[:origin])
                seq = scaler.transform(features[:, 4:5]).ravel().tolist()
                recent_mean = float(historical[-5:].mean())
                with torch.inference_mode():
                    for h, target in enumerate(raw_speeds[origin:origin+horizon]):
                        x = torch.as_tensor(seq[-seq_length:], dtype=torch.float32, device=core.device)
                        z = net(x.reshape(1, -1, 1)).item()
                        seq.append(z)
                        v = max(.1, scaler.inverse_transform([[z]])[0, 0])
                        comparison.append((float(target-v), recent_mean-v, h/max(1,horizon-1)))
    finally:
        core.speed_model = prior_speed_model
    weight, core.speed_constant = choose_distance_schedule(comparison)
    return weight, len(comparison)


def prepare(core, stable, speeds, lons, lats, times, epochs, output, vessel):
    start = perf_counter()
    core.speed_constant = False
    if stable:
        weight, n = 0., 0
    else:
        weight, n = estimate(core, speeds, lons, lats, times, epochs)
    folder = Path(output) / 'speed_weights'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f'{vessel}.json'
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(stable=bool(stable), weight=weight, constant_schedule=core.speed_constant, pseudo_forecasts=n,
                                   weight_fit_s=perf_counter()-start)), encoding='utf-8')
    tmp.replace(target)
    return weight


def choose_distance_schedule(rows):
    if not rows:
        return 0., False
    error, delta, ramp = np.asarray(rows).T
    groups = np.split(np.arange(len(rows)), np.flatnonzero(ramp[1:] == 0) + 1)
    cumulative_error = np.concatenate([np.cumsum(error[g]) for g in groups])
    results = []
    for constant in (False, True):
        adjustment = delta if constant else delta * ramp
        direction = np.concatenate([np.cumsum(adjustment[g]) for g in groups])
        norm = direction @ direction
        weight = float(np.clip(direction @ cumulative_error / norm, 0, 1)) if norm else 0.
        loss = np.sum((cumulative_error - weight * direction)**2)
        results.append((loss, weight, constant))
    _, weight, constant = min(results, key=lambda row: row[0])
    return weight, constant
''',
    'cv': r'''
import os
import time
from math import radians, cos
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from main import (
    build_evaluation_samples, TEST_DIR, OUTPUT_ROOT,
    PREDICTION_STEPS, haversine_km_vec,
    compute_displacement_errors, compute_heading_speed_error, classify_sample,
)
RESULT_CSV = os.path.join(OUTPUT_ROOT, "baseline_cv_kf_metrics.csv")
SUMMARY_TXT = os.path.join(OUTPUT_ROOT, "baseline_cv_kf_summary.txt")
ACTIVITY_THRESHOLD_KM = 1.0
MATCH_PROPOSED_METHOD_SAMPLES = True
PROPOSED_METRICS_CSV = os.path.join(OUTPUT_ROOT, "per_sample_metrics.csv")
Q_SCALE = 1e-5
MEAS_NOISE_STD_KM = 0.2
INIT_VEL_VARIANCE = 1.0
def filter_observation(obs_lons, obs_lats, sigma=1.0):
    return gaussian_filter1d(obs_lons, sigma=sigma), gaussian_filter1d(obs_lats, sigma=sigma)
def kf_predict(x, P, F, Q):
    x = F @ x
    P = F @ P @ F.T + Q
    return x, P
def kf_update(x, P, z, H, R):
    y = z - H @ x
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (np.eye(len(x)) - K @ H) @ P
    return x, P
def run_cv_kf(obs_lons, obs_lats, n_future_steps, q_scale=Q_SCALE, meas_noise_km=MEAS_NOISE_STD_KM):
    anchor_lon, anchor_lat = obs_lons[-1], obs_lats[-1]
    coslat0 = cos(radians(anchor_lat))
    def to_xy(lon, lat):
        return (lon - anchor_lon) * 111.0 * coslat0, (lat - anchor_lat) * 111.0
    def to_lonlat(x, y):
        return anchor_lon + x / (111.0 * coslat0), anchor_lat + y / 111.0
    xy = np.array([to_xy(lo, la) for lo, la in zip(obs_lons, obs_lats)], dtype=float)
    dt = 1.0
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)
    Q = q_scale * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                            [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                            [dt ** 3 / 2, 0, dt ** 2, 0],
                            [0, dt ** 3 / 2, 0, dt ** 2]], dtype=float)
    R = (meas_noise_km ** 2) * np.eye(2)
    x0, y0 = xy[0]
    if len(xy) >= 2:
        vx0, vy0 = xy[1] - xy[0]
    else:
        vx0, vy0 = 0.0, 0.0
    state = np.array([x0, y0, vx0, vy0], dtype=float)
    P = np.diag([meas_noise_km ** 2, meas_noise_km ** 2, INIT_VEL_VARIANCE, INIT_VEL_VARIANCE])
    for t in range(1, len(xy)):
        state, P = kf_predict(state, P, F, Q)
        state, P = kf_update(state, P, xy[t], H, R)
    pred_lons, pred_lats = [anchor_lon], [anchor_lat]
    pos_cov_list = []
    for _ in range(n_future_steps):
        state, P = kf_predict(state, P, F, Q)
        lon, lat = to_lonlat(state[0], state[1])
        pred_lons.append(lon)
        pred_lats.append(lat)
        pos_cov_list.append(P[:2, :2].copy())
    return pred_lons, pred_lats, pos_cov_list
def analytic_mpqr(pos_cov_list, confidence=0.95):
    radii = []
    for cov in pos_cov_list:
        sigma_eff2 = 0.5 * np.trace(cov)
        r = float(np.sqrt(max(-2.0 * sigma_eff2 * np.log(1 - confidence), 0.0)))
        radii.append(r)
    return float(np.mean(radii))
def net_displacement_km(lons, lats):
    return haversine_km_vec(lats[0], lons[0], lats[-1], lons[-1])
def print_and_log_summary(df, lines_out):
    def W(s=''):
        lines_out.append(s)
        print(s)
    def block(sub, label):
        W(f"\n[{label}] Number of samples: {len(sub)}")
        for col, metric_label, unit in [('ADE_km', 'ADE', 'km'),
                                        ('FDE_km', 'FDE', 'km'),
                                        ('heading_MAE_deg', 'Heading MAE', 'degrees'),
                                        ('speed_MAE_kmh', 'Speed MAE', 'km/h'),
                                        ('MPQR95_km', 'MPQR@95 (Analytical Approximation)', 'km')]:
            v = sub[col]
            W(f"  {metric_label}: Mean={v.mean():.3f} {unit}  Median={v.median():.3f} {unit}  Standard deviation={v.std():.3f}")
        W(f"  Average inference time per sample: {sub['infer_time_sec'].mean() * 1000:.3f} ms")
    W("=" * 78)
    W("Baseline: Constant-Velocity Kalman Filter (CV-KF) - Summary Results")
    W("=" * 78)
    W(f"Kalman filter hyperparameters: Q_SCALE={Q_SCALE}, MEAS_NOISE_STD_KM={MEAS_NOISE_STD_KM}, INIT_VEL_VARIANCE={INIT_VEL_VARIANCE} (fixed default values without per-sample tuning)")
    block(df, "All Samples")
    block(df[df['activity_category'] == 'Underway'], "Underway Subset")
    W("=" * 78)
def main():
    print("=" * 78)
    print("Baseline: Constant-Velocity Kalman Filter (CV-KF)")
    print("=" * 78)
    samples = build_evaluation_samples(TEST_DIR)
    if MATCH_PROPOSED_METHOD_SAMPLES and os.path.exists(PROPOSED_METRICS_CSV):
        proposed_df = pd.read_csv(PROPOSED_METRICS_CSV, encoding='utf-8-sig')
        keep_ids = set(proposed_df['sample_id'].tolist())
        samples = [s for s in samples if s['sample_id'] in keep_ids]
        print(f"[Alignment] Detected {PROPOSED_METRICS_CSV}. Evaluating only the {len(samples)} samples listed in that file to ensure comparison with the proposed model uses exactly the same test instances.")
    else:
        print(f"[Notice] Sample alignment is disabled or {PROPOSED_METRICS_CSV} was not found. Evaluating all {len(samples)} samples.")
    rows = []
    for sample in samples:
        t0 = time.time()
        obs_lons_raw, obs_lats_raw = sample['obs_lons'], sample['obs_lats']
        true_lons, true_lats = sample['true_future_lons'], sample['true_future_lats']
        filtered_lons, filtered_lats = filter_observation(obs_lons_raw, obs_lats_raw)
        pred_lons, pred_lats, pos_cov_list = run_cv_kf(filtered_lons, filtered_lats, PREDICTION_STEPS)
        ade, fde, _ = compute_displacement_errors(pred_lons, pred_lats, true_lons, true_lats)
        heading_mae, speed_mae = compute_heading_speed_error(pred_lons, pred_lats, true_lons, true_lats)
        mpqr95 = analytic_mpqr(pos_cov_list, confidence=0.95)
        cls_info = classify_sample(true_lons, true_lats, filtered_lons[-1], filtered_lats[-1])
        net_disp = net_displacement_km(true_lons, true_lats)
        rows.append({
            'sample_id': sample['sample_id'],
            'source_file': sample['source_file'],
            'mmsi': sample['mmsi'],
            'ADE_km': ade,
            'FDE_km': fde,
            'heading_MAE_deg': heading_mae,
            'speed_MAE_kmh': speed_mae,
            'MPQR95_km': mpqr95,
            'net_displacement_km': net_disp,
            'activity_category': 'Underway' if net_disp >= ACTIVITY_THRESHOLD_KM else 'Near-stationary',
            'turn_category': cls_info['turn_category'],
            'speed_category': cls_info['speed_category'],
            'infer_time_sec': time.time() - t0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n[Saved] {RESULT_CSV} ({len(df)} records)")
    lines = []
    print_and_log_summary(df, lines)
    with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n[Saved] {SUMMARY_TXT}")
''',
    'batch': r'''
"""Particle-batched implementation of the released angle and speed rollout."""
import numpy as np
import torch
def batched(self,initial_features,initial_angles,initial_speeds,initial_headings,initial_lons,initial_lats,n_trajectories,speed_model_components=None):
    n=n_trajectories
    ft=np.repeat(initial_features[None,:,:],n,axis=0)
    ang=np.tile(initial_angles.reshape(1,-1,1),(n,1,1))
    headings=np.full(n,initial_headings[-1]);lon=np.full(n,initial_lons[-1]);lat=np.full(n,initial_lats[-1])
    vhist=self.speed_scaler.inverse_transform(initial_speeds.reshape(-1,1)).ravel()
    avg=vhist.mean()
    if self.stable_rollout_speed is not None:
        avg=max(.1,self.stable_rollout_speed)
    speed,scaler=speed_model_components or (None,None)
    if speed is not None:
        speed.eval()
        seq=scaler.transform(vhist.reshape(-1,1)).ravel().tolist()
    self.angle_model.eval()
    angles=[];speeds=[];logp=np.zeros(n)
    tensor=lambda a: torch.as_tensor(a,dtype=torch.float32,device=self.device)
    with torch.inference_mode():
        for step in range(self.prediction_steps):
            v=avg
            if speed is not None:
                speed_z=speed(tensor(seq[-self.sequence_length:]).reshape(1,-1,1)).item()
                seq.append(speed_z)
                v=max(.1,scaler.inverse_transform([[speed_z]])[0,0])
                ramp = 1. if self.speed_constant else step / max(1, self.prediction_steps - 1)
                weight = self.adaptive_weight * ramp
                v = max(.1, (1-weight)*v + weight*self.recent_speed_mean)


            model=self.angle_model
            feature_out,_=model.feature_lstm(tensor(ft))
            angle_out,_=model.angle_lstm(tensor(ang))
            if hasattr(model,'feature_attention'):
                joint=torch.cat([model.feature_attention(feature_out)[0],model.angle_attention(angle_out)[0]],dim=1)
            else:
                joint=torch.cat([feature_out[:,-1],angle_out[:,-1]],dim=1)
            params=model.mdn_output(joint).cpu().numpy()
            if n>1 and step==1 and not getattr(self,'verified_batch',False):
                individual=np.concatenate([model(tensor(ft[j:j+1]),tensor(ang[j:j+1])).cpu().numpy() for j in range(min(8,n))])
                np.testing.assert_allclose(params[:len(individual)],individual,atol=1e-4,rtol=1e-4)
                self.verified_batch=True
            logits,means,raw=np.split(params,3,axis=1)
            exp=np.exp(logits-logits.max(axis=1,keepdims=True))
            p=exp/exp.sum(axis=1,keepdims=True)
            k=np.minimum((np.random.random(n)[:,None]>p.cumsum(1)).sum(1),p.shape[1]-1)
            r=np.arange(n)
            std=np.clip(np.log1p(np.exp(np.clip(raw[r,k],-30,30))),1e-4,1e2)
            z=np.clip(np.random.normal(np.clip(means[r,k],-1e2,1e2),std),-1e2,1e2)
            turn=self.angle_scaler.inverse_transform(z.reshape(-1,1)).ravel()
            logp+=np.log(p[r,k]+1e-10)
            rad=np.deg2rad(headings)
            lon=lon+v/12*np.sin(rad)/(111*np.cos(np.deg2rad(lat)))
            lat=lat+v/12*np.cos(rad)/111
            headings=(headings+turn)%360
            accel=(v-(speeds[-1] if speeds else vhist[-1]))*12
            feat=np.column_stack((lon,lat,np.sin(np.deg2rad(headings)),np.cos(np.deg2rad(headings)),np.full(n,v),np.full(n,accel),abs(np.deg2rad(turn))/(v/12)))
            ft=np.concatenate([ft[:,1:,:],self.feature_scaler.transform(feat)[:,None,:]],axis=1)
            ang=np.concatenate([ang[:,1:,:],z[:,None,None]],axis=1)
            angles.append(turn)
            speeds.append(v)
    return np.array(angles).T,np.tile(speeds,(n,1)),np.exp(logp)
''',
    'search': r'''
def search_index(core):
    flt=core.similar_trajectory_filter
    flt.load_historical_trajectories(core.reference_dir)
    tracks=flt.historical_trajectories
    indices=[]
    for name,t in tracks.items():
        lo,la=t['lons'],t['lats']

        idx=np.arange(19,len(lo)-10)
        dist=m.haversine_km_vec(la[idx-19],lo[idx-19],la[idx],lo[idx])
        indices.append((name,idx,dist))
    def retrieve(current,top_ratio=.2,max_count=20):
        qlo,qla=current['lons'],current['lats']
        qdist=m.haversine_km_vec(qla[0],qlo[0],qla[-1],qlo[-1])
        feat,_,ang=m.fe_global.calculate_features(qla,qlo,current['times'])
        scores=[]
        total=0
        for name,idx,dist in indices:
            t=tracks[name];lo,la=t['lons'],t['lats']
            total+=len(idx)
            mask=m.haversine_km_vec(la[idx],lo[idx],qla[-1],qlo[-1])<=m.SPATIAL_HARD_THRESHOLD_KM
            mask&=(abs(dist-qdist)<=m.DISTANCE_CHECK_THRESHOLD*qdist if qdist>1e-6 else dist<.1)
            for end in idx[mask]:
                sl=slice(end-19,end+1)
                seg=dict(filename=name,start_idx=end-19,end_idx=end+1,lons=lo[sl],lats=la[sl],times=None)
                score,_=flt.calculate_similarity_score(current,seg,current_features=feat,current_turning_angles=ang)
                if score>0:scores.append((score,seg))
        scores.sort(key=lambda v:-v[0])
        found={}
        for score,seg in scores[:max(1,int(total*top_ratio))]:
            found.setdefault(seg['filename'],dict(total_score=score,segment=seg))
            if len(found)>=max_count:break
        return list(found.values())
    return retrieve
''',
}

# %% Execution and complete prediction archive
from pathlib import Path
from copy import deepcopy
from time import perf_counter
import sys, types, json, zipfile, hashlib
import numpy as np
import pandas as pd
import torch

try:
    PACKAGE_DIR = Path(__file__).resolve().parent
except NameError:
    PACKAGE_DIR = Path.cwd()

def _module(name, source, output):
    module = types.ModuleType(name)
    module.__file__ = str(PACKAGE_DIR / (name + '.py'))
    module.__dict__['__bundle_output_root__'] = str(output)
    sys.modules[name] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module

def prepare_data(region='Gulf', package_dir=None):
    """Extract the regional dataset and checkpoints."""
    if region not in ('Gulf', 'MidAtlantic'): raise ValueError('region: Gulf or MidAtlantic')
    root=Path(package_dir or PACKAGE_DIR).resolve()
    archive=root/(region+'_data.zip')
    dest=root/'runtime_data'/region
    marker=dest/'archive_sha256.txt'
    checksum=hashlib.sha256(archive.read_bytes()).hexdigest()
    if not marker.exists() or marker.read_text()!=checksum:
        dest.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                target=(dest/item.filename).resolve()
                if not target.is_relative_to(dest.resolve()): raise ValueError('Unsafe archive path')
            z.extractall(dest)
        marker.write_text(checksum)
    return root,dest,json.loads((dest/'experiment_records.json').read_text(encoding='utf8'))

def run(region='Gulf', limit=None, package_dir=None, retrain_global=False,
        retrieval_mode='archived', resume=True):
    root,base,records=prepare_data(region,package_dir)
    output=root/'run_results'/region/METHOD
    output.mkdir(parents=True,exist_ok=True)
    saved={k:sys.modules.get(k) for k in ('main','_portable_baseline','_portable_abl','_portable_batch')}
    try:
        main_source=SOURCES['main'].replace('OUTPUT_ROOT = "prediction_results"','OUTPUT_ROOT = __bundle_output_root__')
        m=_module('main',main_source,output)
        torch.set_num_threads(1)
        if 'baseline' in SOURCES: b=_module('_portable_baseline',SOURCES['baseline'],output)
        batch=_module('_portable_batch',SOURCES['batch'],output)
        m.TrajectoryModelCore.predict_trajectories=batch.batched
        original_class=m.AngleMDN
        if 'ablation' in SOURCES: abl=_module('_portable_abl',SOURCES['ablation'],output)
        core=m.TrajectoryModelCore(str(base/'historical'))
        samples={str(s['mmsi']):s for s in m.build_evaluation_samples(str(base/'ais_dataset/test_trajectories'))}
        cases=sorted(records['cases'],key=lambda x:str(x['mmsi']))
        assert len(cases)==(248 if region=='Gulf' else 106)
        if limit is not None: cases=cases[:int(limit)]
        core.similar_trajectory_filter.load_historical_trajectories(core.reference_dir)
        tracks=core.similar_trajectory_filter.historical_trajectories
        offline=0.; net=None
        if FAMILY in ('LSTM','BiLSTM-Attn','BiLSTM-MDN','Prob-GRU'):
            b.REFERENCE_DIR=str(base/'historical')
            b.MODEL_CACHE_PATH=str(output/'global_model.pt') if retrain_global else str(base/'checkpoints'/(region+'_'+FAMILY+'.pt'))
            b.FORCE_RETRAIN=retrain_global
            m.set_seed(42);net,offline=b.train_or_load_model()
        elif FAMILY=='Proposed':
            core.fit_scalers_from_reference()
            cls=abl.AngleMDN_NoAttention if VARIANT=='no_attention' else original_class
            m.AngleMDN=cls;m.set_seed(42)
            path=base/'checkpoints'/(region+('_abl.pt' if VARIANT=='no_attention' else '_full.pt'))
            if retrain_global:
                start=perf_counter();core.train_model(core.pooled_features_scaled,core.pooled_angles_scaled,epochs=300)
                offline=perf_counter()-start;net=core.angle_model
                torch.save(net.state_dict(),output/'global_model.pt')
            else:
                net=cls(feature_dim=7,n_mixtures=core.n_mixtures).to(m.device)
                net.load_state_dict(torch.load(path,map_location=m.device,weights_only=True))
            m.ANGLE_EPOCHS=m.SPEED_EPOCHS=20
        if retrieval_mode not in ('archived','recompute'): raise ValueError('Unknown retrieval_mode')
        if retrieval_mode=='recompute':
            namespace={'np':np,'m':m}
            exec(SOURCES['search'],namespace)
            search=namespace['search_index'](core)
        trainer,speed_trainer=core.train_model,core.train_speed_model
        records_out=[]
        for idx,case in enumerate(cases):
            target=output/'cases'/(str(case['mmsi'])+'.json')
            if resume and target.exists():
                row=json.loads(target.read_text(encoding='utf8'));records_out.append(row);continue
            sample=samples[str(case['mmsi'])]
            assert sample['source_file']==case['source'] and sample['window_start']==case['start']
            seed=42+sample['sample_id'];m.set_seed(seed)
            lo,la=core.filter_trajectory(sample['obs_lons'],sample['obs_lats'])
            if retrieval_mode=='archived':
                ids=case['matches'];retrieval_s=0.
            else:
                start=perf_counter();matches=search(dict(lons=lo,lats=la,times=sample['obs_times']))
                retrieval_s=perf_counter()-start
                ids=[(v['segment']['filename'],int(v['segment']['end_idx'])) for v in matches]
            matches=[dict(segment=dict(filename=n,end_idx=end)) for n,end in ids]
            core.similar_trajectory_filter.find_similar_trajectories=lambda *a,**kw:matches
            ens=None;fit_s=0.;losses=[];start=perf_counter()
            if FAMILY=='CV-KF': top=b.run_cv_kf(lo,la,10)[:2]
            elif FAMILY!='Proposed':
                model=net
                if USE_DTR and ids:

                    X,Y=[],[]
                    for filename,end in ids:
                        t=tracks[filename]
                        a,c=core.filter_trajectory(t['lons'][end-20:end],t['lats'][end-20:end])
                        xy,_=b.make_local_frame(a[-1],c[-1])
                        X.append(np.stack(xy(a,c),axis=1))
                        Y.append(np.stack(xy(t['lons'][end:end+10],t['lats'][end:end+10]),axis=1))
                    x,y=[torch.tensor(np.asarray(v),dtype=torch.float32,device=m.device) for v in (X,Y)]
                    model=deepcopy(net).train();opt=torch.optim.Adam(model.parameters(),lr=.001)
                    start_fit=perf_counter()
                    for _ in range(20):
                        opt.zero_grad()
                        if FAMILY=='Prob-GRU': loss=model.forward_train_loss(x,y)
                        elif FAMILY=='BiLSTM-MDN': loss=b.mdn_nll_loss(*model(x),y)
                        else: loss=(model(x,target_seq=y)-y).square().mean()
                        loss.backward();opt.step();losses.append(loss.item())
                    if m.device.type=='cuda': torch.cuda.synchronize()
                    fit_s=perf_counter()-start_fit
                    models=output/'local_models';models.mkdir(exist_ok=True)
                    torch.save(model.state_dict(),models/(str(case['mmsi'])+'.pt'))
                model.eval();start=perf_counter()
                if FAMILY=='LSTM': top=b.predict_global_lstm(model,lo,la)
                elif FAMILY=='BiLSTM-Attn': top=b.predict_global_bilstm_attn(model,lo,la)
                elif FAMILY=='BiLSTM-MDN': top,ens,_=b.predict_global_mdn_bilstm(model,lo,la,sample_id=seed,n_trajectories=1000)
                else: ens,top=b.predict_prob_gru(model,lo,la,n_trajectories=1000)
            else:
                m.set_seed(seed);m.USE_SIMILARITY_RETRIEVAL=USE_DTR
                core.angle_model=net
                m.AngleMDN=original_class if VARIANT=='original' else lambda **kw:deepcopy(net)
                measured=[]
                def timed(fn,*a,**kw):
                    start_fit=perf_counter();out=fn(*a,**kw)
                    if m.device.type=='cuda':torch.cuda.synchronize()
                    measured.append(perf_counter()-start_fit);return out
                do_train=(USE_DTR and bool(ids)) or VARIANT=='original'
                core.train_model=(lambda *a,**kw:timed(trainer,*a,**kw)) if do_train else (lambda *a,**kw:([],[]))
                core.train_speed_model=lambda *a,**kw:timed(speed_trainer,*a,**kw)
                val,extra=m.evaluate_sample(core,sample)
                top,ens=extra['top5_coords'][0],extra['all_pred_coords']
                fit_s=sum(measured);losses=extra.get('train_losses',[])
            if m.device.type=='cuda':torch.cuda.synchronize()
            predict_s=perf_counter()-start
            if FAMILY=='Proposed': predict_s=float(val['infer_time_sec'])
            ade,fde,_=m.compute_displacement_errors(*top,sample['true_future_lons'],sample['true_future_lats'])
            heading,speed=m.compute_heading_speed_error(*top,sample['true_future_lons'],sample['true_future_lats'])
            row=dict(id=int(case['mmsi']),method=METHOD,ade=float(ade),fde=float(fde),heading=float(heading),speed=float(speed),
                retrieval_s=retrieval_s,train_s=fit_s,predict_s=predict_s,retrieval_mode=retrieval_mode,losses=losses)
            reference_row=next((v for v in case['rows'] if v['method']==METHOD),None)

            if USE_DTR and reference_row is not None:row['reference_retrieval_prepare_s']=reference_row['retrieval_s']
            data=dict(top=top,truth=[sample['true_future_lons'],sample['true_future_lats']],
                observation=[sample['obs_lons'],sample['obs_lats']],seed=seed,
                obs_times=np.asarray(sample['obs_times'],dtype=str),future_times=np.asarray(sample['true_future_times'],dtype=str))
            if ens is not None:
                pts=np.asarray(ens)[:,:,1:];center=pts.mean(0)
                radius=np.quantile(m.haversine_km_vec(pts[:,1],pts[:,0],center[1],center[0]),.95,axis=0)
                error=m.haversine_km_vec(sample['true_future_lats'],sample['true_future_lons'],center[1],center[0])
                metric=m.compute_ensemble_metrics(ens,sample['true_future_lons'],sample['true_future_lats'])
                row.update(minade=float(metric['min_ade']),minfde=float(metric['min_fde']),mpqr=float(radius.mean()),radius=radius.tolist(),error=error.tolist())

                old=next(v for v in case['rows'] if 'true_offset' in v)
                row['true_offset']=old['true_offset']

                cv_namespace={'__name__':'_portable_cv','__file__':str(root/'CV_KF.py')}
                exec(SOURCES['cv'],cv_namespace)
                cv=cv_namespace['run_cv_kf'](lo,la,10)
                row['cv_offset']=m.haversine_km_vec(center[1],center[0],cv[1][1:],cv[0][1:]).tolist()
                data['particles']=ens
            pred=output/'prediction_archive';pred.mkdir(exist_ok=True)
            np.savez_compressed(pred/(str(case['mmsi'])+'.npz'),**data)
            target.parent.mkdir(exist_ok=True);tmp=target.with_suffix('.tmp')
            tmp.write_text(json.dumps(row,ensure_ascii=False,default=lambda v:np.asarray(v).tolist()),encoding='utf8');tmp.replace(target)
            records_out.append(row)
            print(f'{region} {METHOD}: {idx+1}/{len(cases)}, ADE={ade:.4f}, FDE={fde:.4f}',flush=True)
        d=pd.DataFrame(records_out)
        d.drop(columns=['radius','error','cv_offset','true_offset','losses'],errors='ignore').to_csv(output/'metrics.csv',index=False,encoding='utf-8-sig')
        metrics=[k for k in ['ade','fde','minade','minfde','heading','speed','mpqr'] if k in d]
        summary=d[metrics].agg(['mean','std']).T
        summary.to_csv(output/'summary.csv',encoding='utf-8-sig')
        (output/'run_config.json').write_text(json.dumps(dict(method=METHOD,region=region,limit=limit,
            retrain_global=retrain_global,global_training_s=offline,retrieval_mode=retrieval_mode,
            torch=torch.__version__,device=str(m.device),selected_cases=len(cases)),indent=2),encoding='utf8')
        print(summary);print('Saved:',output)
        return d
    finally:
        for name,old in saved.items():
            if old is None:sys.modules.pop(name,None)
            else:sys.modules[name]=old

# %% Jupyter: edit REGION, then run this cell or use %run filename.py
REGION = 'Gulf'
LIMIT = None
if __name__ == '__main__':
    results = run(region=REGION,limit=LIMIT)
