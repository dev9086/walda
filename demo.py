import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pathlib import Path
import joblib
from typing import Tuple, Dict, List, Optional
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PredictiveMaintenanceManager:
    def __init__(self, inspection_interval_days: int = 90):
        self.inspection_interval = inspection_interval_days
        self.maintenance_history = []
        self.damage_thresholds = {
            'critical': 75,
            'warning': 50,
            'monitor': 25
        }

    def assess_maintenance_priority(self, damage_pct: float,
                                    degradation_rate: float = None) -> Dict:
        if damage_pct >= self.damage_thresholds['critical']:
            priority = 'CRITICAL'
            recommended_action = 'Immediate replacement required'
            days_to_service = 0
        elif damage_pct >= self.damage_thresholds['warning']:
            priority = 'HIGH'
            recommended_action = 'Schedule maintenance within 2 weeks'
            days_to_service = 14
        elif damage_pct >= self.damage_thresholds['monitor']:
            priority = 'MEDIUM'
            recommended_action = 'Monitor progression, schedule within 1-2 months'
            days_to_service = 45
        else:
            priority = 'LOW'
            recommended_action = 'Routine inspection during regular maintenance'
            days_to_service = self.inspection_interval

        if degradation_rate and degradation_rate > 2:
            days_to_service = max(0, days_to_service // 2)

        return {
            'priority': priority,
            'damage_percentage': damage_pct,
            'degradation_rate_pct_per_month': degradation_rate,
            'recommended_action': recommended_action,
            'days_to_service': days_to_service,
            'next_inspection': (datetime.now() +
                                timedelta(days=days_to_service)).isoformat()
        }

    def predict_remaining_lifespan(self, current_damage: float,
                                   degradation_rate: float) -> float:
        if degradation_rate <= 0:
            return 120
        months_remaining = (self.damage_thresholds['critical'] -
                            current_damage) / degradation_rate
        return max(0, months_remaining)


class AdvancedSolarDamageDetector:
    def __init__(self, model_type='gradient_boosting', image_type='grayscale'):
        self.model = None
        self.scaler = StandardScaler()
        self.model_type = model_type
        self.feature_names: List[str] = []
        self.image_type = image_type
        self.maintenance_manager = PredictiveMaintenanceManager()
        self.panel_history: Dict = {}

    # ---------- DATA LOADING USING labels.csv ----------
    def load_elpv_dataset(self, csv_path: str,
                          images_dir: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load ELPV labels.csv (rel_path, damage, material) and compute features.
        Damage is converted to percentage (0–100).
        """
        logger.info(f"Loading ELPV labels from {csv_path}")
        df = pd.read_csv(
            csv_path,
            sep=r"\s+",
            header=None,
            names=["rel_path", "damage", "material"],
            engine="python"
        )

        X_features = []
        valid_indices = []

        for idx, row in df.iterrows():
            rel_path = str(row["rel_path"]).strip()   # e.g. images/cell0001.png
            img_path = Path(images_dir) / Path(rel_path).name
            if not img_path.exists():
                logger.debug(f"Image not found: {img_path}")
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.debug(f"Failed to read image: {img_path}")
                continue

            feats = self.extract_features(img)
            X_features.append(feats)
            valid_indices.append(idx)

            if (idx + 1) % 500 == 0:
                logger.info(f"Processed {idx + 1} rows...")

        df_valid = df.iloc[valid_indices].reset_index(drop=True)
        X = np.array(X_features, dtype=np.float32)
        # damage in csv is probability (0-1) or fraction → convert to %
        y = df_valid["damage"].values.astype(float) * 100.0

        logger.info(f"Loaded {len(X)} samples with features shape {X.shape}")
        logger.info(f"Damage range: {y.min():.1f}% – {y.max():.1f}%")

        return X, y, df_valid

    # ---------- FEATURE ENGINEERING ----------
    def extract_features(self, image: np.ndarray,
                         thermal_data: Optional[np.ndarray] = None) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if gray.size == 0:
            return np.zeros(23, dtype=np.float32)

        h, w = gray.shape
        features: List[float] = []

        # 1. Brightness & contrast
        mean_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))
        contrast = float(np.max(gray) - np.min(gray))
        features.extend([mean_brightness, std_brightness, contrast])

        # 2. Edges
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        edge_density = float(np.sum(edge_magnitude > 30) / (h * w))
        edge_intensity = float(
            np.mean(edge_magnitude[edge_magnitude > 0])
        ) if np.any(edge_magnitude > 0) else 0.0
        features.extend([edge_density, edge_intensity])

        # 3. Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_variance = float(np.var(laplacian))
        laplacian_mean = float(np.mean(np.abs(laplacian)))
        features.extend([laplacian_variance, laplacian_mean])

        # 4. Histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / (np.sum(hist) + 1e-7)
        entropy = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-7)))
        hist_skewness = float(
            np.mean((gray - np.mean(gray)) ** 3) / ((np.std(gray) ** 3) + 1e-7)
        )
        features.extend([entropy, hist_skewness])

        # 5. LBP
        lbp_var, lbp_mean, lbp_entropy = self._compute_lbp_features(gray)
        features.extend([lbp_var, lbp_mean, lbp_entropy])

        # 6. Canny edges
        edges = cv2.Canny(gray, 50, 150)
        edge_percentage = float(np.sum(edges > 0) / (h * w) * 100)
        features.append(edge_percentage)

        # 7. Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph_open = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        morph_diff = float(
            np.mean(np.abs(gray.astype(float) - morph_open.astype(float)))
        )
        features.append(morph_diff)

        # 8–10. texture, quadrant, frequency
        features.extend(self._compute_texture_features(gray))
        features.extend(self._compute_quadrant_features(gray))
        features.extend(self._compute_frequency_features(gray))

        self.feature_names = [
            'brightness_mean', 'brightness_std', 'contrast',
            'edge_density', 'edge_intensity',
            'laplacian_var', 'laplacian_mean',
            'entropy', 'hist_skewness',
            'lbp_var', 'lbp_mean', 'lbp_entropy',
            'canny_edge_percentage', 'morph_diff',
            'texture_contrast', 'texture_homogeneity', 'texture_energy',
            'quad_mean_var', 'quad_std_var', 'quad_max_var',
            'freq_power_low', 'freq_power_high', 'freq_ratio'
        ]

        return np.array(features[:23], dtype=np.float32)

    def _compute_lbp_features(self, gray: np.ndarray) -> List[float]:
        h, w = gray.shape
        if h < 3 or w < 3:
            return [0.0, 0.0, 0.0]

        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                center = gray[i, j]
                neighbors = [
                    gray[i - 1, j - 1], gray[i - 1, j], gray[i - 1, j + 1],
                    gray[i, j + 1],
                    gray[i + 1, j + 1], gray[i + 1, j], gray[i + 1, j - 1],
                    gray[i, j - 1]
                ]
                binary = ''.join(['1' if n >= center else '0' for n in neighbors])
                lbp[i - 1, j - 1] = int(binary, 2)

        lbp_hist = np.bincount(lbp.flatten(), minlength=256)
        lbp_hist = lbp_hist / (lbp.size + 1e-7)
        entropy = float(-np.sum(lbp_hist * np.log2(lbp_hist + 1e-7)))

        return [float(np.var(lbp)), float(np.mean(lbp)), entropy]

    def _compute_texture_features(self, gray: np.ndarray) -> List[float]:
        g = ((gray - gray.min()) /
             (gray.max() - gray.min() + 1e-7) * 255).astype(np.uint8)
        diff = np.abs(np.diff(g.flatten()))
        contrast = float(np.mean(diff))
        homogeneity = float(np.mean(1.0 / (1.0 + diff)))
        gy, gx = np.gradient(g.astype(float))
        energy = float(np.sum(gy ** 2 + gx ** 2) / g.size)
        return [contrast, homogeneity, energy]

    def _compute_quadrant_features(self, gray: np.ndarray) -> List[float]:
        h, w = gray.shape
        quads = [
            gray[:h // 2, :w // 2],
            gray[:h // 2, w // 2:],
            gray[h // 2:, :w // 2],
            gray[h // 2:, w // 2:]
        ]
        vars_ = [float(np.var(q)) for q in quads if q.size > 0]
        return [float(np.mean(vars_)), float(np.std(vars_)), float(np.max(vars_))]

    def _compute_frequency_features(self, gray: np.ndarray) -> List[float]:
        if gray.size < 4:
            return [0.0, 0.0, 1.0]
        fft = np.abs(np.fft.fft2(gray))
        fft_shift = np.fft.fftshift(fft)
        h, w = fft_shift.shape
        if h >= 4 and w >= 4:
            center = fft_shift[h // 4:3 * h // 4, w // 4:3 * w // 4]
            outer = np.concatenate([fft_shift[:h // 4, :],
                                    fft_shift[3 * h // 4:, :]])
            power_low = float(np.mean(center))
            power_high = float(np.mean(outer))
        else:
            power_low = float(np.mean(fft_shift))
            power_high = float(np.mean(fft_shift))
        ratio = float((power_high + 1) / (power_low + 1))
        return [power_low, power_high, ratio]

    # ---------- TRAINING ----------
    def grid_search_models(self, X_train: np.ndarray,
                           y_train: np.ndarray,
                           cv_folds: int = 3) -> Dict:
        logger.info("Starting grid search...")
        X_scaled = self.scaler.fit_transform(X_train)
        results = {}

        # Random Forest
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }
        rf_grid = GridSearchCV(RandomForestRegressor(random_state=42),
                               rf_params, cv=cv_folds, n_jobs=-1)
        rf_grid.fit(X_scaled, y_train)
        results['RandomForest'] = {
            'model': rf_grid.best_estimator_,
            'best_params': rf_grid.best_params_,
            'best_score': rf_grid.best_score_
        }

        # Gradient Boosting
        gb_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        }
        gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=42),
                               gb_params, cv=cv_folds, n_jobs=-1)
        gb_grid.fit(X_scaled, y_train)
        results['GradientBoosting'] = {
            'model': gb_grid.best_estimator_,
            'best_params': gb_grid.best_params_,
            'best_score': gb_grid.best_score_
        }

        # SVR
        svr_params = {
            'C': [1, 10],
            'epsilon': [0.01, 0.1],
            'kernel': ['rbf']
        }
        svr_grid = GridSearchCV(SVR(), svr_params, cv=cv_folds, n_jobs=-1)
        svr_grid.fit(X_scaled, y_train)
        results['SVR'] = {
            'model': svr_grid.best_estimator_,
            'best_params': svr_grid.best_params_,
            'best_score': svr_grid.best_score_
        }

        return results

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              use_grid_search: bool = True,
              cv_folds: int = 3) -> Dict:
        X_scaled = self.scaler.fit_transform(X_train)
        if use_grid_search:
            results = self.grid_search_models(X_train, y_train, cv_folds)
            best_name = max(results.items(),
                            key=lambda x: x[1]['best_score'])[0]
            self.model = results[best_name]['model']
            logger.info(f"Selected best model: {best_name}")
            return results
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05,
                max_depth=5, random_state=42
            )
            self.model.fit(X_scaled, y_train)
            return {}

    # ---------- PREDICTION + MAINTENANCE ----------
    def predict_with_maintenance(self, features: np.ndarray,
                                 panel_id: str = None,
                                 thermal_image: Optional[np.ndarray] = None) -> Dict:
        if self.model is None:
            raise ValueError("Model not trained.")

        feats = features.astype(np.float32).reshape(1, -1)
        feats_scaled = self.scaler.transform(feats)
        damage_pct = float(np.clip(self.model.predict(feats_scaled)[0], 0, 100))

        degradation_rate = None
        if panel_id and panel_id in self.panel_history:
            prev = self.panel_history[panel_id]
            days = (datetime.now() - prev['date']).days
            if days > 0:
                degradation_rate = (damage_pct - prev['damage']) / (days / 30)

        if panel_id:
            self.panel_history[panel_id] = {
                'damage': damage_pct,
                'date': datetime.now()
            }

        maintenance = self.maintenance_manager.assess_maintenance_priority(
            damage_pct, degradation_rate
        )

        if degradation_rate:
            remaining_months = self.maintenance_manager.predict_remaining_lifespan(
                damage_pct, degradation_rate
            )
        else:
            remaining_months = 120.0

        return {
            'panel_id': panel_id,
            'damage_percentage': damage_pct,
            'degradation_rate_pct_per_month': degradation_rate,
            'remaining_lifespan_months': remaining_months,
            'maintenance': maintenance
        }

    # ---------- SAVE / LOAD ----------
    def save_model(self, filepath: str):
        joblib.dump(
            {'model': self.model,
             'scaler': self.scaler,
             'feature_names': self.feature_names},
            filepath
        )
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        logger.info(f"Model loaded from {filepath}")


# ================= MAIN =================
if __name__ == "__main__":
    csv_path = r"C:\Users\asus\problem 3\elpv-dataset\src\elpv_dataset\data\labels.csv"
    images_dir = r"C:\Users\asus\problem 3\elpv-dataset\src\elpv_dataset\data\images"

    detector = AdvancedSolarDamageDetector()

    try:
        X, y, df_info = detector.load_elpv_dataset(csv_path, images_dir)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        results = detector.train(X_tr, y_tr, use_grid_search=True, cv_folds=3)

        y_pred = detector.model.predict(detector.scaler.transform(X_te))
        y_pred = np.clip(y_pred, 0, 100)

        mse = mean_squared_error(y_te, y_pred)
        mae = mean_absolute_error(y_te, y_pred)
        r2 = r2_score(y_te, y_pred)

        print("\n===== MODEL PERFORMANCE =====")
        print(f"MSE : {mse:.4f}")
        print(f"MAE : {mae:.4f}")
        print(f"R2  : {r2:.4f}")

        model_path = "solar_damage_model.pkl"
        detector.save_model(model_path)

        print("\n===== SAMPLE PREDICTIONS =====")
        for i in range(min(5, len(X_te))):
            result = detector.predict_with_maintenance(
                X_te[i], panel_id=f"PANEL_{i}"
            )
            print(f"Panel {result['panel_id']}: "
                  f"actual={y_te[i]:.2f}%, "
                  f"pred={result['damage_percentage']:.2f}%, "
                  f"priority={result['maintenance']['priority']}")

    except FileNotFoundError as e:
        logger.error(f"Path error: {e}")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
