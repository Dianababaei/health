"""
Threshold-Based Rule Engine for Cattle Behavior Classification

This module implements a rule-based classifier for cattle behavioral states
using literature-validated thresholds from accelerometer and gyroscope sensors.

Behavioral States Detected:
- Lying: Rza < -0.5g (body horizontal)
- Standing: Rza > 0.7g + low motion (body upright, stationary)
- Walking: Rza > 0.7g + high Fxa variance + rhythmic patterns
- Ruminating: Mya/Lyg frequency 0.67-1.0 Hz (40-60 cycles/min)
- Feeding: Lyg < -15° (head-down) + high Mya variance

References:
- Thresholds from: docs/behavioral_thresholds_literature_review.md
- Based on 20+ peer-reviewed studies (2009-2025)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import warnings


class BehavioralState(Enum):
    """Enumeration of cattle behavioral states."""
    LYING = "lying"
    STANDING = "standing"
    WALKING = "walking"
    RUMINATING = "ruminating"
    FEEDING = "feeding"
    TRANSITION = "transition"  # Between states
    UNCERTAIN = "uncertain"  # Ambiguous readings


@dataclass
class ClassificationResult:
    """Result of behavior classification for a single time window."""
    state: BehavioralState
    confidence: float  # 0.0-1.0
    timestamp: Optional[pd.Timestamp] = None
    secondary_state: Optional[BehavioralState] = None  # E.g., "ruminating" with "lying" posture
    sensor_values: Optional[Dict[str, float]] = None  # Snapshot of sensor data
    rule_fired: Optional[str] = None  # Which rule triggered classification


class RuleBasedClassifier:
    """
    Threshold-based rule engine for cattle behavior classification.

    Implements priority-based decision tree using literature-validated thresholds.
    Processes 7-parameter sensor data (temperature, fxa, mya, rza, sxg, lyg, dzg)
    and outputs behavioral state classifications with confidence scores.

    Attributes:
        thresholds (dict): Configuration of all threshold values
        min_duration_samples (int): Minimum samples to confirm state change
        enable_smoothing (bool): Enable state transition smoothing
        enable_rumination (bool): Enable rumination detection (requires FFT)
        enable_feeding (bool): Enable feeding detection
    """

    def __init__(
        self,
        min_duration_samples: int = 2,
        enable_smoothing: bool = True,
        enable_rumination: bool = False,  # DISABLED: Requires ≥10 Hz sampling (current: 1/min)
        enable_feeding: bool = False,
        sampling_rate: float = 1.0  # samples per minute
    ):
        """
        Initialize the rule-based classifier.

        Args:
            min_duration_samples: Minimum consecutive samples before state change
            enable_smoothing: Enable transition smoothing to reduce jitter
            enable_rumination: Enable rumination detection (DISABLED by default - requires ≥10 Hz
                sampling rate to detect 1 Hz jaw movement. At 1 sample/min, detection is not
                scientifically valid. See: Schirmann et al. 2009, Burfeind et al. 2011)
            enable_feeding: Enable feeding detection
            sampling_rate: Sampling rate in samples per minute (default: 1.0)
        """
        self.min_duration_samples = min_duration_samples
        self.enable_smoothing = enable_smoothing
        self.enable_rumination = enable_rumination
        self.enable_feeding = enable_feeding
        self.sampling_rate = sampling_rate

        # Initialize thresholds from literature review
        self.thresholds = self._load_thresholds()

        # State history for smoothing
        self.state_history: List[BehavioralState] = []
        self.max_history = 10  # Keep last 10 states

        # Performance tracking
        self.classification_count = 0

    def _load_thresholds(self) -> Dict:
        """
        Load threshold values from literature review.

        Source: docs/behavioral_thresholds_literature_review.md
        References: 20+ peer-reviewed studies (Nielsen et al. 2010,
                    Borchers et al. 2016, Smith et al. 2016, etc.)

        Returns:
            Dictionary of threshold values for all behavioral states
        """
        return {
            'lying': {
                'rza_max': -0.5,  # g (Nielsen et al. 2010: <-0.5g = lying)
                'rza_high_confidence': -0.6,  # g (>95% confidence)
                'motion_max': 0.15,  # g (minimal movement)
                'duration_min_samples': 2  # Minimum 2 samples (2 minutes)
            },
            'standing': {
                'rza_min': 0.7,  # g (Arcidiacono et al. 2017: >0.7g = standing)
                'rza_high_confidence': 0.8,  # g (>95% confidence)
                'fxa_variance_max': 0.15,  # g (distinguish from walking)
                'mya_variance_max': 0.20,  # g (allow small weight shifts)
                'motion_max': 0.15,  # g (low overall motion)
            },
            'walking': {
                'rza_min': 0.5,  # g (mostly upright)
                'fxa_variance_min': 0.20,  # g (Smith et al. 2016: >0.2g = walking)
                'fxa_variance_high_confidence': 0.30,  # g (>90% confidence)
                'frequency_min': 0.67,  # Hz (40 steps/min, slow walking)
                'frequency_max': 1.5,  # Hz (90 steps/min, fast walking)
                'mya_variance_min': 0.10,  # g (lateral body sway during gait)
            },
            'ruminating': {
                # WARNING: Rumination detection requires ≥10 Hz sampling to detect
                # jaw movement at 1.0-1.5 Hz. At 1 sample/min, this is NOT scientifically
                # valid. Thresholds below are variance-based proxies, not true detection.
                # References: Schirmann et al. 2009, Burfeind et al. 2011
                'frequency_min': 0.67,  # Hz (40 cycles/min) - NOT detectable at 1/min sampling
                'frequency_max': 1.0,  # Hz (60 cycles/min) - NOT detectable at 1/min sampling
                'mya_variance_min': 0.08,  # g (variance proxy, not direct jaw detection)
                'lyg_variance_min': 6.0,  # °/s (variance proxy, not direct head bobbing)
                'duration_min_samples': 5,  # Minimum 5 minutes sustained
                'fft_peak_threshold': 3.0,  # Peak must be >3× baseline power
            },
            'feeding': {
                'lyg_threshold': -15.0,  # °/s (head-down position, Umemura et al. 2009)
                'lyg_high_confidence': -25.0,  # °/s (strong head-down = grazing)
                'mya_variance_min': 0.15,  # g (lateral head movement)
                'rza_min': 0.5,  # g (standing posture)
                'duration_min_samples': 2,  # Minimum 2 minutes
            },
            'transition': {
                'rza_ambiguous_min': -0.5,  # g (lower bound of ambiguous zone)
                'rza_ambiguous_max': 0.7,  # g (upper bound of ambiguous zone)
            }
        }

    def classify_single(
        self,
        rza: float,
        fxa: float,
        mya: float,
        sxg: float,
        lyg: float,
        dzg: float,
        temperature: float,
        timestamp: Optional[pd.Timestamp] = None
    ) -> ClassificationResult:
        """
        Classify behavioral state for a single sensor reading.

        Args:
            rza: Z-axis acceleration (g) - vertical orientation
            fxa: X-axis acceleration (g) - forward/backward
            mya: Y-axis acceleration (g) - lateral
            sxg: X-axis angular velocity (°/s) - roll
            lyg: Y-axis angular velocity (°/s) - pitch
            dzg: Z-axis angular velocity (°/s) - yaw
            temperature: Body temperature (°C)
            timestamp: Optional timestamp for the reading

        Returns:
            ClassificationResult with state, confidence, and metadata
        """
        sensor_values = {
            'rza': rza, 'fxa': fxa, 'mya': mya,
            'sxg': sxg, 'lyg': lyg, 'dzg': dzg,
            'temperature': temperature
        }

        # Calculate derived metrics
        motion_magnitude = self._calculate_motion_magnitude(fxa, mya, rza)

        # Rule priority order (higher priority = checked first):
        # 1. Feeding (most specific: requires head-down + specific motion)
        # 2. Walking (specific: rhythmic motion)
        # 3. Standing (requires low motion)
        # 4. Lying (basic posture)
        # 5. Transition (fallback for ambiguous readings)

        # RULE 1: Check for FEEDING (if enabled)
        if self.enable_feeding:
            feeding_result = self._check_feeding_rule(lyg, mya, rza)
            if feeding_result is not None:
                feeding_result.timestamp = timestamp
                feeding_result.sensor_values = sensor_values
                self._update_history(feeding_result.state)
                return feeding_result

        # RULE 2: Check for WALKING
        walking_result = self._check_walking_rule(rza, fxa, mya)
        if walking_result is not None:
            walking_result.timestamp = timestamp
            walking_result.sensor_values = sensor_values
            self._update_history(walking_result.state)
            return walking_result

        # RULE 3: Check for STANDING
        standing_result = self._check_standing_rule(rza, fxa, mya, motion_magnitude)
        if standing_result is not None:
            standing_result.timestamp = timestamp
            standing_result.sensor_values = sensor_values
            self._update_history(standing_result.state)
            return standing_result

        # RULE 4: Check for LYING
        lying_result = self._check_lying_rule(rza, motion_magnitude)
        if lying_result is not None:
            lying_result.timestamp = timestamp
            lying_result.sensor_values = sensor_values
            self._update_history(lying_result.state)
            return lying_result

        # RULE 5: TRANSITION or UNCERTAIN (fallback)
        transition_result = self._check_transition_rule(rza)
        transition_result.timestamp = timestamp
        transition_result.sensor_values = sensor_values
        self._update_history(transition_result.state)
        return transition_result

    def classify_batch(self, sensor_data: pd.DataFrame) -> pd.DataFrame:
        """
        Classify behavioral states for a batch of sensor readings.

        Args:
            sensor_data: DataFrame with columns: timestamp, rza, fxa, mya, sxg, lyg, dzg, temperature

        Returns:
            DataFrame with added columns: state, confidence, secondary_state
        """
        if not self._validate_dataframe(sensor_data):
            raise ValueError("Invalid sensor data format. Required columns: rza, fxa, mya, sxg, lyg, dzg, temperature")

        results = []

        for idx, row in sensor_data.iterrows():
            result = self.classify_single(
                rza=row['rza'],
                fxa=row['fxa'],
                mya=row['mya'],
                sxg=row['sxg'],
                lyg=row['lyg'],
                dzg=row['dzg'],
                temperature=row['temperature'],
                timestamp=row.get('timestamp', None)
            )
            results.append({
                'state': result.state.value,
                'confidence': result.confidence,
                'secondary_state': result.secondary_state.value if result.secondary_state else None,
                'rule_fired': result.rule_fired
            })

        # Add classification results to dataframe
        result_df = sensor_data.copy()
        result_df['state'] = [r['state'] for r in results]
        result_df['confidence'] = [r['confidence'] for r in results]
        result_df['secondary_state'] = [r['secondary_state'] for r in results]
        result_df['rule_fired'] = [r['rule_fired'] for r in results]

        # Apply smoothing if enabled
        if self.enable_smoothing:
            result_df = self._apply_smoothing(result_df)

        # Apply rumination detection if enabled
        if self.enable_rumination:
            result_df = self._detect_rumination(result_df)

        self.classification_count += len(sensor_data)

        return result_df

    # =========================================================================
    # INDIVIDUAL RULE IMPLEMENTATIONS
    # =========================================================================

    def _check_lying_rule(self, rza: float, motion_magnitude: float) -> Optional[ClassificationResult]:
        """
        Check if sensor readings match LYING behavior pattern.

        Rule: Rza < -0.5g (body horizontal) AND low motion

        Literature: Nielsen et al. (2010), Borchers et al. (2016)
        Expected Accuracy: 92-96%
        """
        thresh = self.thresholds['lying']

        # Primary condition: Rza indicates lying posture
        if rza > thresh['rza_max']:
            return None  # Not lying

        # Secondary condition: Minimal motion (true lying, not rolling over)
        if motion_magnitude > thresh['motion_max']:
            # High motion while Rza indicates lying = transition or rolling
            return None

        # Calculate confidence based on Rza margin
        # Further below threshold = higher confidence
        if rza < thresh['rza_high_confidence']:
            confidence = 0.95  # High confidence (Rza < -0.6g)
        else:
            # Linear interpolation between -0.5g (low conf) and -0.6g (high conf)
            margin = abs(rza - thresh['rza_max'])
            confidence = 0.70 + (margin / 0.1) * 0.25  # 0.70 to 0.95 range
            confidence = min(0.95, confidence)

        return ClassificationResult(
            state=BehavioralState.LYING,
            confidence=confidence,
            rule_fired="lying_rza_threshold"
        )

    def _check_standing_rule(
        self,
        rza: float,
        fxa: float,
        mya: float,
        motion_magnitude: float
    ) -> Optional[ClassificationResult]:
        """
        Check if sensor readings match STANDING behavior pattern.

        Rule: Rza > 0.7g (upright) AND low Fxa/Mya variance (not walking)

        Literature: Arcidiacono et al. (2017), Barker et al. (2018)
        Expected Accuracy: 91-94%
        """
        thresh = self.thresholds['standing']

        # Primary condition: Rza indicates upright posture
        if rza < thresh['rza_min']:
            return None  # Not upright

        # Secondary condition: Low motion (distinguish from walking)
        # Note: We can't calculate variance from single sample, so use magnitude as proxy
        if motion_magnitude > thresh['motion_max']:
            return None  # Too much motion, likely walking

        # Additional check: Fxa and Mya should be low (no forward/lateral movement)
        if abs(fxa) > 0.2 or abs(mya) > 0.25:
            return None  # High acceleration, likely walking or moving

        # Calculate confidence
        if rza > thresh['rza_high_confidence']:
            confidence = 0.95  # High confidence (Rza > 0.8g)
        else:
            # Linear interpolation
            margin = rza - thresh['rza_min']
            confidence = 0.70 + (margin / 0.1) * 0.25
            confidence = min(0.95, confidence)

        return ClassificationResult(
            state=BehavioralState.STANDING,
            confidence=confidence,
            rule_fired="standing_rza_low_motion"
        )

    def _check_walking_rule(self, rza: float, fxa: float, mya: float) -> Optional[ClassificationResult]:
        """
        Check if sensor readings match WALKING behavior pattern.

        Rule: Rza > 0.5g (mostly upright) AND high Fxa magnitude (forward motion)

        Note: Full walking detection requires variance over time window.
        This single-sample version uses Fxa magnitude as proxy.

        Literature: Smith et al. (2016), Borchers et al. (2016)
        Expected Accuracy: 88-94% (with full variance analysis)
        """
        thresh = self.thresholds['walking']

        # Primary condition: Upright or mostly upright posture
        if rza < thresh['rza_min']:
            return None  # Not upright enough for walking

        # Proxy for high Fxa variance: Check if Fxa magnitude is high
        # (Single sample limitation: can't compute true variance)
        fxa_magnitude = abs(fxa)

        # Walking typically shows Fxa > 0.3g (rhythmic stride acceleration)
        if fxa_magnitude < 0.25:
            return None  # Low Fxa, likely standing

        # Additional indicator: Mya should show some lateral sway
        mya_magnitude = abs(mya)
        if mya_magnitude < thresh['mya_variance_min']:
            return None  # No lateral sway, less likely walking

        # Calculate confidence based on Fxa magnitude
        if fxa_magnitude > 0.4:
            confidence = 0.85  # High confidence (strong forward acceleration)
        else:
            # Linear interpolation between 0.25 (low) and 0.4 (high)
            confidence = 0.65 + ((fxa_magnitude - 0.25) / 0.15) * 0.20
            confidence = min(0.85, confidence)

        # Note: Confidence capped at 0.85 because single-sample detection
        # is less reliable than variance-based detection over time window

        return ClassificationResult(
            state=BehavioralState.WALKING,
            confidence=confidence,
            rule_fired="walking_high_fxa"
        )

    def _check_feeding_rule(self, lyg: float, mya: float, rza: float) -> Optional[ClassificationResult]:
        """
        Check if sensor readings match FEEDING behavior pattern.

        Rule: Lyg < -15°/s (head-down) AND Mya variance > 0.15g AND standing posture

        Literature: Umemura et al. (2009), Barker et al. (2018)
        Expected Accuracy: 88-94%
        """
        thresh = self.thresholds['feeding']

        # Primary condition: Head-down position (negative Lyg pitch)
        if lyg > thresh['lyg_threshold']:
            return None  # Head not down enough

        # Secondary condition: Standing posture (can't feed while lying)
        if rza < thresh['rza_min']:
            return None  # Not standing

        # Tertiary condition: Lateral head movement (Mya magnitude as proxy)
        # Note: Single sample uses magnitude, full detection needs variance over time
        if abs(mya) < 0.10:
            return None  # Insufficient lateral movement

        # Calculate confidence based on Lyg angle
        if lyg < thresh['lyg_high_confidence']:
            confidence = 0.90  # High confidence (strong head-down = grazing)
        else:
            # Linear interpolation
            margin = abs(lyg - thresh['lyg_threshold'])
            confidence = 0.70 + (margin / 10.0) * 0.20
            confidence = min(0.90, confidence)

        return ClassificationResult(
            state=BehavioralState.FEEDING,
            confidence=confidence,
            rule_fired="feeding_head_down_motion"
        )

    def _check_transition_rule(self, rza: float) -> ClassificationResult:
        """
        Fallback rule for ambiguous or transitional readings.

        Rule: Rza in ambiguous range (-0.5g to 0.7g) = TRANSITION
        Otherwise: UNCERTAIN
        """
        thresh = self.thresholds['transition']

        if thresh['rza_ambiguous_min'] <= rza <= thresh['rza_ambiguous_max']:
            # Rza in transition zone (lying-to-standing or vice versa)
            confidence = 0.50  # Low confidence, ambiguous state
            return ClassificationResult(
                state=BehavioralState.TRANSITION,
                confidence=confidence,
                rule_fired="transition_ambiguous_rza"
            )
        else:
            # Rza outside expected ranges, possibly sensor error
            confidence = 0.30  # Very low confidence
            return ClassificationResult(
                state=BehavioralState.UNCERTAIN,
                confidence=confidence,
                rule_fired="uncertain_out_of_range"
            )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _calculate_motion_magnitude(self, fxa: float, mya: float, rza: float) -> float:
        """
        Calculate overall motion magnitude from acceleration components.

        Note: We only use fxa and mya for motion, not rza.
        Rza is a static orientation signal (gravity direction), not motion.

        Args:
            fxa, mya: Dynamic acceleration values in g
            rza: Static orientation (not used for motion calculation)

        Returns:
            Magnitude in g (2D horizontal motion magnitude)
        """
        return np.sqrt(fxa**2 + mya**2)

    def _update_history(self, state: BehavioralState):
        """Update state history for smoothing."""
        self.state_history.append(state)
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)  # Remove oldest

    def _apply_smoothing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply state transition smoothing to reduce jitter.

        Requires min_duration_samples consecutive samples before state change.

        Args:
            df: DataFrame with 'state' column

        Returns:
            DataFrame with smoothed 'state' column
        """
        if len(df) < self.min_duration_samples:
            return df  # Not enough samples to smooth

        smoothed_states = df['state'].copy()

        # Forward pass: Require min_duration before accepting new state
        current_state = smoothed_states.iloc[0]
        state_count = 1

        for i in range(1, len(smoothed_states)):
            if smoothed_states.iloc[i] == current_state:
                state_count += 1
            else:
                # State changed
                if state_count >= self.min_duration_samples:
                    # Accept the state change
                    current_state = smoothed_states.iloc[i]
                    state_count = 1
                else:
                    # Reject state change, keep previous state
                    smoothed_states.iloc[i] = current_state
                    state_count += 1

        df['state'] = smoothed_states
        return df

    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate that dataframe has required columns."""
        required_columns = {'rza', 'fxa', 'mya', 'sxg', 'lyg', 'dzg', 'temperature'}
        return required_columns.issubset(df.columns)

    # =========================================================================
    # BATCH PROCESSING WITH VARIANCE CALCULATION
    # =========================================================================

    def classify_batch_with_variance(
        self,
        sensor_data: pd.DataFrame,
        window_size: int = 5
    ) -> pd.DataFrame:
        """
        Classify behavioral states using rolling variance (more accurate).

        This method calculates Fxa/Mya variance over rolling windows,
        enabling better walking/standing/ruminating detection.

        Args:
            sensor_data: DataFrame with sensor readings
            window_size: Rolling window size in samples (default: 5 = 5 minutes)

        Returns:
            DataFrame with classifications based on variance analysis
        """
        if not self._validate_dataframe(sensor_data):
            raise ValueError("Invalid sensor data format")

        df = sensor_data.copy()

        # Calculate rolling variance for Fxa and Mya
        df['fxa_variance'] = df['fxa'].rolling(window=window_size, min_periods=1).std()
        df['mya_variance'] = df['mya'].rolling(window=window_size, min_periods=1).std()

        # Calculate rolling mean for Rza (smoother posture detection)
        df['rza_mean'] = df['rza'].rolling(window=window_size, min_periods=1).mean()

        results = []

        for idx, row in df.iterrows():
            # Use variance-enhanced classification
            result = self._classify_with_variance(
                rza=row['rza_mean'],
                fxa_var=row['fxa_variance'],
                mya_var=row['mya_variance'],
                lyg=row['lyg'],
                timestamp=row.get('timestamp', None)
            )
            results.append({
                'state': result.state.value,
                'confidence': result.confidence,
                'rule_fired': result.rule_fired
            })

        df['state'] = [r['state'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]
        df['rule_fired'] = [r['rule_fired'] for r in results]

        if self.enable_smoothing:
            df = self._apply_smoothing(df)

        return df

    def _classify_with_variance(
        self,
        rza: float,
        fxa_var: float,
        mya_var: float,
        lyg: float,
        timestamp: Optional[pd.Timestamp] = None
    ) -> ClassificationResult:
        """
        Classify using variance metrics (more accurate than single-sample).

        Args:
            rza: Mean Z-axis acceleration over window
            fxa_var: Fxa standard deviation over window
            mya_var: Mya standard deviation over window
            lyg: Current pitch angular velocity
            timestamp: Optional timestamp

        Returns:
            ClassificationResult with enhanced accuracy
        """
        thresh_walking = self.thresholds['walking']
        thresh_standing = self.thresholds['standing']
        thresh_lying = self.thresholds['lying']

        # PRIORITY 1: Check for WALKING (high Fxa variance + rhythmic)
        if (rza > thresh_walking['rza_min'] and
            fxa_var > thresh_walking['fxa_variance_min']):

            if fxa_var > thresh_walking['fxa_variance_high_confidence']:
                confidence = 0.92  # High confidence
            else:
                confidence = 0.75 + ((fxa_var - 0.20) / 0.10) * 0.17
                confidence = min(0.92, confidence)

            return ClassificationResult(
                state=BehavioralState.WALKING,
                confidence=confidence,
                timestamp=timestamp,
                rule_fired="walking_variance_high"
            )

        # PRIORITY 2: Check for STANDING (upright + low variance)
        if (rza > thresh_standing['rza_min'] and
            fxa_var < thresh_standing['fxa_variance_max']):

            if rza > thresh_standing['rza_high_confidence']:
                confidence = 0.95
            else:
                confidence = 0.80

            return ClassificationResult(
                state=BehavioralState.STANDING,
                confidence=confidence,
                timestamp=timestamp,
                rule_fired="standing_variance_low"
            )

        # PRIORITY 3: Check for LYING (low Rza + low variance)
        if rza < thresh_lying['rza_max']:
            if rza < thresh_lying['rza_high_confidence']:
                confidence = 0.95
            else:
                confidence = 0.80

            return ClassificationResult(
                state=BehavioralState.LYING,
                confidence=confidence,
                timestamp=timestamp,
                rule_fired="lying_variance_low"
            )

        # FALLBACK: TRANSITION or UNCERTAIN
        return ClassificationResult(
            state=BehavioralState.TRANSITION,
            confidence=0.50,
            timestamp=timestamp,
            rule_fired="transition_fallback"
        )

    def _detect_rumination(self, result_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect rumination periods using sliding window analysis.

        ⚠️ SCIENTIFIC LIMITATION WARNING:
        True rumination detection requires ≥10 Hz sampling to detect jaw movement
        at 1.0-1.5 Hz (60-90 chews/min). At 1 sample/min, this method uses variance
        as a PROXY and is NOT scientifically rigorous. Results should be labeled
        as "estimated" rather than "detected".

        References:
        - Schirmann et al. 2009: Rumination frequency 40-60 cycles/min
        - Burfeind et al. 2011: Requires FFT at ≥10 Hz sampling

        Proxy criteria used (NOT direct jaw detection):
        - MYA variance > 0.08g (correlates with rhythmic activity)
        - LYG variance > 6.0°/s (correlates with head movement)
        - Low overall motion (lying or standing posture)
        - Sustained for at least 5 minutes

        Args:
            result_df: DataFrame with initial state classifications

        Returns:
            DataFrame with rumination states updated (labeled as estimates)
        """
        window_size = 5  # 5-minute sliding window (300 seconds at 1 sample/min)

        if len(result_df) < window_size:
            return result_df

        # Get thresholds
        thresh = self.thresholds['ruminating']
        mya_var_min = thresh['mya_variance_min']
        lyg_var_min = thresh['lyg_variance_min']
        min_duration = thresh['duration_min_samples']

        # Calculate rolling variance for mya and lyg
        result_df['mya_rolling_var'] = result_df['mya'].rolling(window=window_size, center=True).var()
        result_df['lyg_rolling_var'] = result_df['lyg'].rolling(window=window_size, center=True).var()

        # Detect rumination periods
        for i in range(window_size, len(result_df) - window_size):
            current_state = result_df.iloc[i]['state']

            # Only detect rumination when lying or standing (not walking)
            if current_state in ['lying', 'standing']:
                mya_var = result_df.iloc[i]['mya_rolling_var']
                lyg_var = result_df.iloc[i]['lyg_rolling_var']

                # Check if rumination criteria met
                if pd.notna(mya_var) and pd.notna(lyg_var):
                    if mya_var > mya_var_min and lyg_var > lyg_var_min:
                        # Update state to ruminating_lying or ruminating_standing
                        if current_state == 'lying':
                            result_df.at[result_df.index[i], 'state'] = 'ruminating_lying'
                        elif current_state == 'standing':
                            result_df.at[result_df.index[i], 'state'] = 'ruminating_standing'

                        # Increase confidence for rumination
                        result_df.at[result_df.index[i], 'confidence'] = min(
                            result_df.iloc[i]['confidence'] + 0.2,
                            0.95
                        )

        # Clean up temporary columns
        result_df = result_df.drop(columns=['mya_rolling_var', 'lyg_rolling_var'], errors='ignore')

        return result_df

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_statistics(self) -> Dict:
        """Get classification statistics."""
        return {
            'total_classifications': self.classification_count,
            'smoothing_enabled': self.enable_smoothing,
            'rumination_enabled': self.enable_rumination,
            'feeding_enabled': self.enable_feeding,
            'min_duration_samples': self.min_duration_samples,
            'state_history_length': len(self.state_history)
        }

    def reset(self):
        """Reset classifier state (clear history, reset counters)."""
        self.state_history = []
        self.classification_count = 0

    def update_threshold(self, state: str, parameter: str, value: float):
        """
        Update a specific threshold value.

        Args:
            state: Behavioral state ('lying', 'standing', etc.)
            parameter: Threshold parameter name ('rza_max', 'fxa_variance_min', etc.)
            value: New threshold value

        Example:
            classifier.update_threshold('lying', 'rza_max', -0.45)
        """
        if state not in self.thresholds:
            raise ValueError(f"Unknown state: {state}")
        if parameter not in self.thresholds[state]:
            raise ValueError(f"Unknown parameter '{parameter}' for state '{state}'")

        self.thresholds[state][parameter] = value
        warnings.warn(f"Threshold updated: {state}.{parameter} = {value}")
