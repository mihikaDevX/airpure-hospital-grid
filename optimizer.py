"""
AirPure Hospital Grid - Ventilation Optimizer
===============================================
Converts anomaly flags (from the autoencoder) and forecasts (from the
airflow model) into concrete HVAC control recommendations per ward.

Starts rule-based (transparent, explainable, safe to demo to
non-technical reviewers) with clear extension points for a
reinforcement-learning controller once enough live/simulated interaction
data exists to train one (see `RLPlaceholderNote` at the bottom).

Thresholds are illustrative — in a real deployment they'd be set with
hospital infection-control staff, per ASHRAE / national ventilation
guidelines for healthcare facilities.
"""

from dataclasses import dataclass

CO2_WARN_PPM = 800
CO2_CRITICAL_PPM = 1000
AIRFLOW_MIN_MS = 0.15
PM25_WARN = 25


@dataclass
class ControlAction:
    ward: str
    action: str
    fan_speed_delta_pct: int
    damper_delta_pct: int
    priority: str
    reason: str


def decide_action(ward, current_reading, forecast, is_anomaly, anomaly_type):
    """Return a ControlAction given current state, forecast and anomaly flag."""

    co2_now = current_reading["co2_ppm"]
    co2_next = forecast["co2_ppm"]
    flow_now = current_reading["airflow_ms"]
    flow_next = forecast["airflow_ms"]
    pm25_now = current_reading["pm25"]

    # 1. Reactive: confirmed anomaly from the autoencoder takes priority
    if is_anomaly:
        if anomaly_type == "stagnation":
            return ControlAction(ward, "INCREASE_FAN_SPEED", +40, +20, "HIGH",
                                  "Anomaly detected: stagnant air pocket (low flow, rising CO2)")
        if anomaly_type == "contamination_spike":
            return ControlAction(ward, "INCREASE_EXHAUST_AND_FILTER", +50, +30, "CRITICAL",
                                  "Anomaly detected: particulate contamination spike")
        if anomaly_type == "hvac_fault":
            return ControlAction(ward, "ALERT_MAINTENANCE", 0, 0, "CRITICAL",
                                  "Anomaly detected: possible HVAC hardware fault (flow collapsed, temp/humidity drifting)")
        return ControlAction(ward, "INCREASE_FAN_SPEED", +25, +10, "HIGH",
                              "Unclassified anomaly detected by autoencoder")

    # 2. Predictive: forecast crossing a threshold before it happens
    if co2_next > CO2_CRITICAL_PPM:
        return ControlAction(ward, "PREEMPT_INCREASE_FAN_SPEED", +30, +15, "MEDIUM",
                              f"CO2 forecast to reach {co2_next:.0f}ppm next interval")
    if co2_next > CO2_WARN_PPM and co2_next > co2_now:
        return ControlAction(ward, "PREEMPT_MILD_INCREASE", +15, +5, "LOW",
                              f"CO2 trending up, forecast {co2_next:.0f}ppm")
    if flow_next < AIRFLOW_MIN_MS:
        return ControlAction(ward, "PREEMPT_INCREASE_FAN_SPEED", +25, +10, "MEDIUM",
                              f"Airflow forecast to drop to {flow_next:.2f}m/s")
    if pm25_now > PM25_WARN:
        return ControlAction(ward, "INCREASE_FILTRATION", +20, 0, "MEDIUM",
                              f"PM2.5 elevated at {pm25_now:.1f}ug/m3")

    # 3. Steady state
    return ControlAction(ward, "MAINTAIN", 0, 0, "NONE", "Ward within normal ventilation parameters")


class RLPlaceholderNote:
    """
    Extension point, not yet implemented in this prototype.

    Once enough (state, action, resulting-air-quality) triples accumulate
    from the rule-based controller in production, an RL agent (e.g. DQN /
    PPO from Stable-Baselines3) can be trained with:
        state  = [co2, pm25, pm10, temp, humidity, airflow, forecast_co2, forecast_flow]
        action = {fan_speed_delta, damper_delta} (discretized)
        reward = -w1*infection_risk_proxy - w2*energy_cost - w3*occupant_discomfort
    replacing decide_action() with agent.predict(state) while keeping the
    same ControlAction output contract so the rest of the pipeline (digital
    twin, dashboard, alerts) needs no changes.
    """
    pass


if __name__ == "__main__":
    demo_reading = {"co2_ppm": 950, "airflow_ms": 0.12, "pm25": 14}
    demo_forecast = {"co2_ppm": 1080, "airflow_ms": 0.10}
    action = decide_action("ICU", demo_reading, demo_forecast, is_anomaly=True, anomaly_type="stagnation")
    print(action)
