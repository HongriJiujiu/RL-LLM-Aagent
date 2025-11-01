from sumo_rl.environment.env import SumoEnvironment as OriginalSumoEnvironment
"""SUMO Environment for Traffic Signal Control."""

import os
import sys
from typing import Dict, List

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import traci
import numpy as np
import statistics


class SumoEnvironment(OriginalSumoEnvironment):

    def get_feedback(self) -> Dict[str, Dict[str, float]]:
        """
        Get key metrics per traffic light that influence phase decision.
        Returns a dict with per-tls_id stats:
        - halting_vehicles
        - accumulated_waiting_time
        - average_speed
        - incoming_density
        - current_phase
        """
        feedback = {}
        tls_ids = self.ts_ids

        for tls_id in tls_ids:
            # 当前相位编号
            current_phase = self.sumo.trafficlight.getPhase(tls_id)

            # 控制的 lane 列表（所有受控 lane）
            controlled_links = self.sumo.trafficlight.getControlledLinks(tls_id)
            if controlled_links is None:
                continue

            lane_ids = set()
            for link_group in controlled_links:
                for link in link_group:
                    lane_ids.add(link[0])  # 这是来自的车道（from lane）

            halting = 0
            waiting_time = 0.0
            speeds = []
            densities = []

            for lane_id in lane_ids:
                try:
                    halting += self.sumo.lane.getLastStepHaltingNumber(lane_id)
                    waiting_time += self.sumo.lane.getWaitingTime(lane_id)
                    speeds.append(self.sumo.lane.getLastStepMeanSpeed(lane_id))
                    densities.append(self.sumo.lane.getLastStepOccupancy(lane_id))  # 占有率 ∈ [0,1]
                except Exception:
                    continue

            avg_speed = np.mean(speeds) if speeds else 0.0
            avg_density = np.mean(densities) if densities else 0.0

            feedback[tls_id] = {
                "halting_vehicles": halting,
                "accumulated_waiting_time": waiting_time,
                "average_speed": avg_speed,
                "incoming_density": avg_density,
                "current_phase": current_phase,
            }

        return feedback
    
    def get_tls_info(self, tls_ids):
        result = {}
        for tls_id in tls_ids:
            result[tls_id] = self.traffic_signals[tls_id].get_rl_phase_info()
        return result