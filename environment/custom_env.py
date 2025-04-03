import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RuralHealthEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, grid_size=10, num_patients=5, num_facilities=3, num_health_workers=5, 
                 num_mobile_clinics=2, render_mode=None):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_patients = num_patients
        self.num_facilities = num_facilities
        self.num_health_workers = num_health_workers
        self.num_mobile_clinics = num_mobile_clinics
        self.render_mode = render_mode
        
        # State space: Includes positions of patients, facilities, health workers, mobile clinics,
        # patient urgency levels, and availability status of resources
        self.observation_space = spaces.Dict({
            "patient_positions": spaces.Box(low=0, high=grid_size-1, shape=(num_patients, 2), dtype=np.int32),
            "patient_urgency": spaces.Box(low=0, high=10, shape=(num_patients,), dtype=np.int32),
            "facility_positions": spaces.Box(low=0, high=grid_size-1, shape=(num_facilities, 2), dtype=np.int32),
            "facility_capacity": spaces.Box(low=0, high=10, shape=(num_facilities,), dtype=np.int32),
            "health_worker_positions": spaces.Box(low=0, high=grid_size-1, shape=(num_health_workers, 2), dtype=np.int32),
            "health_worker_availability": spaces.Box(low=0, high=1, shape=(num_health_workers,), dtype=np.int32),
            "mobile_clinic_positions": spaces.Box(low=0, high=grid_size-1, shape=(num_mobile_clinics, 2), dtype=np.int32),
            "mobile_clinic_availability": spaces.Box(low=0, high=1, shape=(num_mobile_clinics,), dtype=np.int32),
            "digital_records_status": spaces.Box(low=0, high=1, shape=(num_patients,), dtype=np.int32),
            "insurance_claims_status": spaces.Box(low=0, high=1, shape=(num_patients,), dtype=np.int32),
            "pharmacy_services_status": spaces.Box(low=0, high=1, shape=(num_patients,), dtype=np.int32),
            "stakeholder_info_status": spaces.Box(low=0, high=1, shape=(num_patients,), dtype=np.int32)
        })
        
        # Action space: 8 discrete actions as defined in your proposal
        self.action_space = spaces.Dict({
            "patient_id": spaces.Discrete(num_patients),
            "action_type": spaces.Discrete(8)  # 8 actions as defined in your proposal
        })
        
        # Initialize rendering components
        self.window = None
        self.clock = None
        self.window_size = 700  # Size of the PyGame window
        
        # Initialize state
        self.reset()
    
    def _get_obs(self):
        # Return the current state as an observation
        return {
            "patient_positions": self.patient_positions,
            "patient_urgency": self.patient_urgency,
            "facility_positions": self.facility_positions,
            "facility_capacity": self.facility_capacity,
            "health_worker_positions": self.health_worker_positions,
            "health_worker_availability": self.health_worker_availability,
            "mobile_clinic_positions": self.mobile_clinic_positions,
            "mobile_clinic_availability": self.mobile_clinic_availability,
            "digital_records_status": self.digital_records_status,
            "insurance_claims_status": self.insurance_claims_status,
            "pharmacy_services_status": self.pharmacy_services_status,
            "stakeholder_info_status": self.stakeholder_info_status
        }
    
    def _get_info(self):
        # Return additional information
        return {
            "patients_served": self.patients_served,
            "critical_cases_missed": self.critical_cases_missed,
            "digital_records_completed": self.digital_records_completed,
            "stakeholders_updated": self.stakeholders_updated
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize patient positions and urgency levels
        self.patient_positions = self.np_random.integers(0, self.grid_size, size=(self.num_patients, 2))
        self.patient_urgency = self.np_random.integers(1, 11, size=(self.num_patients,))
        
        # Initialize healthcare facility positions and capacities
        self.facility_positions = self.np_random.integers(0, self.grid_size, size=(self.num_facilities, 2))
        self.facility_capacity = self.np_random.integers(1, 11, size=(self.num_facilities,))
        
        # Initialize community health worker positions and availability
        self.health_worker_positions = self.np_random.integers(0, self.grid_size, size=(self.num_health_workers, 2))
        self.health_worker_availability = self.np_random.integers(0, 2, size=(self.num_health_workers,))
        
        # Initialize mobile clinic positions and availability
        self.mobile_clinic_positions = self.np_random.integers(0, self.grid_size, size=(self.num_mobile_clinics, 2))
        self.mobile_clinic_availability = self.np_random.integers(0, 2, size=(self.num_mobile_clinics,))
        
        # Initialize digital health management statuses
        self.digital_records_status = np.zeros(self.num_patients, dtype=np.int32)
        self.insurance_claims_status = np.zeros(self.num_patients, dtype=np.int32)
        self.pharmacy_services_status = np.zeros(self.num_patients, dtype=np.int32)
        self.stakeholder_info_status = np.zeros(self.num_patients, dtype=np.int32)
        
        # Initialize metrics
        self.patients_served = 0
        self.critical_cases_missed = 0
        self.digital_records_completed = 0
        self.stakeholders_updated = 0
        self.steps = 0
        self.max_steps = 100
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        # Unpack action
        patient_id = action["patient_id"]
        action_type = action["action_type"]
        
        # Initialize reward
        reward = 0
        
        # Process action based on action_type
        if action_type == 0:  # Connect Patient to Nearest Facility
            # Find nearest facility
            distances = np.sum((self.facility_positions - self.patient_positions[patient_id])**2, axis=1)
            nearest_facility = np.argmin(distances)
            
            # Check if facility has capacity
            if self.facility_capacity[nearest_facility] > 0:
                # Connect patient to facility
                self.facility_capacity[nearest_facility] -= 1
                self.patients_served += 1
                
                # Reward based on urgency
                if self.patient_urgency[patient_id] > 7:  # Critical case
                    reward += 10
                else:
                    reward += 5
            else:
                # Facility at capacity
                if self.patient_urgency[patient_id] > 7:  # Critical case
                    reward -= 10
                    self.critical_cases_missed += 1
                else:
                    reward -= 3
        
        elif action_type == 1:  # Link with Community Health Worker
            # Find available health worker
            available_workers = np.where(self.health_worker_availability == 1)[0]
            if len(available_workers) > 0:
                # Find nearest available health worker
                worker_positions = self.health_worker_positions[available_workers]
                distances = np.sum((worker_positions - self.patient_positions[patient_id])**2, axis=1)
                nearest_worker_idx = np.argmin(distances)
                nearest_worker = available_workers[nearest_worker_idx]
                
                # Assign health worker to patient
                self.health_worker_availability[nearest_worker] = 0
                self.patients_served += 1
                
                # Reward based on urgency
                if self.patient_urgency[patient_id] > 5:  # Moderate to high urgency
                    reward += 8
                else:
                    reward += 4
            else:
                # No available health workers
                if self.patient_urgency[patient_id] > 7:  # Critical case
                    reward -= 8
                    self.critical_cases_missed += 1
                else:
                    reward -= 2
        
        elif action_type == 2:  # Schedule Mobile Clinic Visit
            # Find available mobile clinic
            available_clinics = np.where(self.mobile_clinic_availability == 1)[0]
            if len(available_clinics) > 0:
                # Find nearest available mobile clinic
                clinic_positions = self.mobile_clinic_positions[available_clinics]
                distances = np.sum((clinic_positions - self.patient_positions[patient_id])**2, axis=1)
                nearest_clinic_idx = np.argmin(distances)
                nearest_clinic = available_clinics[nearest_clinic_idx]
                
                # Schedule mobile clinic visit
                self.mobile_clinic_availability[nearest_clinic] = 0
                self.patients_served += 1
                
                # Reward based on urgency and distance
                distance = distances[nearest_clinic_idx]
                if distance < 5:  # Close proximity
                    reward += 7
                else:
                    reward += 3
            else:
                # No available mobile clinics
                if self.patient_urgency[patient_id] > 6:  # Moderately critical case
                    reward -= 7
                    self.critical_cases_missed += 1
                else:
                    reward -= 2
        
        elif action_type == 3:  # Initiate Emergency Response
            # Emergency response is always available but costly
            # Only worthwhile for critical cases
            if self.patient_urgency[patient_id] > 8:  # Highly critical case
                reward += 15
                self.patients_served += 1
            elif self.patient_urgency[patient_id] > 5:  # Moderately critical case
                reward += 5
                self.patients_served += 1
            else:  # Non-critical case
                reward -= 10  # Penalty for unnecessary emergency response
        
        elif action_type == 4:  # Generate Digital Health Records
            if self.digital_records_status[patient_id] == 0:
                self.digital_records_status[patient_id] = 1
                self.digital_records_completed += 1
                reward += 8
            else:
                reward -= 2  # Penalty for redundant action
        
        elif action_type == 5:  # Process Insurance Claims
            if self.insurance_claims_status[patient_id] == 0:
                self.insurance_claims_status[patient_id] = 1
                reward += 6
            else:
                reward -= 2  # Penalty for redundant action
        
        elif action_type == 6:  # Coordinate Pharmacy Services
            if self.pharmacy_services_status[patient_id] == 0:
                self.pharmacy_services_status[patient_id] = 1
                reward += 7
            else:
                reward -= 2  # Penalty for redundant action
        
        elif action_type == 7:  # Update Stakeholder Information
            if self.stakeholder_info_status[patient_id] == 0:
                self.stakeholder_info_status[patient_id] = 1
                self.stakeholders_updated += 1
                reward += 5
            else:
                reward -= 1  # Penalty for redundant action
        
        # Increment step counter
        self.steps += 1
        
        # Check terminal conditions
        done = False
        
        # Terminal condition 1: All patients served
        if self.patients_served >= self.num_patients:
            done = True
            reward += 20  # Bonus for serving all patients
        
        # Terminal condition 2: All digital records completed
        if np.all(self.digital_records_status == 1):
            reward += 15  # Bonus for completing all digital records
        
        # Terminal condition 3: All stakeholders updated
        if np.all(self.stakeholder_info_status == 1):
            reward += 10  # Bonus for updating all stakeholders
        
        # Terminal condition 4: Maximum steps reached
        if self.steps >= self.max_steps:
            done = True
            
            # Penalty for critical cases missed at the end
            if self.critical_cases_missed > 0:
                reward -= 5 * self.critical_cases_missed
        
        observation = self._get_obs()
        info = self._get_info()

        # Add these debug print statements here
        print(f"Action taken: {action}, Patient urgency: {self.patient_urgency[patient_id]}")
        print(f"Reward: {reward}, Patients served: {self.patients_served}")
        
        return observation, reward, done, False, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            from environment.rendering import render_rural_health_env
            return render_rural_health_env(self)
        elif self.render_mode == "human":
            from environment.rendering import render_rural_health_env
            frame = render_rural_health_env(self)
            # Display the frame (implementation depends on your setup)
            return frame
    
    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None