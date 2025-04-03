import pygame
import numpy as np

def render_rural_health_env(env):
    """
    Render the Rural Health Environment using PyGame
    """
    if env.window is None:
        pygame.init()
        pygame.display.init()
        env.window = pygame.display.set_mode((env.window_size, env.window_size))
        pygame.display.set_caption("Rural Health Access Optimization System")
    
    if env.clock is None:
        env.clock = pygame.time.Clock()
    
    canvas = pygame.Surface((env.window_size, env.window_size))
    canvas.fill((255, 255, 255))  # White background
    
    # Calculate grid cell size
    cell_size = env.window_size / env.grid_size
    
    # Draw grid lines
    for x in range(env.grid_size + 1):
        pygame.draw.line(
            canvas, (200, 200, 200), 
            (0, cell_size * x), 
            (env.window_size, cell_size * x), 
            width=1
        )
        pygame.draw.line(
            canvas, (200, 200, 200), 
            (cell_size * x, 0), 
            (cell_size * x, env.window_size), 
            width=1
        )
    
    # Draw patients (red circles with size based on urgency)
    for i in range(env.num_patients):
        position = env.patient_positions[i]
        urgency = env.patient_urgency[i]
        
        # Color gradient based on urgency (green to red)
        color = (min(255, urgency * 25), max(0, 255 - urgency * 25), 0)
        
        # Size based on urgency
        size = max(5, min(15, urgency * 1.5))
        
        pygame.draw.circle(
            canvas, color,
            ((position[0] + 0.5) * cell_size, (position[1] + 0.5) * cell_size),
            size
        )
    
    # Draw healthcare facilities (blue squares)
    for i in range(env.num_facilities):
        position = env.facility_positions[i]
        capacity = env.facility_capacity[i]
        
        # Color intensity based on capacity
        color_intensity = max(50, min(255, capacity * 25))
        color = (0, 0, color_intensity)
        
        pygame.draw.rect(
            canvas, color,
            pygame.Rect(
                position[0] * cell_size, position[1] * cell_size,
                cell_size, cell_size
            )
        )
    
    # Draw community health workers (green triangles)
    for i in range(env.num_health_workers):
        position = env.health_worker_positions[i]
        available = env.health_worker_availability[i]
        
        # Color based on availability
        color = (0, 255, 0) if available else (100, 100, 100)
        
        # Draw triangle
        center_x = (position[0] + 0.5) * cell_size
        center_y = (position[1] + 0.5) * cell_size
        size = cell_size * 0.4
        
        pygame.draw.polygon(
            canvas, color, [
                (center_x, center_y - size),
                (center_x - size, center_y + size),
                (center_x + size, center_y + size)
            ]
        )
    
    # Draw mobile clinics (orange diamonds)
    for i in range(env.num_mobile_clinics):
        position = env.mobile_clinic_positions[i]
        available = env.mobile_clinic_availability[i]
        
        # Color based on availability
        color = (255, 165, 0) if available else (100, 100, 100)
        
        # Draw diamond
        center_x = (position[0] + 0.5) * cell_size
        center_y = (position[1] + 0.5) * cell_size
        size = cell_size * 0.4
        
        pygame.draw.polygon(
            canvas, color, [
                (center_x, center_y - size),
                (center_x + size, center_y),
                (center_x, center_y + size),
                (center_x - size, center_y)
            ]
        )
    
    # Draw status information
    font = pygame.font.SysFont('Arial', 16)
    
    # Display metrics
    metrics_text = [
        f"Patients Served: {env.patients_served}/{env.num_patients}",
        f"Critical Cases Missed: {env.critical_cases_missed}",
        f"Digital Records: {env.digital_records_completed}/{env.num_patients}",
        f"Stakeholders Updated: {env.stakeholders_updated}/{env.num_patients}",
        f"Steps: {env.steps}/{env.max_steps}"
    ]
    
    for i, text in enumerate(metrics_text):
        text_surface = font.render(text, True, (0, 0, 0))
        canvas.blit(text_surface, (10, 10 + i * 20))
    
    if env.render_mode == "human":
        env.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        env.clock.tick(env.metadata["render_fps"])
    
    return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))