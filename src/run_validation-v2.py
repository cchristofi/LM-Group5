import time
import robobo
import neat
import numpy as np
from neat.nn.feed_forward import FeedForwardNetwork
import random
import math
import cv2
import socket


#%%
random.seed(123)
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

MAX_TIMESTEPS = 60

checkpoint_dir = 'checkpoints'
experiment_name = 'Robobo Experiment 2023-01-31 21;24 - Generation 17'

CONFIG = neat.config.Config(
    neat.genome.DefaultGenome,
    neat.reproduction.DefaultReproduction,
    neat.species.DefaultSpeciesSet,
    neat.stagnation.DefaultStagnation,
    'config/config_neat'
)

best_genome = True

SIMULATION_DEMO_PORT = 21000 if is_port_in_use(21000) else None
POSSIBLE_BOTS= [""]#"#0"]#, "#2"]

FOOD_DETECTION_THRESHOLD = 0.07
MAX_SPEED = 75


#%%
random.seed(123)
def sort_checkpoint_population(pop):
    population = [{"idx":idx, "genome":genome, "fitness":genome.fitness}  for idx, genome in pop.population.items()]
    sorted_pop = sorted(population, key= lambda x: x["fitness"], reverse = True)
        
    return sorted_pop


def extract_cluster(hsv, mask):
    masked = np.zeros(hsv.shape[:-1])
    for m_bottom, m_top in mask:
        masked += cv2.inRange(hsv, m_bottom, m_top)/255
    
    if masked.sum() == 0:
        return [1.25, .75] 
    
    cols = np.average(masked, axis = 0)
    x = np.average(np.arange(cols.size), weights = cols)
    rows = np.average(masked, axis = 1)
    y = np.average(np.arange(rows.size), weights = rows)

    x = 2*x/hsv.shape[1] - 1
    y = 1-y/hsv.shape[0]
    return [x, y]


def distance_to_target(target, bias=0):
    distance_squared = np.square(target[0])+np.square(target[1])
    distance = np.sqrt(distance_squared) #tussen 0 en 1.46 
    
    distance += (distance > 0 ) * bias 
    
    return distance 

         
def simulation(portnum, bot_num, net, fitness_dict = None, genomeID = None, log_run = True, example_run = False):
    rob = robobo.SimulationRobobo(bot_num).connect(port = portnum)
    
    rob.play_simulation()
    rob.set_phone_pan(pan_position = 0*math.pi, pan_speed = .5)
    tilt_degrees = 22.5
    rob.set_phone_tilt(tilt_position =0.35 + (tilt_degrees/90)*1.55, tilt_speed = .1)
    
    scores = []
    for t in range(MAX_TIMESTEPS):
        if rob.base_detects_food():
            break
        irs = rob.read_irs()
        front_middle_irs = irs[5]
        cam = rob.get_image_front()
        hsv = cv2.cvtColor(cam, cv2.COLOR_BGR2HSV)
        
        hasFoodInGrip = front_middle_irs and front_middle_irs < FOOD_DETECTION_THRESHOLD
        
        if hasFoodInGrip:
            green = extract_cluster(hsv, mask = [((35, 70, 70), (70, 255, 255))])
            red = [0, 0]
        else:
            green = [0, 0]
            red = extract_cluster(hsv, mask = [((170, 70, 70), (180, 255, 255)), ((0, 70, 70), (10, 255, 255))])
        
        model_input = np.array(green + red) #[x if x!=False else .3 for x in irs] + 
        
        model_output = np.array(net.activate(model_input)) * 2 - 1
        act = MAX_SPEED * model_output
        
        dis_green = distance_to_target(green) + 1.46 * (1-hasFoodInGrip)
        dis_red = distance_to_target(red)

        # Found scores per timestep
        timestep_score = {
            "time": t,
            "distToRed":   (-1 * dis_red) / MAX_TIMESTEPS,
            "distToGreen": (-1 * dis_green) / MAX_TIMESTEPS}
        
        timestep_score["Cumulative"] = sum([v for k, v in timestep_score.items() if k != "time"]) + (0 if t==0 else scores[-1]["Cumulative"])

        timestep_score = timestep_score | {f"irs{i}":v for i, v in enumerate(irs)} | {"green_x":green[0], "green_y":green[1], "red_x":red[0], "red_y":red[1]}

        scores.append(timestep_score)
        
      
        rob.move(act[0], act[1], 1000)
        
    # Accumulation of the different fitness parts
    distToRed   = sum([x["distToRed"]   for x in scores])
    distToGreen = sum([x["distToGreen"] for x in scores])
    BaseDetectFood = 3*rob.base_detects_food()
    fitness = distToRed + distToGreen + BaseDetectFood

    scores.append({"time":MAX_TIMESTEPS, "Cumulative":fitness})

    rob.stop_world()
    time.sleep(1)
    rob.disconnect()

#%%
checkpoint = f"{checkpoint_dir}/{experiment_name}"

pop = neat.Checkpointer.read_checkpoint(checkpoint)
sorted_list = sort_checkpoint_population(pop)
# print(sorted_list)

if best_genome:
    genome = sorted_list[0]['genome']
else:
    for i in range(len(sorted_list)):
        print("genome options: ")
        print(sorted_list[i]['idx'])
        if sorted_list[i]['idx'] == 1:
            genome = sorted_list[i]['genome']
            break

simulation(19999, "", FeedForwardNetwork.create(genome, CONFIG))