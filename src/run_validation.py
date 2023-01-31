import time
import robobo
import neat
import numpy as np
from neat.nn.feed_forward import FeedForwardNetwork
import random
import math
import cv2


#%%
random.seed(123)

MAX_TIMESTEPS = 60

checkpoint_dir = 'checkpoints'
experiment_name = 'Robobo Experiment 2023-01-31 14;02 - Generation 25'

CONFIG = neat.config.Config(
    neat.genome.DefaultGenome,
    neat.reproduction.DefaultReproduction,
    neat.species.DefaultSpeciesSet,
    neat.stagnation.DefaultStagnation,
    'config/config_neat'
)

best_genome = True

CAMERA_RESOLUTION = [128, 128]

KERNEL_H = int(CAMERA_RESOLUTION[0]/3)
KERNEL_W = int(CAMERA_RESOLUTION[1]/3)
CONVOLUTION_KERNEL = np.full((KERNEL_H, KERNEL_W), 1/(KERNEL_W*KERNEL_H))

#%%
def sort_checkpoint_population(pop):
    population = [{"idx":idx, "genome":genome, "fitness":genome.fitness}  for idx, genome in pop.population.items()]
    sorted_pop = sorted(population, key= lambda x: x["fitness"], reverse = True)        
    return sorted_pop

def convolve(data, kernel, overlap):
    kernel_h, kernel_w = kernel.shape
    stride_h = kernel_h - overlap
    stride_w = kernel_w - overlap
    height, width = data.shape
    
    n_h_c = (height - kernel_h) // stride_h + 1
    n_w_c = (width  - kernel_w)  // stride_w + 1
    
    result = np.zeros((n_h_c, n_w_c))
    
    for w in range(n_w_c):
        for h in range(n_h_c):
            X_slice = data[stride_h*h:(kernel_h +stride_h*h),stride_w*w:(kernel_w+stride_w*w)]
            result[h,w] = np.sum(np.multiply(X_slice, kernel))
    return result

def simulation(portnum, bot_num, net):
    rob = robobo.SimulationRobobo(bot_num).connect(port = portnum)
    
    rob.play_simulation()
    rob.set_phone_pan(pan_position = 0*math.pi, pan_speed = .5)
    rob.set_phone_tilt(tilt_position = .55, tilt_speed = .1)
    cam = rob.get_image_front()
    hsv = cv2.cvtColor(cam, cv2.COLOR_BGR2HSV)
    
    fitness = 0
    for t in range(MAX_TIMESTEPS):
        irs = rob.read_irs()
        cam = rob.get_image_front()
        hsv = cv2.cvtColor(cam, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (45, 70, 70), (85, 255, 255))/255
        transformed_image = convolve(mask, CONVOLUTION_KERNEL, 0)
        image_input = transformed_image.flatten()
        model_input = np.array([x if x!=False else .3 for x in irs] + image_input.tolist())
        
        model_output = np.array(net.activate(model_input)) * 2 - 1
        act = 85 * model_output
        
        timestep_fitness = transformed_image[2,1]+sum(model_output)/2
        fitness += (timestep_fitness/MAX_TIMESTEPS)
        
      
        rob.move(act[0], act[1], 1000)  
    print("fitness: ", fitness)
   
    #fitness += 3 * (rob.collected_food()/ 7) **3

    rob.stop_world()
    time.sleep(1)
    rob.disconnect()

#%%
checkpoint = f"{checkpoint_dir}/{experiment_name}"

pop = neat.Checkpointer.read_checkpoint(checkpoint)
sorted_list = sort_checkpoint_population(pop)

if best_genome:
    genome = sorted_list[0]['genome']
else:
    for i in range(len(sorted_list)):
        if sorted_list[i]['idx'] == 1:
            genome = sorted_list[i]['genome']
            break
    
simulation(19999, "", FeedForwardNetwork.create(genome, CONFIG))