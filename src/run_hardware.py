#!/usr/bin/env python3

#import multiprocessing as mp
import time
import robobo
import neat
import cv2
import numpy as np
from neat.nn.feed_forward import FeedForwardNetwork
from datetime import datetime
import random
#%%
MAX_TIMESTEPS = 25 # 60

experiment_name = f"Robobo Experiment {datetime.now().strftime('%Y-%m-%d %H;%M')}"
checkpoint_dir = 'checkpoints'


CONFIG = neat.config.Config(
    neat.genome.DefaultGenome,
    neat.reproduction.DefaultReproduction,
    neat.species.DefaultSpeciesSet,
    neat.stagnation.DefaultStagnation,
    'src/config/config_neat' 
)

id_of_genome = 253

CAMERA_RESOLUTION = [640, 480]

KERNEL_H = int(CAMERA_RESOLUTION[0]/3)
KERNEL_W = int(CAMERA_RESOLUTION[1]/3)
CONVOLUTION_KERNEL = np.full((KERNEL_H, KERNEL_W), 1/(KERNEL_W*KERNEL_H))

#%%
random.seed(123)

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

def simulation(net):
    rob = robobo.HardwareRobobo(camera=True).connect(address="192.168.178.244")
        
    rob.set_phone_tilt(75, 100)
    
    fitness = 0
    for t in range(MAX_TIMESTEPS):
        irs = rob.read_irs()
        side_back_irs = irs[:4] + irs[-1:]
        cam = rob.get_image_front()
        hsv = cv2.cvtColor(cam, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (45, 70, 70), (85, 255, 255))/255
        transformed_image = convolve(mask, CONVOLUTION_KERNEL, 0)
        image_input = transformed_image.flatten()
        model_input = np.array([x if x!=False else .3 for x in irs] + image_input.tolist())
        
        model_output = np.array(net.activate(model_input)) * 2 - 1
        act = 85 * model_output
        
        timestep_fitness = transformed_image[2,1] + sum([x-.3 if x else 0 for x in side_back_irs])/5
        fitness += (timestep_fitness/MAX_TIMESTEPS)
     
        print(model_output[0], model_output[1])
        print(int(round(act[0],0)), int(round(act[1],0)))
        rob.move(int(round(act[0],0)), int(round(act[1],0)), 1000)
    
    rob.stop_world()
    time.sleep(1)
    rob.disconnect()

class PoolLearner:
        
    def evaluate(self, genomes, config):

        for genomeID, genome in genomes:
            if genomeID == id_of_genome:
                bestGenome = genome
                simulation(FeedForwardNetwork.create(bestGenome, config))
        
if __name__ == "__main__":
    checkpoint = "Robobo Experiment 2023-01-24 13;52 - Generation 17"

    print(f'Restoring checkpoint: {checkpoint}')
    pop = neat.Checkpointer.restore_checkpoint(f'{checkpoint_dir}/{checkpoint}')

    pool = PoolLearner()
    while True:
        pop.run(pool.evaluate)



