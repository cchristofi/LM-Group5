import time
import robobo
import neat
import numpy as np
from neat.nn.feed_forward import FeedForwardNetwork
import random

#%%
random.seed(123)

MAX_TIMESTEPS = 35

checkpoint_dir = 'checkpoints'
experiment_name = 'Robobo Experiment 2023-01-21 11;23 - Generation 0'

CONFIG = neat.config.Config(
    neat.genome.DefaultGenome,
    neat.reproduction.DefaultReproduction,
    neat.species.DefaultSpeciesSet,
    neat.stagnation.DefaultStagnation,
    'config/config_neat'
)

best_genome = True

#%%
def sort_checkpoint_population(pop):
    population = [{"idx":idx, "genome":genome, "fitness":genome.fitness}  for idx, genome in pop.population.items()]
    sorted_pop = sorted(population, key= lambda x: x["fitness"], reverse = True)        
    return sorted_pop

def simulation(portnum, bot_num, genomeID, net):
    rob = robobo.SimulationRobobo(bot_num).connect(port = portnum)
    rob.play_simulation()
    
    fitness = 0
    for t in range(MAX_TIMESTEPS):
        irs = rob.read_irs()
        model_input = np.array([x if x!=False else .3 for x in irs])
        
        model_output = np.array(net.activate(model_input)) * 2 - 1
        act = 85 * model_output
        
        timestep_fitness = sum(np.log(np.array([x for x in irs if x != False]))) / 10 + np.abs(model_output).sum()/2
        fitness += (timestep_fitness/MAX_TIMESTEPS)
        rob.move(act[0], act[1], 1000)
    #rob.pause_simulation()
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
    
simulation(20000, "", FeedForwardNetwork.create(genome, CONFIG))
