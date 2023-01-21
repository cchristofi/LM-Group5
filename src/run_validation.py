import multiprocessing as mp
import time
import robobo
import neat
import os
import numpy as np
from neat.nn.feed_forward import FeedForwardNetwork
from datetime import datetime
import random
#%%
MAX_TIMESTEPS = 30

experiment_name = f"Robobo Experiment {datetime.now().strftime('%Y-%m-%d %H;%M')}"
checkpoint_dir = 'checkpoints'

def mkdir(dir_name: str):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
# mkdir(experiment_name)
mkdir(checkpoint_dir)


CONFIG = neat.config.Config(
    neat.genome.DefaultGenome,
    neat.reproduction.DefaultReproduction,
    neat.species.DefaultSpeciesSet,
    neat.stagnation.DefaultStagnation,
    'config/config_neat'
)
POSSIBLE_BOTS= [""]

#%%
random.seed(123)
def simulation(portnum, bot_num, genomeID, net): #, fitness_dict):
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


class PoolLearner:
    
    def __init__(self, num_instances, generation = 0):
        self.num_instances = num_instances
        self.generation = generation
        manager = mp.Manager()
        self.fitness_dict = manager.dict()
        
    def evaluate(self, genomes, config):

        for genome_id, genome in genomes:
            if genome_id == 1:
                bestGenome = genome
                simulation(20000, "", genome_id, FeedForwardNetwork.create(bestGenome, config))

            
def get_last_checkpoint(experiment_name):
    """Find the latest checkpoint in the current directory."""
    local_dir = os.path.join(os.path.dirname(__file__), checkpoint_dir)
    checkpoints = [int(f.split('- Generation ')[-1]) for f in os.listdir(local_dir) if f.startswith(experiment_name)]
    checkpoints = sorted(checkpoints)
    
    if not checkpoints:
        return ''
    for checkpoint in checkpoints[:-2]:
        os.remove(f'{checkpoint_dir}/{experiment_name} - Generation {checkpoint}')
    
    last_number = checkpoints[-1]
    latest_checkpoint = f'{checkpoint_dir}/{experiment_name} - Generation {last_number}'
    return latest_checkpoint, last_number 
        
def run(num_gens, num_instances, config, experiment_continuation = None):

    
    if experiment_continuation:
        checkpoint, gen = get_last_checkpoint(experiment_continuation)
        print(f'Restoring checkpoint: {checkpoint}')
        pop = neat.Checkpointer.restore_checkpoint(checkpoint)
        
    else:
        gen = 0
        pop = neat.Population(config)
        
    pool = PoolLearner(num_instances, generation = gen)
    print(f'Running with: {num_instances} instances')
    while True:
        pop.run(pool.evaluate, num_gens)
        
        
if __name__ == "__main__":
    experiment_continuation = "Robobo Experiment 2023-01-21 11;23"

    run(num_gens = 100, num_instances = 1,  config = CONFIG, experiment_continuation = experiment_continuation)