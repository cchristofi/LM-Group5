import multiprocessing as mp
import time
import robobo
import neat
import os
import math
import socket
import numpy as np
from neat.nn.feed_forward import FeedForwardNetwork
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import random
#%%
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0
    
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

SIMULATION_DEMO_PORT = 21000 if is_port_in_use(21000) else None
POSSIBLE_BOTS= [""]#"#0"]#, "#2"]

#%%
random.seed(123)
def simulate_example(portnum, bot_num, net):
     rob = robobo.SimulationRobobo(bot_num).connect(port = portnum)
     
     rob.play_simulation()
     rob.set_phone_pan(pan_position = 0*math.pi, pan_speed = .5)
     rob.set_phone_tilt(tilt_position = .55, tilt_speed = .1)
     
     fitness = 0
     for t in range(MAX_TIMESTEPS):
         irs = rob.read_irs()
         model_input = np.array([x if x!=False else .3 for x in irs])
         
         model_output = np.array(net.activate(model_input)) * 2 - 1
         act = 85 * model_output
         
         timestep_fitness = sum(np.log(np.array([x for x in irs if x != False]))) / 10 + np.abs(model_output).sum()/2
         fitness += (timestep_fitness/MAX_TIMESTEPS)
    
         rob.move(act[0], act[1], 1000)
     fitness += 3/50 * rob.collected_food()**2
     # Max fitness: MAX_TIMESTEPS * (8 * 0 / 10 + (1+1)/2)/MAX_TIMESTEPS + 3/50 max_food^2
     # -> 1 + 3/50 * 49 = 3.94, minimum is -INF as log(0) -> -inf
    
     rob.stop_world()
     time.sleep(1)
     rob.disconnect()
     print(f"Example simulation fitness: {fitness}\n")

         
def simulation(portnum, bot_num, genomeID, net, fitness_dict):
    rob = robobo.SimulationRobobo(bot_num).connect(port = portnum)
    
    rob.play_simulation()
    rob.set_phone_pan(pan_position = 0*math.pi, pan_speed = .5)
    rob.set_phone_tilt(tilt_position = .55, tilt_speed = .1)
    
    
    fitness = 0
    for t in range(MAX_TIMESTEPS):
        irs = rob.read_irs()
        model_input = np.array([x if x!=False else .3 for x in irs])
        
        model_output = np.array(net.activate(model_input)) * 2 - 1
        act = 85 * model_output
        
        timestep_fitness = sum(np.log(np.array([x for x in irs if x != False]))) / 10 + np.abs(model_output).sum()/2
        fitness += (timestep_fitness/MAX_TIMESTEPS)

        rob.move(act[0], act[1], 1000)
    fitness += 3/50 * rob.collected_food()**2
    # Max fitness: MAX_TIMESTEPS * (8 * 0 / 10 + (1+1)/2)/MAX_TIMESTEPS + 3/50 max_food^2
    # -> 1 + 3/50 * 49 = 3.94, minimum is -INF as log(0) -> -inf

    rob.stop_world()
    time.sleep(1)
    rob.disconnect()
    print(f"Genome: {genomeID}, fitness: {fitness}")
    if genomeID in fitness_dict.keys():
        fitness_dict[genomeID] += fitness/len(POSSIBLE_BOTS)
    else:
        fitness_dict[genomeID] = fitness/len(POSSIBLE_BOTS)



class PoolLearner:
    
    def __init__(self, num_instances, generation = 0):
        self.num_instances = num_instances
        self.generation = generation
        self.manager = mp.Manager()
        
    def evaluate(self, genomes, config):
        self.fitness_dict = self.manager.dict()

        
        self.generation += 1
        
        nets = []
        for genome_id, genome in genomes:
            nets.append((genome_id, genome, FeedForwardNetwork.create(genome, config)))
        print(f"Population Size: {len(nets)}\nArenas {len(POSSIBLE_BOTS)}")
        for i in range(0, len(nets), self.num_instances):

            for bot_number in POSSIBLE_BOTS:
                tic = time.time()
                process_list = []
                for j, (genomeID, genome, net) in enumerate([x for x in nets[i: i + self.num_instances]]):
                    p =  mp.Process(target= simulation, args=(j+20000, bot_number, genomeID, net, self.fitness_dict, ))
                    p.start()
                    process_list.append(p)
                
                for process in process_list:
                    process.join()
                toc = time.time()
        
                print(f"\nInstances [({i+1} - {i+self.num_instances}) / {len(nets)}] This batch of {len(process_list)} took {round(toc-tic)}sec\n\n")
            
        bestFitness_genomeID = max(self.fitness_dict, key = self.fitness_dict.get)

        for genomeID, genome, net in nets:
            if genomeID == bestFitness_genomeID:
                print(f"Best genome from this generation is {genomeID}")
                best_net = net
            genome.fitness = self.fitness_dict[genomeID]
            
        
        if SIMULATION_DEMO_PORT:
            for bot_number in POSSIBLE_BOTS:
                simulate_example(SIMULATION_DEMO_PORT, bot_number, best_net)
                
        fitnesses = [genome.fitness for genome_id, genome in genomes]
        nodes = [genome.size()[0] for genome_id, genome in genomes]
        connections = [genome.size()[1] for genome_id, genome in genomes]
        
        tb.add_scalar("AvgFitness", sum(fitnesses)/len(fitnesses), self.generation)
        tb.add_scalar("MedFitness", np.median(fitnesses), self.generation)
        tb.add_scalar("MaxFitness", max(fitnesses), self.generation)
        tb.add_scalar("MinFitness", min(fitnesses), self.generation)
        tb.add_scalar("StdFitness", np.std(fitnesses), self.generation)
        tb.add_histogram("fitnesses", np.asarray(fitnesses), self.generation)
        tb.add_histogram("num_nodes", np.asarray(nodes), self.generation)
        tb.add_histogram("num_connections", np.asarray(connections), self.generation)
            
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
        
    pop.add_reporter(neat.reporting.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(1, 30, filename_prefix=f'{checkpoint_dir}/{experiment_name} - Generation '))
    
    pool = PoolLearner(num_instances, generation = gen)
    print(f'Running with: {num_instances} instances')
    while True:
        pop.run(pool.evaluate, num_gens)
        
        
if __name__ == "__main__":
    experiment_continuation = None # Either like "Robobo Experiment <date> <time>" or None
    
    if experiment_continuation:
        experiment_name = experiment_continuation

    tb = SummaryWriter(f"tb_runs/{experiment_name}")
    run(num_gens = 5, num_instances = 2,  config = CONFIG, experiment_continuation = experiment_continuation)