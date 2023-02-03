import multiprocessing as mp
import time
import robobo
import neat
import os
import math
import cv2
import socket
import numpy as np
from neat.nn.feed_forward import FeedForwardNetwork
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import random
#%%
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0
    
    
MAX_TIMESTEPS = 60 #has to be minute!!! according assignment 

experiment_name = f"Robobo Experiment {datetime.now().strftime('%Y-%m-%d %H;%M')}"
checkpoint_dir = 'checkpoints'
log_dir = "log"

def mkdir(dir_name: str):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

mkdir(checkpoint_dir)
mkdir(log_dir)


CONFIG = neat.config.Config(
    neat.genome.DefaultGenome,
    neat.reproduction.DefaultReproduction,
    neat.species.DefaultSpeciesSet,
    neat.stagnation.DefaultStagnation,
    'config/config_neat'
)

SIMULATION_DEMO_PORT = 21000 if is_port_in_use(21000) else None
POSSIBLE_BOTS= ["#0"]#"#0"]#, "#2"]

FOOD_DETECTION_THRESHOLD = 0.07
MAX_SPEED = 75


#%%
random.seed(123)
def sort_checkpoint_population(pop):
    population = [{"idx":idx, "genome":genome, "fitness":genome.fitness}  for idx, genome in pop.population.items()]
    sorted_pop = sorted(population, key= lambda x: x["fitness"], reverse = True)
        
    return sorted_pop

         
def simulation(portnum, bot_num, net, fitness_dict = None, genomeID = None, log_run = True, example_run = False):
    rob = robobo.SimulationRobobo(bot_num).connect(port = portnum)
    
    rob.play_simulation()
    
    scores = []
    for t in range(MAX_TIMESTEPS):
        try:
            irs = rob.read_irs()
        except:
            irs = [False for i in range(8)]
        model_input = np.array([x if x!=False else .3 for x in irs])

        model_output = np.array(net.activate(model_input)) * 2 - 1
        act = MAX_SPEED * model_output
        
        
        # Found scores per timestep
        timestep_score = {
            "time": t,
            "logDistance2blocks":   (sum(np.log(np.array([x for x in irs if x != False]))) / 10) / MAX_TIMESTEPS,
            "powerDifferential": (- np.abs(model_output[0] - model_output[1])/2) / MAX_TIMESTEPS,
            "power": (np.abs(model_output[0] + model_output[1])/2) / MAX_TIMESTEPS,
            "powerPositive": (sum([x for x in model_output if x>0]) /2) / MAX_TIMESTEPS}
        
        timestep_score["Cumulative"] = sum([v for k, v in timestep_score.items() if k != "time"]) + (0 if t==0 else scores[-1]["Cumulative"])

        timestep_score = timestep_score | {f"irs{i}":v for i, v in enumerate(irs)}

        scores.append(timestep_score)
        
      
        rob.move(act[0], act[1], 1000)
        
    # Accumulation of the different fitness parts
    logDistance2blocks   = sum([x["logDistance2blocks"]   for x in scores])
    powerDifferential = sum([x["powerDifferential"] for x in scores])
    power = sum([x["power"] for x in scores])
    powerPositive = sum([x["powerPositive"] for x in scores])
    
    #Fitness 1
    fitness = logDistance2blocks + powerDifferential
    #Fitness 2
    fitness = logDistance2blocks + power
    #Fitness 3
    fitness = logDistance2blocks + powerPositive

    scores.append({"time":MAX_TIMESTEPS, "Cumulative":fitness})

    rob.stop_world()
    time.sleep(1)
    rob.disconnect()
    
    if log_run:
        pd.DataFrame.from_records(scores).to_csv(f"{log_dir}/{experiment_name}{'_' + str(genomeID) if genomeID else '_example'}.csv", index = False)
        
    if example_run:
        return


        
    # Returning the found fitness (and the fitness parts) to the evaluate function
    print(f"Genome: {genomeID}\tfitness: {fitness:.2f}\tlogDistance2blocks: {logDistance2blocks:.2f}\tpowerDifferential: {powerDifferential:.2f}\tpower: {power:.2f}\tpowerPositive: {powerPositive:.2f}")
    if genomeID in fitness_dict.keys():
        fitness_dict[genomeID]["fitness"] += fitness/len(POSSIBLE_BOTS)
        fitness_dict[genomeID]["fitness_parts"] = {"t":t, "logDistance2blocks": logDistance2blocks, "powerDifferential": powerDifferential, "power": power, "powerPositive": powerPositive}
    else:
        fitness_dict[genomeID] = {"fitness":fitness/len(POSSIBLE_BOTS),
                                  "fitness_parts": {"t":t, "logDistance2blocks": logDistance2blocks, "powerDifferential": powerDifferential, "power": power, "powerPositive": powerPositive}}
        



class PoolLearner:
    def __init__(self, num_instances, generation = 0):
        self.num_instances = num_instances
        self.generation = generation
        self.manager = mp.Manager()
        self.pop_fitness_list = []
        
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
                    p =  mp.Process(target= simulation, args=(j+20000, bot_number, net, self.fitness_dict, genomeID, ))
                    p.start()
                    process_list.append(p)
                
                for process in process_list:
                    process.join()
                toc = time.time()
        
                print(f"\nInstances [({i+1} - {min(i+self.num_instances, len(nets))}) / {len(nets)}] This batch of {len(process_list)} took {round(toc-tic)}sec\n\n")
            
        bestFitness_genomeID = max(self.fitness_dict, key = lambda x: self.fitness_dict.get(x)["fitness"])

        for genomeID, genome, net in nets:
            if genomeID == bestFitness_genomeID:
                print(f"Best genome from this generation is {genomeID}")
                best_net = net
            
            results = self.fitness_dict[genomeID]
            genome.fitness = results["fitness"]
            
        
        if SIMULATION_DEMO_PORT:
            for bot_number in POSSIBLE_BOTS:
                simulation(SIMULATION_DEMO_PORT, bot_number, net = best_net, example_run = True, log_run=True)
                
        fitnesses = [genome.fitness for genome_id, genome in genomes]
        
        
        self.pop_fitness_list += [{"generation":self.generation, "genomeID":genome_id, "fitness":v["fitness"]} | v["fitness_parts"] for genome_id, v in self.fitness_dict.items()]
        pd.DataFrame.from_records(self.pop_fitness_list).to_csv(f"{checkpoint_dir}/{experiment_name}.csv")

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
            

        for part in self.fitness_dict[genome_id]["fitness_parts"].keys():
            scores = [v["fitness_parts"][part] for k, v in self.fitness_dict.items()]
            tb.add_scalar(f"Avg{part}", sum(scores)/len(scores), self.generation)
            tb.add_scalar(f"Max{part}", max(scores), self.generation)
            tb.add_scalar(f"Med{part}", np.median(scores), self.generation)
            tb.add_scalar(f"Min{part}", min(scores), self.generation)
            tb.add_scalar(f"std{part}", np.std(scores), self.generation)
            tb.add_histogram(part, np.asarray(scores), self.generation)


        
        
def get_last_checkpoint(experiment_name, generation = None):
    """Find the latest checkpoint in the current directory."""
    if generation:
        return f'{checkpoint_dir}/{experiment_name} - Generation {generation}', generation
    else:
        local_dir = os.path.join(os.path.dirname(__file__), checkpoint_dir)

        checkpoints = [int(f.split('- Generation ')[-1]) for f in os.listdir(local_dir) if f.startswith(experiment_name) and ".csv" not in f]
        checkpoints = sorted(checkpoints)
        
        if not checkpoints:
            return ''
        for checkpoint in checkpoints[:-5]:
            os.remove(f'{checkpoint_dir}/{experiment_name} - Generation {checkpoint}')
        
        last_number = checkpoints[-1]
        latest_checkpoint = f'{checkpoint_dir}/{experiment_name} - Generation {last_number}'
        return latest_checkpoint, last_number 
        
def run(num_gens, num_instances, config, experiment_continuation = None):

    
    if experiment_continuation:
        checkpoint, gen = get_last_checkpoint(experiment_continuation)
        print(f'Restoring checkpoint: {checkpoint}')
        pop = neat.Checkpointer.restore_checkpoint(checkpoint, config = CONFIG) # use config = None if you don't want to update the config
        
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
    num_instances = input("Number of instances: ")
    try:
        num_instances = int(num_instances)
    except:
        num_instances = 1
        
    experiment_continuation = None  # Either like "Robobo Experiment <date> <time>" or None
    
    if experiment_continuation:
        experiment_name = experiment_continuation


    tb = SummaryWriter(f"tb_runs/{experiment_name}")

    
    run(num_gens = 5, num_instances = num_instances,  config = CONFIG, experiment_continuation = experiment_continuation)