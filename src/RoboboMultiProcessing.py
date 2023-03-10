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
    
    if log_run:
        pd.DataFrame.from_records(scores).to_csv(f"{log_dir}/{experiment_name}{'_' + str(genomeID) if genomeID else '_example'}.csv", index = False)
        
    if example_run:
        return


        
    # Returning the found fitness (and the fitness parts) to the evaluate function
    print(f"Genome: {genomeID},\tfitness: {fitness:.2f},\tdistToRed: {distToRed:.2f},\tdistToGreen: {distToGreen:.2f}")
    if genomeID in fitness_dict.keys():
        fitness_dict[genomeID]["fitness"] += fitness/len(POSSIBLE_BOTS)
        fitness_dict[genomeID]["fitness_parts"] = {"t":t, "distToRed": distToRed, "distToGreen": distToGreen, "BDF": BaseDetectFood}
    else:
        fitness_dict[genomeID] = {"fitness":fitness/len(POSSIBLE_BOTS),
                                  "fitness_parts": {"t":t, "distToRed": distToRed, "distToGreen": distToGreen, "BDF": BaseDetectFood}}
        



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
        
    experiment_continuation = "Robobo Experiment 2023-02-01 16;55"  # Either like "Robobo Experiment <date> <time>" or None
    
    if experiment_continuation:
        experiment_name = experiment_continuation


    tb = SummaryWriter(f"tb_runs/{experiment_name}")

    
    run(num_gens = 5, num_instances = num_instances,  config = CONFIG, experiment_continuation = experiment_continuation)