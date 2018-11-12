import retro
import numpy as np
import cv2 
import neat
import pickle

env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')


imgarray = []

xpos_end = 0


def eval_genomes(genomes, config):


    for genome_id, genome in genomes:
        ob = env.reset()
        ac = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0
        xpos_start = 0
        
        # setup tracking of rings and time
        prev_frame_rings = 0
        prev_frame_time = 0
        prev_frame_xpos = 0

        done = False
        #cv2.namedWindow("main", cv2.WINDOW_NORMAL)

        while not done:
            
            env.render()
            frame += 1
            # Show was the Neural Network sees
            #scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)
            #scaledimg = cv2.resize(scaledimg, (iny, inx)) 
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx,iny))
            # display image
            #cv2.imshow('main', scaledimg)
            #cv2.waitKey(1) 

            imgarray = np.ndarray.flatten(ob)

            nnOutput = net.activate(imgarray)
            
            ob, rew, done, info = env.step(nnOutput)

            # START OF FITNESS CALCULATIONS # 

            # Set variables from the game ram
            xpos = info['x']
            xpos_end = info['screen_x_end']
            rings = info['rings']

            # set start x postition
            if frame == 1:
                xpos_start = xpos
                #print("Starting position: " + str(xpos_start))

            # You done fucked up going backwards bruh...
            if xpos < xpos_start:
                done = True
            elif xpos < (prev_frame_xpos - 500):
                fitness_current -= 2
                
            prev_frame_xpos = xpos

            # rings add to fitness score
            if rings > prev_frame_rings:
                fitness_current += 5
                prev_frame_rings = rings
            elif rings == prev_frame_rings:
                pass
            else:
                fitness_current -= 20
            
            prev_frame_rings += rings
            
            # Add +1 to fitness score for each movement towards the end of the level
            if xpos > xpos_max:
                fitness_current += 2
                xpos_max = xpos
            
            # End if they reach the end of the level 
            if xpos == xpos_end and xpos > 500:
                fitness_current += 100000
                done = True
            
            # Calculate addition to fitness score for any reward per frame
            fitness_current += rew
            if rew:
                fitness_current += 1
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
                
            # Set done == true    
            if done or counter == 500:
                done = True
                print(genome_id, fitness_current)
                # print("Reward: " + str(rew))
                # print("Fitness: " + str(fitness_current))
    
            genome.fitness = fitness_current
                
            
            
    

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)


p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
