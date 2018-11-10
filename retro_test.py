import neat 
import retro
import pickle
import cv2
import numpy as np
#cp = neat.Checkpointer.restore_checkpoint()

imgarray = []

xpos_end = 0

env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')

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
        
        prev_frame_rings = 0

        done = False
        #cv2.namedWindow("main", cv2.WINDOW_NORMAL)

        while not done:
            
            env.render()
            frame += 1
            #scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)
            #scaledimg = cv2.resize(scaledimg, (iny, inx)) 
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx,iny))
            #cv2.imshow('main', scaledimg)
            #cv2.waitKey(1) 

            imgarray = np.ndarray.flatten(ob)

            nnOutput = net.activate(imgarray)
            
            ob, rew, done, info = env.step(nnOutput)
            
            
            xpos = info['x']
            xpos_end = info['screen_x_end']
                            
            rings = info['rings']
            
            if rings > prev_frame_rings:
                fitness_current += 100
                previous_frame_rings += rings

                print("Rings: " + str(rings))
                print("Total Rings: " + str(prev_frame_rings))
            
            prev_frame_rings += rings
            
            if xpos > xpos_max:
                fitness_current += 1
                xpos_max = xpos
            
            if xpos == xpos_end and xpos > 500:
                # fitness_current += 100000
                done = True
            
            fitness_current += rew
            if rew:
            	fitness_current += 500
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
                
            if done or counter == 500:
                done = True
                print(genome_id, fitness_current)
                
            genome.fitness = fitness_current


def main():
	p = neat.Checkpointer().restore_checkpoint('top_genomes')
	env.reset()
	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,'config-feedforward')
	# p = neat.Population(config)
	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)
	p.run(eval_genomes, 2)

if __name__ == '__main__':
	main()
