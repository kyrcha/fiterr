package info.kyrcha.fiterr.rlglue.experiments;

import info.kyrcha.fiterr.Utils;

import info.kyrcha.fiterr.ne.MetaNEATEvolvable;
import info.kyrcha.fiterr.ne.MetaNEATGenome;
import info.kyrcha.fiterr.ne.esn.cmaes.CMAESESN;
import info.kyrcha.fiterr.rlglue.Platform;

import java.lang.reflect.Constructor;
import java.util.Properties;

import org.apache.commons.math.stat.StatUtils;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

import org.rlcommunity.rlglue.codec.RLGlue;

/**
 *
 * @author Kyriakos Chatzidimitriou, email: kyrcha [at] issel (dot) ee (dot) auth (dot) gr
 * 
 */
public class CMAESExperiment {
	
	// Class variables
	
	/** Logger named after the class */
	private static final Logger logger = Logger.getLogger(CMAESExperiment.class.getName());
	
	private String agentParamsFile;
	
	private String algorithm;
	
	private int runs = 25;
	
	private int generations = 100;
	
	private int episodes = 100;
	
	protected int steps = 2500;
    
    public CMAESExperiment(Properties props) {
    	runs = Integer.parseInt(props.getProperty("experiment.runs"));
    	generations = Integer.parseInt(props.getProperty("experiment.generations"));
    	episodes = Integer.parseInt(props.getProperty("experiment.episodes"));
    	steps = Integer.parseInt(props.getProperty("experiment.steps"));
    	algorithm = props.getProperty("experiment.algorithm");
    	agentParamsFile = props.getProperty("experiment.pop.params");
    }

    public void runExperiment() {
        logger.info("Experiment starts!");
        try {
	        for(int i = 1; i <= runs; i++) {
	        	logger.info("Run " + i);
	        	// Create population
				// Create the Agent
	        	Class<?> neAlg = Class.forName(algorithm);
	        	Constructor<?> ctor = neAlg.getDeclaredConstructor(String.class);
				CMAESESN population = (CMAESESN)ctor.newInstance(agentParamsFile);
	        	for(int gen = 1; gen <= generations; gen++) {
					double championFitness = -1000000;
	        		for(int pop = 0; pop < population.getPopulationSize(); pop++) {
	        			population.setFitness(pop, 0);
	        			population.setEpisodes(pop, 0);
	        			// Evaluate the population
	        			for(int episode = 0; episode < episodes * population.getPopulationSize(); episode++) {
	        				// Select random genome to be evaluated
	        				int selection = (episode % population.getPopulationSize());
	        				String genome = population.getGenomeString(selection);
	        				RLGlue.RL_agent_message("disable-learning");
	        				RLGlue.RL_agent_message(genome);
	        				RLGlue.RL_init();
	        				RLGlue.RL_episode(steps);
	        				double totalReward = RLGlue.RL_return();
	        				population.addFitness(selection, rewardToFitness(totalReward));
	        				population.incrEpisodes(selection);
	        				RLGlue.RL_cleanup();
	        			}
	        		}
	        		// Calculate fitness
	        		double[] populationPerformance = new double[population.getPopulationSize()];
	        		for(int pop = 0; pop < population.getPopulationSize(); pop++) {
	        			if(population.getEpisodes(pop) > 0) {
	        				population.setFitness(pop, population.getFitness(pop) / population.getEpisodes(pop));
	        			} else {
	        				population.setFitness(pop, 0);
	        			}
	        			populationPerformance[pop] = fitnessToReward(population.getFitness(pop));
	        			if(championFitness < populationPerformance[pop]) {
	        				championFitness = populationPerformance[pop];
	        			}
	        		}
	        		logger.info("Generation Champion: " + gen + " " + StatUtils.max(populationPerformance));
	        		logger.info("Avg Performance: " + gen + " " + StatUtils.mean(populationPerformance));
	        		population.evolveNextGeneration();
	        		logger.info("===*** " + gen + " ***===");
	        	}
	        }
        } catch(Exception e) {
        	e.printStackTrace();
        }
    }
    
    protected double rewardToFitness(double reward) {
    	return steps + reward;
    }
    
    protected double fitnessToReward(double fitness) {
    	return fitness - steps;
    }

    public static void main(String[] args) {
    	try {
			// Must have a parameter file
			if (args.length < 1) {
				Utils.fatalError("You must specify a parameter file");
			}
			// Configure the logger
			String fileName = args[0];
			Properties props = Utils.loadProperties(fileName);
			PropertyConfigurator.configure(props.getProperty(Platform.P_LOG));
			CMAESExperiment theExperiment = new CMAESExperiment(props);
	        theExperiment.runExperiment();
    	} catch (Exception e) {
			logger.error(e);
			e.printStackTrace();
		}
    }
}
