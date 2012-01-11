package info.kyrcha.fiterr.rlglue.experiments;

import info.kyrcha.fiterr.Utils;

import info.kyrcha.fiterr.ne.MetaNEATEvolvable;
import info.kyrcha.fiterr.ne.MetaNEATGenome;
import info.kyrcha.fiterr.rlglue.Platform;

import java.lang.reflect.Constructor;
import java.util.Properties;

import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

import org.rlcommunity.rlglue.codec.RLGlue;

/**
 *
 * @author Kyriakos Chatzidimitriou, email: kyrcha [at] issel (dot) ee (dot) auth (dot) gr
 * 
 */
public class NEARPBExperiment {
	
	// Class variables
	
	/** Logger named after the class */
	private static final Logger logger = Logger.getLogger(NEARPBExperiment.class.getName());
	
	private String agentParamsFile;
	
	private String algorithm;
	
	protected int runs = 25;
	
	protected int generations = 100;
	
	protected int episodes = 100;
	
	protected int steps = 2500;
	
	protected boolean enableLearning = true;
    
    public NEARPBExperiment(Properties props) {
    	runs = Integer.parseInt(props.getProperty("experiment.runs"));
    	generations = Integer.parseInt(props.getProperty("experiment.generations"));
    	episodes = Integer.parseInt(props.getProperty("experiment.episodes"));
    	steps = Integer.parseInt(props.getProperty("experiment.steps"));
    	algorithm = props.getProperty("experiment.algorithm");
    	agentParamsFile = props.getProperty("experiment.pop.params");
    	enableLearning = Boolean.parseBoolean(props.getProperty("experiment.learning"));
    }

    public void runExperiment() {
        logger.info("Experiment starts!");
        try {
        	int sum = 0;
	        for(int i = 1; i <= runs; i++) {
	        	logger.info("Run " + i);
	        	// Create population
				// Create the Agent
	        	Class<?> neAlg = Class.forName(algorithm);
	        	Constructor<?> ctor = neAlg.getDeclaredConstructor(String.class);
				MetaNEATEvolvable population = (MetaNEATEvolvable)ctor.newInstance(agentParamsFile);
				boolean found = false; 
	        	int evals = 0;
	        	for(int gen = 1; gen <= generations; gen++) {
	        		int maxGenSteps = -1;
	        		for(int pop = 0; pop < population.getPopulationSize(); pop++) {
	        			MetaNEATGenome genome = population.getGenome(pop);
	        			genome.setFitness(0);
	        			genome.setEpisodes(0);
	        		}
	        		if(enableLearning) {
	        			RLGlue.RL_agent_message("enable-learning:0.00001");
	        			RLGlue.RL_agent_message("enable-exploration:0.01");
	        		}
	        		// Evaluate the population
	        		for(int episode = 0; episode < episodes * population.getPopulationSize(); episode++) {
		        		// Select random genome to be evaluated
//		        		int selection = Utils.rand.nextInt(population.getPopulationSize());
		        		int selection = (episode % population.getPopulationSize());
		        		MetaNEATGenome genome = population.getGenome(selection);
		        		evals++;
		        		int currSteps = runEpisode(genome);
		        		// ask for data for fitness
		        		if(maxGenSteps < currSteps) maxGenSteps = currSteps;
		        		if(currSteps >= 100000) {
		        			sum++;
		                	found = true;
		                	logger.info("Found solution! Evaluations: " + evals);
		                	RLGlue.RL_agent_message("enable-output");
		                	runEpisode(genome);
		                	RLGlue.RL_agent_message("disable-output");
		                	System.out.println(genome);
		                	break;
		                }
		        		String weights = RLGlue.RL_agent_message("get-learned-weights");
		        		genome.message(weights);
	        		}
	                if(found) {
	                	break;
	                }
	                System.out.println(maxGenSteps);
	        		population.evolveNextGeneration();
	        		logger.info("===*** " + gen + " ***===");
	        	}
	        }
	        logger.info("Solutions Found: " + sum);
        } catch(Exception e) {
        	e.printStackTrace();
        }
    }
    
    private int runEpisode(MetaNEATGenome genome) {
    	RLGlue.RL_agent_message(genome.toPhenotype().toString());
		RLGlue.RL_init();
		RLGlue.RL_episode(steps);
		RLGlue.RL_return();
		int numOfSteps = RLGlue.RL_num_steps();
		genome.addFitness(numOfSteps);
		genome.incrEpisodes();
		RLGlue.RL_cleanup();
		return numOfSteps;
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
			NEARPBExperiment theExperiment = new NEARPBExperiment(props);
	        theExperiment.runExperiment();
    	} catch (Exception e) {
			logger.error(e);
			e.printStackTrace();
		}
    }
}
