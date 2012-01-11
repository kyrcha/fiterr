package info.kyrcha.fiterr.rlglue.experiments;

import info.kyrcha.fiterr.Utils;

import info.kyrcha.fiterr.ne.MetaNEATEvolvable;
import info.kyrcha.fiterr.ne.MetaNEATGenome;
import info.kyrcha.fiterr.ne.esn.near.NEARGenome;
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
public class NEARExperiment {
	
	// Class variables
	
	/** Logger named after the class */
//	private static final Logger logger = Logger.getLogger(NEARExperiment.class.getName());
	
	private String agentParamsFile;
	
	private String algorithm;
	
	protected int runs = 25;
	
	protected int generations = 100;
	
	protected int episodes = 100;
	
	protected int steps = 2500;
	
	protected boolean enableLearning = true;
    
    public NEARExperiment(Properties props) {
    	runs = Integer.parseInt(props.getProperty("experiment.runs"));
    	generations = Integer.parseInt(props.getProperty("experiment.generations"));
    	episodes = Integer.parseInt(props.getProperty("experiment.episodes"));
    	steps = Integer.parseInt(props.getProperty("experiment.steps"));
    	algorithm = props.getProperty("experiment.algorithm");
    	agentParamsFile = props.getProperty("experiment.pop.params");
    	enableLearning = Boolean.parseBoolean(props.getProperty("experiment.learning"));
    }

    public void runExperiment() {
//        logger.info("Experiment starts!");
        System.out.println("Experiment starts!");
        try {
	        for(int i = 1; i <= runs; i++) {
//	        	logger.info("Run " + i);
	        	System.out.println("Run " + i);
	        	// Create population
				// Create the Agent
	        	Class<?> neAlg = Class.forName(algorithm);
	        	Constructor<?> ctor = neAlg.getDeclaredConstructor(String.class);
				MetaNEATEvolvable population = (MetaNEATEvolvable)ctor.newInstance(agentParamsFile);
        		MetaNEATGenome championOfChampions = null;
				double championOfChampionsFitness = -1000000;
	        	for(int gen = 1; gen <= generations; gen++) {
	        		MetaNEATGenome champion = null;
					double championFitness = -1000000;
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
		        		runEpisode(genome);
		        		String weights = RLGlue.RL_agent_message("get-learned-weights");
		        		genome.message(weights);
		        		
	        		}
	        		double Dmean = 0;
	        		double Nmean = 0;
	        		double rhomean = 0;
	        		// Calculate fitness
	        		double[] populationPerformance = new double[population.getPopulationSize()];
	        		for(int pop = 0; pop < population.getPopulationSize(); pop++) {
	        			MetaNEATGenome genome = population.getGenome(pop);
	        			if(genome.getEpisodes() > 0) {
	        				genome.setFitness(genome.getFitness() / genome.getEpisodes());
	        			} else {
	        				genome.setFitness(0);
	        			}
	        			Dmean += ((NEARGenome)genome).getSparseness();
	        			Nmean += ((NEARGenome)genome).getNInternalUnits();
	        			rhomean += ((NEARGenome)genome).getSpectralRadius();
	        			populationPerformance[pop] = fitnessToReward(genome.getFitness());
	        			if(championFitness < populationPerformance[pop]) {
	        				championFitness = populationPerformance[pop];
	        				champion = genome.clone();
	        			}
	        		}
	        		System.out.println("Mean N: " + gen + " " + (Nmean/population.getPopulationSize()));
	        		System.out.println("Mean D: " + gen + " " + (Dmean/population.getPopulationSize()));
	        		System.out.println("Mean rho: " + gen + " " + (rhomean/population.getPopulationSize()));
	        		RLGlue.RL_agent_message("disable-learning");
	        		RLGlue.RL_agent_message("disable-exploration");
	        		double championGeneralizationPerf = generalizationPerformance(1000, champion);
//	        		logger.info("Generation Champion: " + gen + " " + StatUtils.max(populationPerformance));
//	        		logger.info("Generalization Performance: " + gen + " " + championGeneralizationPerf);
//	        		logger.info("Avg Performance: " + gen + " " + StatUtils.mean(populationPerformance));
	        		System.out.println("Generation Champion: " + gen + " " + StatUtils.max(populationPerformance));
	        		System.out.println("Generalization Performance: " + gen + " " + championGeneralizationPerf);
	        		System.out.println("Avg Performance: " + gen + " " + StatUtils.mean(populationPerformance));
	        		System.out.println(champion);
        			if(championOfChampionsFitness < championGeneralizationPerf) {
        				championOfChampionsFitness = championGeneralizationPerf;
        				championOfChampions = champion.clone();
        			}
	        		population.evolveNextGeneration();
//	        		logger.info("===*** " + gen + " ***===");
	        		System.out.println("===*** " + gen + " ***===");
	        	}
	        	RLGlue.RL_agent_message("disable-learning");
	        	RLGlue.RL_agent_message("disable-exploration");
//	        	logger.info("Champion Of Generations Performace: " + championOfChampionsFitness);
	        	System.out.println("Champion Of Generations Performace: " + championOfChampionsFitness);
	        	double cocGenPerf = generalizationPerformance(1000, championOfChampions);
//	        	logger.info("Champion Of Generations Generalization Performace: " + cocGenPerf);
//	        	logger.info(championOfChampions);
	        	System.out.println("Champion Of Generations Generalization Performace: " + cocGenPerf);
	        	System.out.print("ChampNet: ");
	        	System.out.println(championOfChampions);
	        }
        } catch(Exception e) {
        	e.printStackTrace();
        }
    }
    
    protected void runEpisode(MetaNEATGenome genome) {
    	RLGlue.RL_agent_message(genome.toPhenotype().toString());
		RLGlue.RL_init();
		RLGlue.RL_episode(steps);
		double totalReward = RLGlue.RL_return();
		genome.addFitness(rewardToFitness(totalReward));
		genome.incrEpisodes();
		RLGlue.RL_cleanup();
    }
    
    protected double rewardToFitness(double reward) {
    	return steps + reward;
    }
    
    protected double fitnessToReward(double fitness) {
    	return fitness - steps;
    }
    
    protected double generalizationPerformance(int episodes, MetaNEATGenome genome) {
    	genome.setFitness(0);
    	genome.setEpisodes(0);
    	for(int e = 0; e < episodes; e++) {
    		runEpisode(genome);
    	}
    	genome.setFitness(genome.getFitness() / genome.getEpisodes());
    	return fitnessToReward(genome.getFitness());
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
//			PropertyConfigurator.configure(props.getProperty(Platform.P_LOG));
			NEARExperiment theExperiment = new NEARExperiment(props);
	        theExperiment.runExperiment();
    	} catch (Exception e) {
//			logger.error(e);
			e.printStackTrace();
		}
    }
}
