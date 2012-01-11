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
public class NEARDPBGenExperiment {
	
	// Class variables
	
	/** Logger named after the class */
	private static final Logger logger = Logger.getLogger(NEARDPBGenExperiment.class.getName());
	
	private String agentParamsFile;
	
	private String algorithm;
	
	protected int runs = 25;
	
	protected int generations = 100;
	
	protected int episodes = 1;
	
	protected int steps1 = 1000;
	protected int steps2 = 100000;
	
	protected boolean enableLearning = true;
    
    public NEARDPBGenExperiment(Properties props) {
    	runs = Integer.parseInt(props.getProperty("experiment.runs"));
    	generations = Integer.parseInt(props.getProperty("experiment.generations"));
    	episodes = Integer.parseInt(props.getProperty("experiment.episodes"));
    	steps1 = Integer.parseInt(props.getProperty("experiment.steps1"));
    	steps2 = Integer.parseInt(props.getProperty("experiment.steps2"));
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
	        		double maxFitness = -1;
	        		int champIndex = -1;
	        		for(int pop = 0; pop < population.getPopulationSize(); pop++) {
	        			MetaNEATGenome genome = population.getGenome(pop);
	        			genome.setFitness(0);
	        			genome.setEpisodes(0);
	        			RLGlue.RL_agent_message("enable-trace");
	        			RLGlue.RL_agent_message(genome.toPhenotype().toString());
	        			RLGlue.RL_init();
		        		RLGlue.RL_episode(steps1);
		        		RLGlue.RL_return();
		        		int currSteps = RLGlue.RL_num_steps();
		        		RLGlue.RL_cleanup();
	        			String traceString = RLGlue.RL_agent_message("get-trace");
	        			double[][] trace = parse(traceString);
	        			double fitness1 = (currSteps/1000.0);
	        			double den = 0;
	        			double fitness2 = 0;
	        			if(currSteps >= 100) {
	        				for(int k = currSteps-100; k < currSteps; k++) {
	        					for(int j = 0; j < 4; j++) {
	        						den += Math.abs(trace[k][j]);
	        					}
	        				}
	        				fitness2 += 0.75/den;
	        			}
	        			double fitness = 0.1 * fitness1 + 0.9 * fitness2;
//	        			System.out.println("Steps: " + currSteps + " Pop: " + pop + " Fitness: " + fitness);
	        			genome.addFitness(fitness);
	        			genome.incrEpisodes();
	        			evals++;
	        			// ask for data for fitness
		        		if(maxFitness < fitness) {
		        			maxFitness = fitness;
		        			champIndex = pop;
		        		}
	        		}
	        		System.out.print("Champ: " + maxFitness + " ");
	        		// Test champion
	        		MetaNEATGenome genome = population.getGenome(champIndex);
	            	RLGlue.RL_agent_message(genome.toPhenotype().toString());
	            	RLGlue.RL_agent_message("disable-trace");
	        		RLGlue.RL_init();
	        		RLGlue.RL_episode(steps2);
	        		RLGlue.RL_return();
	        		int currSteps = RLGlue.RL_num_steps();
	        		RLGlue.RL_cleanup();
	        		System.out.println("Steps: " + currSteps + "\n");
	        		if(currSteps >= steps2) {
//                		System.out.println("Solution Found! Evaluations: " + evals);
//                		sum++;
//                		found = true;
////                		RLGlue.RL_agent_message("enable-output");
//                		RLGlue.RL_agent_message(genome.toPhenotype().toString());
//                		RLGlue.RL_init();
//                		RLGlue.RL_episode(steps2);
//                		RLGlue.RL_return();
//                		int numOfSteps = RLGlue.RL_num_steps();
//                		RLGlue.RL_cleanup();
////                		RLGlue.RL_agent_message("disable-output");
//                		System.out.println(genome);
//                		break;
	                	logger.info("Reached 1000 steps goal!");
	                	int generalizationSols = 0;
	                	int countExperiments = 0;
	                	// generalization test (200 out of 625 solutions for 1000 time steps)
	                	double[] x = {0.05, 0.25, 0.5, 0.75, 0.95};
	                	double[] xdot = {0.05, 0.25, 0.5, 0.75, 0.95};
	                	double[] theta1 = {0.05, 0.25, 0.5, 0.75, 0.95};
	                	double[] theta1dot = {0.05, 0.25, 0.5, 0.75, 0.95};
	                	for(int xi = 0; xi < 5; xi++) {
	                		for(int xdoti = 0; xdoti < 5; xdoti++) {
	                			for(int theta1i = 0; theta1i < 5; theta1i++) {
	                				for(int theta1doti = 0; theta1doti < 5; theta1doti++) {
	                					double xVal = (x[xi] * 4.8) - 2.4;
	                					double xDotVal = (xdot[xdoti] * 2) - 1;
	                					double theta1Val = (theta1[theta1i] * 0.4) - 0.2;
	                					double theta1DotVal = (theta1dot[theta1doti] * 3) - 1.5;
	                					String message = "custom:" + xVal + ";" + xDotVal + ";" + theta1Val + ";" + theta1DotVal;
	                					RLGlue.RL_env_message(message);
	                					RLGlue.RL_agent_message(genome.toPhenotype().toString());
	                					RLGlue.RL_init();
	        	                		RLGlue.RL_episode(steps1);
	        	                		RLGlue.RL_return();
	        	                		currSteps = RLGlue.RL_num_steps();
	        	                		RLGlue.RL_cleanup();
	        	                		countExperiments++;
//	        	                		System.out.println("Exp Num: " + countExperiments);
	                					if(currSteps >= steps1) {
	                						generalizationSols++;
//	                						System.out.println("Sol Found: " + generalizationSols);
	                					}
	                				}
	                			}
	                		}
	                	}
	                	System.out.println("Generalization Sols found: " + generalizationSols);
	                	if(generalizationSols >= 200) {
	                		System.out.println("Solution Found! Evaluations: " + evals);
	                		sum++;
	                		found = true;
	                		RLGlue.RL_agent_message("enable-output");
	                		RLGlue.RL_agent_message(genome.toPhenotype().toString());
	                		RLGlue.RL_init();
	                		RLGlue.RL_episode(steps2);
	                		RLGlue.RL_return();
	                		int numOfSteps = RLGlue.RL_num_steps();
	                		RLGlue.RL_cleanup();
	                		RLGlue.RL_agent_message("disable-output");
	                		System.out.println(genome);
	                		break;
	                	}
	                }
	                if(found) {
	                	break;
	                }
	        		population.evolveNextGeneration();
	        		logger.info("===*** " + gen + " ***===");
	        	}
	        }
	        logger.info("Solutions Found: " + sum);
        } catch(Exception e) {
        	e.printStackTrace();
        }
    }
    
    private double[][] parse(String s) {
    	String[] tokens = s.split("\\;");
    	double[][] trace = new double[1000][4];
    	int counter = 0;
    	for(int i = 0; i < 1000; i++) {
    		for(int j = 0; j < 4; j++) {
    			trace[i][j] = Double.parseDouble(tokens[counter]);
    			counter++;
    		}
    	}
		return trace;
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
			NEARDPBGenExperiment theExperiment = new NEARDPBGenExperiment(props);
	        theExperiment.runExperiment();
    	} catch (Exception e) {
			logger.error(e);
			e.printStackTrace();
		}
    }
}
