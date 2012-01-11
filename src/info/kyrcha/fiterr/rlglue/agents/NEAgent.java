package info.kyrcha.fiterr.rlglue.agents;

import info.kyrcha.fiterr.Function;
import info.kyrcha.fiterr.Utils;
import info.kyrcha.fiterr.ne.Network;
import info.kyrcha.fiterr.rlglue.environments.mountaincar.MountainCarState2D;

import java.util.Properties;

import org.apache.log4j.Logger;

import org.rlcommunity.rlglue.codec.AgentInterface;
import org.rlcommunity.rlglue.codec.types.Action;
import org.rlcommunity.rlglue.codec.types.Observation;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpec;
import org.rlcommunity.rlglue.codec.taskspec.ranges.IntRange;
import org.rlcommunity.rlglue.codec.taskspec.ranges.DoubleRange;

import Jama.Matrix;

/**
 * Agent class is capable for implementing evolution and learning adaptation in RL tasks.
 * 
 * @author Kyriakos Chatzidimitriou, email: kyrcha [at] issel (dot) ee (dot) auth (dot) gr
 */
public class NEAgent implements AgentInterface {
	
	/** Logger named after the class */
	private static final Logger logger = Logger.getLogger(NEAgent.class.getName());
    
    private Action lastAction;
    
    private Observation lastObservation;
    
    private Matrix previousFeatureVector;
    
    private Matrix gradient;
    
    private double lastOutput;
    
    private int lastIndex;
    
    protected Network network;
    
    private double gamma = 1.0;
    
    private double lambda = 0.0;
    
	protected double epsilon = 0;
	
	protected boolean accumulatedTraces = true;
	
	protected boolean enableOutput = false;
    
    private double learningRate = 0.0;
    
    private boolean learning = false;
    
    private boolean lstd = false;
    
    private double[] lowerBounds;
    
    private double[] upperBounds;
    
    public NEAgent(String fileName) {
		Properties props = Utils.loadProperties(fileName);	
    }

    public void agent_init(String taskSpecification) {
		TaskSpec theTaskSpec = new TaskSpec(taskSpecification);
		logger.trace("NN agent parsed the task spec.");
		logger.trace("Observations have "+ theTaskSpec.getNumContinuousObsDims() + " continuous dimensions");
		logger.trace("Actions have "+ theTaskSpec.getNumDiscreteActionDims() + " integer dimensions");
		lowerBounds = new double[theTaskSpec.getNumContinuousObsDims() + theTaskSpec.getNumDiscreteObsDims()];
		upperBounds = new double[theTaskSpec.getNumContinuousObsDims() + theTaskSpec.getNumDiscreteObsDims()];
		int counter = 0;
		for(int i = 0; i < theTaskSpec.getNumContinuousObsDims(); i++) {
			DoubleRange theObsRange = theTaskSpec.getContinuousObservationRange(i);
			lowerBounds[counter] = theObsRange.getMin();
			upperBounds[counter] = theObsRange.getMax();
			counter++;
		}
		for(int i = 0; i < theTaskSpec.getNumDiscreteObsDims(); i++) {
			IntRange theObsRange = theTaskSpec.getDiscreteObservationRange(i);
			lowerBounds[counter] = theObsRange.getMin();
			upperBounds[counter] = theObsRange.getMax();
			counter++;
		}
		for(int i = 0; i < theTaskSpec.getNumDiscreteActionDims(); i++) {
			IntRange theActRange = theTaskSpec.getDiscreteActionRange(0);
			logger.trace("Action range is: " + theActRange.getMin() + " to " + theActRange.getMax());
		}
		DoubleRange theRewardRange = theTaskSpec.getRewardRange();
		logger.trace("Reward range is: " + theRewardRange.getMin() + " to " + theRewardRange.getMax());
    }

    // TODO add the usage of network here
    public Action agent_start(Observation observation) {
    	double[] input = encodeInput(observation);
        // Flush the network (old values of internal and output units 
        network.flush();
        // Activate the network
    	double[] qvalues = network.activate(input);
    	// Create a structure to hold 1 integer action and set it
        Action returnAction = new Action(1, 0, 0);
    	returnAction.intArray[0] = egreedy(qvalues, observation);
        lastAction = returnAction.duplicate();
        lastObservation = observation.duplicate();
        lastIndex = returnAction.intArray[0];
        lastOutput = qvalues[lastIndex];
        previousFeatureVector = network.getFeatureVector();
        return returnAction;
    }

    // TODO add the usage of network here
    public Action agent_step(double reward, Observation observation) {
    	double[] input = encodeInput(observation);
    	for(int i = 0; i < input.length; i++) {
//    		System.out.print(input[i] + " - ");
    	}
//    	System.out.print("\n");
    	if(learning) {
	    	// Update eligibility traces
	    	network.updateTraces(gamma, lambda);
	    	// Calculate the gradient
	    	Function netOutFunct = network.getOutputFunction();
	    	if(netOutFunct == Function.IDENTITY) {
	    		gradient = previousFeatureVector.copy();
	    	} else if(netOutFunct == Function.SIGMOID) {
	    		gradient = previousFeatureVector.times(lastOutput * (1-lastOutput));
	    	}
	    	network.updateTraces(accumulatedTraces, gradient, lastIndex);
    	}
    	double[] qvalues = network.activate(input);
    	// Create a structure to hold 1 integer action and set it
        Action returnAction = new Action(1, 0, 0);
    	returnAction.intArray[0] = egreedy(qvalues, observation);
        lastAction = returnAction.duplicate();
        lastObservation = observation.duplicate();
        if(learning) {
        	double delta = reward - lastOutput;
        	delta += gamma * qvalues[returnAction.intArray[0]];
        	if(lstd) {
        		network.updateLSTDMatrices(previousFeatureVector, network.getFeatureVector().copy(), reward);
        	} else {
        		network.GDTDLearning(learningRate, delta);
        	}
        }
        lastIndex = returnAction.intArray[0];
        lastOutput = qvalues[lastIndex];
        previousFeatureVector = network.getFeatureVector();
        return returnAction;
    }

    public void agent_end(double reward) {
    	if(learning) {
	    	// Update eligibility traces
	    	network.updateTraces(gamma, lambda);
	    	// Calculate the gradient
	    	Function netOutFunct = network.getOutputFunction();
	    	if(netOutFunct == Function.IDENTITY) {
	    		gradient = previousFeatureVector.copy();
	    	} else if(netOutFunct == Function.SIGMOID) {
	    		gradient = previousFeatureVector.times(lastOutput * (1-lastOutput));
	    	}
	    	network.updateTraces(accumulatedTraces, gradient, lastIndex);
	    	double delta = reward - lastOutput;
	    	if(lstd) {
	    		network.updateLSTDMatrices(previousFeatureVector, network.getFeatureVector().copy(), reward);
	    	} else {
	    		network.GDTDLearning(learningRate, delta);
	    	}
    	}
    }

    public void agent_cleanup() {
        lastAction=null;
        lastObservation=null;
    }

    public String agent_message(String message) {
    	if(message.equalsIgnoreCase("enable-output")) {
    		enableOutput = true;
    		return null;
    	} else if(message.equalsIgnoreCase("disable-output")) {
    		enableOutput = false;
    		return null;
    	}else if(message.equalsIgnoreCase("disable-learning")) {
    		learning = false;
    		return null;
    	} else if(message.equalsIgnoreCase("update-LSTD")) {
    		network.updateWeights();
    		return null;
    	} else if(message.startsWith("enable-learning")) {
    		String[] tokens = message.split("\\:");
    		learning = true;
    		learningRate = Double.parseDouble(tokens[1]);
    		return null;
    	} else if(message.startsWith("enable-exploration")) {
    		String[] tokens = message.split("\\:");
    		epsilon = Double.parseDouble(tokens[1]);
    		return null;
    	} else if(message.equalsIgnoreCase("disable-exploration")) {
    		epsilon = 0;
    		return null;
    	} else if(message.equalsIgnoreCase("get-learned-weights")) {
    		int capacity =  (network.getL() * (network.getK() + network.getN())) * 10;
    		StringBuilder sbnet = new StringBuilder(capacity);
    		for(int i = 0; i < network.getL(); i++) {
    			for(int j = 0; j < (network.getK() + network.getN()); j++) {
    				sbnet.append(network.getWout(i, j));
    				sbnet.append(';');
    			}
    		}
    		sbnet.trimToSize();
    		return sbnet.toString();
    	} else {
	    	// Split the tokens
	    	String[] tokens = message.split("\\/");
	    	// Parse the tokens
	    	Function internal = Enum.valueOf(Function.class, tokens[0]); 
	    	Function output = Enum.valueOf(Function.class, tokens[1]);
	    	int K = Integer.parseInt(tokens[2]); 
	    	int N = Integer.parseInt(tokens[3]);
	    	int L = Integer.parseInt(tokens[4]);
	    	network = new Network(K, N, L, internal, output);
	    	// Parse the matrix tokens
	    	double[][] wout = parseMatrices(tokens[5], L, (K + N + L));
	    	network.setWout(wout);
	    	if(N > 0) {
	    		double[][] win = parseMatrices(tokens[6], N, K);
	    		network.setWin(win);
	    		double[][] w = parseMatrices(tokens[7], N, N);
	    		network.setW(w);
	    		double[][] wback = parseMatrices(tokens[8], N, L);
	    		network.setWback(wback);
	    	}
	    	return null;
    	}
    }
    
    private double[][] parseMatrices(String token, int m, int n) {
    	double[][] matrix = new double[m][n];
    	try {
    		String[] tokens = token.split("\\;");
    		for(int i = 0; i < m; i++) {
    			for(int j = 0; j < n; j++) {
    				matrix[i][j] = Double.parseDouble(tokens[i * n + j]);
    			}
    		}
    	} catch (Exception e) {
    		System.out.println(m + " " + n + " " + token);
    	}
    	return matrix;
    }
    
    protected double[] encodeInput(Observation obsv) {
    	int length = obsv.getNumDoubles() + obsv.getNumInts();
    	if(enableOutput) {
    		for(int i = 0; i < obsv.getNumDoubles(); i++) {
    			System.out.print(obsv.getDouble(i) + " ");
    		}
    		for(int i = 0; i < obsv.getNumInts(); i++) {
    			System.out.print(obsv.getDouble(i) + " ");
    		}
    		System.out.print("\n");
    	}
    	// remove bias
//    	double[] encodedInput = new double[length + 1];
//		encodedInput[0] = 1;
//    	int counter = 1;
    	double[] encodedInput = new double[length];
		int counter = 0;
    	for(int i = 0; i < obsv.getNumDoubles(); i++) {
    		if(network.getHiddenLayerFunction().compareTo(Function.TANH) == 0) {
    			encodedInput[counter] = 2 * ((obsv.getDouble(i) - lowerBounds[i]) / (upperBounds[i] - lowerBounds[i])) - 1;
//    			encodedInput[counter] = obsv.getDouble(i);
//    			System.out.print(encodedInput[counter] + " ");
    		} else {
    			encodedInput[counter] = (obsv.getDouble(i) - lowerBounds[i]) / (upperBounds[i] - lowerBounds[i]);
    		}
    		counter++;
    	}
    	for(int i = 0; i < obsv.getNumInts(); i++) {
    		if(network.getHiddenLayerFunction().compareTo(Function.TANH) == 0) {
    			encodedInput[counter] = 2 * ((obsv.getInt(i) - lowerBounds[i]) / (upperBounds[i] - lowerBounds[i])) - 1;
//    			encodedInput[counter] = obsv.getInt(i);
//    			System.out.print(encodedInput[counter] + " ");
    		} else {
    			encodedInput[counter] = (obsv.getInt(i) - lowerBounds[i]) / (upperBounds[i] - lowerBounds[i]);
    		}
    		counter++;
    	}
//    	System.out.print("\n");
    	return encodedInput;
    }
    
	protected int egreedy(double[] values, Observation observation) {
    	if(Utils.rand.nextDouble() < epsilon) {
    		return Utils.rand.nextInt(values.length);
    	} else {
    		double maxValue = -100000000d;
    		int index = -1;
    		for(int i = 0; i < values.length; i++) {
//    			System.out.println(i + " " + values[i]);
    			if(maxValue < values[i]) {
    				maxValue = values[i];
    				index = i;
    			}
    		}
    		return index;
    	}
	}
    
//    private double[] encodeInput(double x, double x_dot, double theta1, double theta_dot1, double theta2, double theta_dot2) {
//    	double[] input = new double[7];
//    	// Bias
//    	input[0] = 1;
//    	// Normalize [-1, 1]
//    	input[1] = (2 * (x + 2.4) / 4.8) - 1;
//    	input[2] = (2 * (x_dot + 0.75) / 1.5) - 1;
//    	input[3] = (2 * (theta1 + 0.2094384) / 0.41) - 1;
//    	input[4] = (2 * (theta_dot1 + 1.0) / 2.0) - 1;
//    	input[5] = (2 * (theta2 + 0.2094384) / 0.41) - 1;
//    	input[6] = (2 * (theta_dot2 + 1.0) / 2.0) - 1;
//    	return input;
//    }
	
}
