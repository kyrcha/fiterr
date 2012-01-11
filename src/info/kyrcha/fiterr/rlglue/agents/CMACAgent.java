package info.kyrcha.fiterr.rlglue.agents;

import info.kyrcha.fiterr.Utils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Properties;
import java.util.Scanner;

import org.apache.log4j.Logger;

import org.rlcommunity.rlglue.codec.AgentInterface;
import org.rlcommunity.rlglue.codec.types.Action;
import org.rlcommunity.rlglue.codec.types.Observation;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpec;
import org.rlcommunity.rlglue.codec.taskspec.ranges.IntRange;
import org.rlcommunity.rlglue.codec.taskspec.ranges.DoubleRange;

/**
 * Agent class that uses CMAC function approximation with either SARSA or Q-Learning
 * 
 * @author Kyriakos Chatzidimitriou, email: kyrcha [at] issel (dot) ee (dot) auth (dot) gr
 */
public class CMACAgent implements AgentInterface {
	
	private static final int REPLACING = 1;
	
	private static final int ACCUMULATING = 2;
	
	private static final int SARSA = 1;
	
	private static final int QLEARNING = 2;
	
	/** Logger named after the class */
	private static final Logger logger = Logger.getLogger(CMACAgent.class.getName());
    
    private Action lastAction;
    
    private Point lastState;
    
    private int numberOfTilings = 10;
    
    private int tilesPerDimension = 9;
    
    private int numberOfDimensions;
    
    private int numberOfActions;
    
    private double lambda = 0.95;
    
    private double gamma = 1.0;
    
	private double epsilon = 0.0;
    
    private double alpha = 0.05 * (0.1/numberOfTilings);
    
    private double offset = 0.1;
    
    private int typeOfTrace = ACCUMULATING;
    
//    private int approach = QLEARNING;
    private int approach = SARSA;
    
    private double[] lowerBounds;
    
    private double[] upperBounds;
    
    private ArrayList<Point> centers = new ArrayList<Point>();
    
    private ArrayList<HashMap<Integer,Tile>> tilings = new ArrayList<HashMap<Integer,Tile>>();
    
    private boolean initialized = false;
    
    private double step = 1.0/tilesPerDimension;
    
    private boolean randomValues = false;
    
    private FileWriter writer = null;
    
    private Scanner reader = null;
    
    private Scanner mapping = null;
    
    private boolean write = true;
    
    private boolean read = true;
    
    private int[][] stateMappings = null;
    
    private int[][] actionMappings = null;
    
    private String filenameReader;
    
	private String filenameMapping;
	
	private int tlRuns = 1;
    
    /**
     * Inner class representing a n-dimensional point in space
     * 
     * @author Kyriakos C. Chatzidimitriou (EMAIL - kyrcha [at] gmail (dot) com, WEB - http://kyrcha.info)
     *
     */
    class Point {
    	double[] s;
    	
    	public Point(int dimension) {
    		s = new double[dimension];
    	}
    	
    }
    
    /**
     * Inner class representing a tile
     * 
     * @author Kyriakos C. Chatzidimitriou (EMAIL - kyrcha [at] gmail (dot) com, WEB - http://kyrcha.info)
     *
     */
    class Tile {
    	double value;
    	double e = 0;
    	boolean selected = false;
    	
    	public Tile(){}
    	public Tile(double avalue){value = avalue;}
    }
    
    /**
     * Inner class to bundle returned action and associated Q-value
     * 
     * @author Kyriakos C. Chatzidimitriou (EMAIL - kyrcha [at] gmail (dot) com, WEB - http://kyrcha.info)
     *
     */
    class Q {
    	int action = -1;
    	double value;
    	boolean isRandom;
    }
    
    public CMACAgent(String fileName) {
		Properties props = Utils.loadProperties(fileName);	
		numberOfTilings = Integer.parseInt(props.getProperty("tilings"));
		tilesPerDimension = Integer.parseInt(props.getProperty("tiles-per-dimension"));
		lambda = Double.parseDouble(props.getProperty("lambda"));
		gamma = Double.parseDouble(props.getProperty("gamma"));
		double factor = Double.parseDouble(props.getProperty("alpha.factor"));
		double nom = Double.parseDouble(props.getProperty("alpha.nom"));
		alpha = factor * (nom/numberOfTilings);
		epsilon = Double.parseDouble(props.getProperty("epsilon"));
		offset = Double.parseDouble(props.getProperty("offset"));
		randomValues = Boolean.parseBoolean(props.getProperty("values.random"));
		// Check if there is a header in order to export SARSA values
		write = Boolean.parseBoolean(props.getProperty("file.write"));
		String filenameWriter = props.getProperty("file.towrite");
		read = Boolean.parseBoolean(props.getProperty("file.read"));
		filenameReader = props.getProperty("file.toread");
		filenameMapping = props.getProperty("file.mapping");
		if(props.containsKey("tl.runs"))
			tlRuns = Integer.parseInt(props.getProperty("tl.runs"));
		if(write) {
			try {
				File f = new File(filenameWriter);
				f.delete();
				writer = new FileWriter(filenameWriter, true);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
    }

    public void agent_init(String taskSpecification) {
		TaskSpec theTaskSpec = new TaskSpec(taskSpecification);
		logger.trace("CMAC agent parsed the task spec.");
		logger.trace("Observations have "+ theTaskSpec.getNumContinuousObsDims() + " continuous dimensions");
		logger.trace("Actions have "+ theTaskSpec.getNumDiscreteActionDims() + " integer dimensions");
		numberOfDimensions = theTaskSpec.getNumContinuousObsDims() + theTaskSpec.getNumDiscreteObsDims();
		lowerBounds = new double[numberOfDimensions];
		upperBounds = new double[numberOfDimensions];
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
			numberOfActions = theTaskSpec.getDiscreteActionRange(0).getRangeSize();
			logger.trace("Action range is: " + theActRange.getMin() + " to " + theActRange.getMax());
		}
		DoubleRange theRewardRange = theTaskSpec.getRewardRange();
		logger.trace("Reward range is: " + theRewardRange.getMin() + " to " + theRewardRange.getMax());
		
		if(!initialized) {
			System.out.println("initialize tiles");
			// Initialize Tiles Randomly (offsets and values)
			for(int i = 0; i < numberOfTilings; i++) {
				HashMap<Integer,Tile> tiling = new HashMap<Integer,Tile>();
				for(int j = 0; j < (Math.pow(tilesPerDimension, numberOfDimensions) * numberOfActions); j++) {
					double value = 0;
					if(randomValues) {
						value = Utils.rand.nextDouble();
					}
					Tile t = new Tile(value);
					tiling.put(new Integer(j),t);
				}
				System.out.println("Number of tiles/tiling: " + tiling.size());
				tilings.add(tiling);
			}
			// add center at zero
			Point p = new Point(numberOfDimensions);
			for(int j = 0; j < numberOfDimensions; j++) {
				p.s[j] = 0;
			}
			centers.add(p);
			// add the rest with random offset
			for(int j = 1; j < numberOfTilings; j++) {
				p = new Point(numberOfDimensions);
				for(int k = 0; k < numberOfDimensions; k++) {
					p.s[k] = Utils.rand.nextDouble() * offset;
				}
				centers.add(p);
			}
			System.out.println("Number of centers: " + centers.size());
			initialized = true;
			double power = Math.pow(tilesPerDimension, numberOfDimensions);
			for(int r = 0; r < tlRuns; r++) {
				// Mapping
				if(read) {
					try {
						reader = new Scanner(new File(filenameReader));
						mapping = new Scanner(new File(filenameMapping));
					} catch (FileNotFoundException e) {
						e.printStackTrace();
					}
					// Read the state mapping
					String states = mapping.nextLine();
					String[] stateTokens = states.split(";");
					int sourceStates = Integer.parseInt(stateTokens[0]);
					int targetStates = Integer.parseInt(stateTokens[1]);
					stateMappings = new int[sourceStates][targetStates];
					String actions = mapping.nextLine();
					String[] actionsTokens = actions.split(";");
					int sourceActions = Integer.parseInt(actionsTokens[0]);
					int targetActions = Integer.parseInt(actionsTokens[1]);
					actionMappings = new int[sourceActions][targetActions];
					// Read state mappings
					while(mapping.hasNextLine()) {
						String line = mapping.nextLine();
						if(line.equalsIgnoreCase("-")) {
							break;
						} else {
							String[] tokens = line.split(";");
							int source = Integer.parseInt(tokens[0]);
							int target = Integer.parseInt(tokens[1]);
							stateMappings[source][target] = 1;
						}
					}
					// Read actions mappings
					while(mapping.hasNextLine()) {
						String line = mapping.nextLine();
						String[] tokens = line.split(";");
						int source = Integer.parseInt(tokens[0]);
						int target = Integer.parseInt(tokens[1]);
						actionMappings[source][target] = 1;
					}
					// Read the samples and make the updates
				    Action action1 = new Action(1, 0, 0);
				    Action action2 = new Action(1, 0, 0);
				    Point state1 = new Point(targetStates);
				    Point state2 = new Point(targetStates);
				    int samplesCounter = 0;
					while(reader.hasNextLine()) {
						samplesCounter++;
						String line = reader.nextLine();
						String[] tokens = line.split(",");
						state1 = new Point(targetStates);
						Point sourceState = new Point(sourceStates);
						for(int j = 0; j < sourceStates; j++) {
							sourceState.s[j] = Double.parseDouble(tokens[j]);
						}
						for(int i = 0; i < targetStates; i++) {
							int index = -1;
							for(int j = 0; j < sourceStates; j++) {
								if(stateMappings[j][i] == 1) {
									index = j;
									break;
								}
							}
							state1.s[i] = sourceState.s[index];
						}
						int sourceAction = Integer.parseInt(tokens[sourceStates]);
						for(int i = 0; i < targetActions; i++) {
							if(actionMappings[sourceAction][i] == 1) {
								action1.intArray[0] = i;
								break;
							}
						}
						double reward = Double.parseDouble(tokens[sourceStates + 1]);
						int offset = sourceStates + 2;
						if(tokens.length != offset) {
							sourceState = new Point(sourceStates);
							for(int j = offset; j < offset + sourceStates; j++) {
								sourceState.s[j-offset] = Double.parseDouble(tokens[j]);
							}
							for(int i = 0; i < targetStates; i++) {
								int index = -1;
								for(int j = 0; j < sourceStates; j++) {
									if(stateMappings[j][i] == 1) {
										index = j;
										break;
									}
								}
								state2.s[i] = sourceState.s[index];
							}
							sourceAction = Integer.parseInt(tokens[offset + sourceStates]);
							for(int i = 0; i < targetActions; i++) {
								if(actionMappings[sourceAction][i] == 1) {
									action2.intArray[0] = i;
									break;
								}
							}
						}
						double Qvalue = 0;
						for(int j = 0; j < numberOfTilings; j++) {
							int address = (int)power * action1.intArray[0];
							Point center = centers.get(j);
							if(outside(state1, center)) continue;
							int index = -1;
							for(int k = 0; k < numberOfDimensions; k++) {
								index = (int)((state1.s[k] - center.s[k]) / step);
								address += Math.pow(tilesPerDimension, k) * index;
							}
							Integer objectAddress = new Integer(address);
							HashMap<Integer,Tile> tiling = tilings.get(j);
							Tile t = tiling.get(objectAddress);
							Qvalue += t.value;
						}
						double QvaluePrime = 0;
						if(tokens.length != offset) {
							for(int j = 0; j < numberOfTilings; j++) {
								int address = (int)power * action2.intArray[0];
								Point center = centers.get(j);
								if(outside(state2, center)) continue;
								int index = -1;
								for(int k = 0; k < numberOfDimensions; k++) {
									index = (int)((state2.s[k] - center.s[k]) / step);
									address += Math.pow(tilesPerDimension, k) * index;
								}
								Integer objectAddress = new Integer(address);
								HashMap<Integer,Tile> tiling = tilings.get(j);
								Tile t = tiling.get(objectAddress);
								QvaluePrime += t.value;
							}
						}
				    	// find delta
						double delta = reward + gamma*QvaluePrime - Qvalue;
						// Update tiles
						for(int j = 0; j < numberOfTilings; j++) {
							int address = (int)power * action1.intArray[0];
							Point center = centers.get(j);
							if(outside(state1, center)) continue;
							int index = -1;
							for(int k = 0; k < numberOfDimensions; k++) {
								index = (int)((state1.s[k] - center.s[k]) / step);
								address += Math.pow(tilesPerDimension, k) * index;
							}
							Integer objectAddress = new Integer(address);
							HashMap<Integer,Tile> tiling = tilings.get(j);
							Tile t = tiling.get(objectAddress);
							t.value += 0.1 * delta;
						}
					}
//					System.out.println("Number of samples used: " + samplesCounter);
				}
				
			}
    	}
		// discount epsilon
		epsilon *= 0.99;
    }
    
    private Point normalize(Observation obs) {
    	Point p = new Point(numberOfDimensions);
    	for(int i = 0; i < numberOfDimensions; i++) {
    		p.s[i] = (obs.getDouble(i) - lowerBounds[i]) / (upperBounds[i] - lowerBounds[i]);
    	}
    	return p;
    }

    public Action agent_start(Observation observation) {
    	Point p = normalize(observation); // normalize observation
    	// Create the action 
        Action returnAction = new Action(1, 0, 0);
        Q q = egreedy(p);
        int action = q.action;
    	returnAction.intArray[0] = action;
    	lastAction = returnAction;
    	lastState = p;
        return returnAction;
    }
    
    private Action sarsa(double reward, Point p) {
    	// set e
    	double power = Math.pow(tilesPerDimension, numberOfDimensions);
    	for(int i = 0; i < numberOfTilings; i++) {
			HashMap<Integer,Tile> tiling = tilings.get(i);
			for(int j = 0; j < (power * numberOfActions); j++) {
				Tile t = tiling.get(new Integer(j));
				t.e *= gamma * lambda;
			}	
		}
    	// update e and find Qvalue
    	double Qvalue = 0;
		for(int j = 0; j < numberOfTilings; j++) {
			int address = (int)power * lastAction.intArray[0];
			Point center = centers.get(j);
			if(outside(lastState, center)) continue;
			int index = -1;
			for(int k = 0; k < numberOfDimensions; k++) {
				index = (int)((lastState.s[k] - center.s[k]) / step);
				address += Math.pow(tilesPerDimension, k) * index;
			}
			Integer objectAddress = new Integer(address);
			HashMap<Integer,Tile> tiling = tilings.get(j);
			Tile t = tiling.get(objectAddress);
			if(typeOfTrace == ACCUMULATING) {
				t.e += 1.0;
			} else {
				t.e = 1.0;
			}
			Qvalue += t.value;
		}
    	// find delta
		double delta = reward - Qvalue;
        Action returnAction = new Action(1, 0, 0);
        Q q = egreedy(p);
        int action = q.action;
    	returnAction.intArray[0] = action;
    	lastAction = returnAction;
    	lastState = p;
        // final update
    	delta += gamma * q.value;
		for(int i = 0; i < numberOfTilings; i++) {
			HashMap<Integer,Tile> tiling = tilings.get(i);
			for(int j = 0; j < (power * numberOfActions); j++) {
				Tile t = tiling.get(new Integer(j));
				t.value += alpha * delta * t.e;
			}	
		}
        return returnAction;
    }
    
    private Action qlearning(double reward, Point p) {
    	// update e and find Qvalue
    	double Qvalue = 0;
    	double power = Math.pow(tilesPerDimension, numberOfDimensions);
		for(int j = 0; j < numberOfTilings; j++) {
			int address = (int)power * lastAction.intArray[0];
			Point center = centers.get(j);
			if(outside(lastState, center)) continue;
			int index = -1;
			for(int k = 0; k < numberOfDimensions; k++) {
				index = (int)((lastState.s[k] - center.s[k]) / step);
				address += Math.pow(tilesPerDimension, k) * index;
			}
			Integer objectAddress = new Integer(address);
			HashMap<Integer,Tile> tiling = tilings.get(j);
			Tile t = tiling.get(objectAddress);
			t.e += 1.0;
			Qvalue += t.value;
		}
		// find delta
		double delta = reward - Qvalue;
		double[] actionValues = calcActionValues(p);
		double maxValue = actionValues[0];
		for(int i = 1; i < actionValues.length; i++) {
			if(maxValue < actionValues[i]) {
				maxValue = actionValues[i];
			}
		}
		delta += gamma * maxValue;
		for(int i = 0; i < numberOfTilings; i++) {
			HashMap<Integer,Tile> tiling = tilings.get(i);
			for(int j = 0; j < (power * numberOfActions); j++) {
				Tile t = tiling.get(new Integer(j));
				t.value += alpha * delta * t.e;
			}	
		}
		Action returnAction = new Action(1, 0, 0);
	    Q q = egreedy(p);
	    int action = q.action;
	    returnAction.intArray[0] = action;
	    lastAction = returnAction;
	    lastState = p;
     	// set e
    	for(int i = 0; i < numberOfTilings; i++) {
			HashMap<Integer,Tile> tiling = tilings.get(i);
			for(int j = 0; j < (power * numberOfActions); j++) {
				Tile t = tiling.get(new Integer(j));
				if(q.isRandom) {
					t.e = 0;
				} else {
					t.e *= gamma * lambda;
				}
			}	
		}
	    
	    return returnAction;
    }

    public Action agent_step(double reward, Observation observation) {
    	Point p = normalize(observation); // normalize observation
    	StringBuilder sb = new StringBuilder();
    	for(int i = 0; i < lastState.s.length; i++) {
    		sb.append(lastState.s[i]);
    		sb.append(",");
    	}
    	sb.append(lastAction.intArray[0]);
    	sb.append(",");
    	sb.append(reward);
    	sb.append(",");
    	for(int i = 0; i < p.s.length; i++) {
    		sb.append(p.s[i]);
    		sb.append(",");
    	}
    	Action a = null;
    	if(approach == SARSA) {
    		a = sarsa(reward, p);
    	} else {
    		a = qlearning(reward, p);
    	}
    	sb.append(a.intArray[0]);
    	sb.append("\n");
    	if(writer != null) {
    		try {
				writer.write(sb.toString());
				writer.flush();
			} catch (IOException e) {
				e.printStackTrace();
			}
    	}
    	return a;
    }

    public void agent_end(double reward) {
    	double power = Math.pow(tilesPerDimension, numberOfDimensions);
    	for(int i = 0; i < numberOfTilings; i++) {
			HashMap<Integer,Tile> tiling = tilings.get(i);
			for(int j = 0; j < (power * numberOfActions); j++) {
				Tile t = tiling.get(new Integer(j));
				t.e *= gamma * lambda;
			}	
		}
    	// update e and find Qvalue
    	double Qvalue = 0;
		for(int j = 0; j < numberOfTilings; j++) {
			int address = (int)power * lastAction.intArray[0];
			Point center = centers.get(j);
			if(outside(lastState, center)) continue;
			int index = -1;
			for(int k = 0; k < numberOfDimensions; k++) {
				index = (int)((lastState.s[k] - center.s[k]) / step);
				address += Math.pow(tilesPerDimension, k) * index;
			}
			Integer objectAddress = new Integer(address);
			HashMap<Integer,Tile> tiling = tilings.get(j);
			Tile t = tiling.get(objectAddress);
			if(typeOfTrace == ACCUMULATING) {
				t.e += 1.0;
			} else {
				t.e = 1.0;
			}
			Qvalue += t.value;
		}
    	// find delta
		double delta = reward - Qvalue;
		// final update
		for(int i = 0; i < numberOfTilings; i++) {
			HashMap<Integer,Tile> tiling = tilings.get(i);
			for(int j = 0; j < (power * numberOfActions); j++) {
				Tile t = tiling.get(new Integer(j));
				t.value += alpha * delta * t.e;
			}	
		}
		// write to file
		StringBuilder sb = new StringBuilder();
    	for(int i = 0; i < lastState.s.length; i++) {
    		sb.append(lastState.s[i]);
    		sb.append(",");
    	}
    	sb.append(lastAction.intArray[0]);
    	sb.append(",");
    	sb.append(reward);
    	sb.append("\n");
    	if(writer != null) {
    		try {
				writer.write(sb.toString());
				writer.flush();
			} catch (IOException e) {
				e.printStackTrace();
			}
    	}
    }

    public void agent_cleanup() {
        lastAction = null;
        lastState = null;
        // Set e and F to initial values
        double power = Math.pow(tilesPerDimension, numberOfDimensions);
		for(int i = 0; i < numberOfTilings; i++) {
			HashMap<Integer,Tile> tiling = tilings.get(i);
			for(int j = 0; j < (power * numberOfActions); j++) {
				Tile t = tiling.get(new Integer(j));
				t.e = 0;
			}
		}
    }
    
    private boolean outside(Point p, Point center) {
    	boolean outside = false;
    	for(int i = 0; i < p.s.length; i++) {
    		if(p.s[i] < center.s[i]) outside = true;
    	}
    	return outside;
    }
    

    private double[] calcActionValues(Point state) {
    	double[] actionValues = new double[numberOfActions];
    	double power = Math.pow(tilesPerDimension, numberOfDimensions);
		for(int i = 0; i < numberOfActions; i++) {
			for(int j = 0; j < numberOfTilings; j++) {
				int address = (int)power * i;
				Point center = centers.get(j);
				if(outside(state, center)) continue;
				int index = -1;
				for(int k = 0; k < numberOfDimensions; k++) {
					index = (int)((state.s[k] - center.s[k]) / step);
					address += Math.pow(tilesPerDimension, k) * index;
				}
				Integer objectAddress = new Integer(address);
				Tile t = tilings.get(j).get(objectAddress);
				actionValues[i] += t.value;
			}
		}
		return actionValues;
    }
    
	protected Q egreedy(Point state) {
		double[] actionValues = calcActionValues(state);
		Q q = new Q();
		if(Utils.rand.nextDouble() < epsilon) {
			q.action = Utils.rand.nextInt(numberOfActions);
			q.value = actionValues[q.action];
			q.isRandom = true;
		} else {
			double maxValue = actionValues[0];
			q.action = 0;
			for(int i = 1; i < numberOfActions; i++) {
				if(maxValue < actionValues[i]) {
					maxValue = actionValues[i];
					q.action = i;
				}
			}
			q.value = actionValues[q.action];
			q.isRandom = false;
		}
    	return q;
	}

	@Override
	public String agent_message(String message) {
		if(message.equals("close-file")) {
			if(writer != null) {
				try {
					writer.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return null;
	}
	
}
