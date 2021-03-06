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
 * TODO store to a file SARSA-tuple
 * TODO transfer learning through transfering samples and pre-learning 
 * 
 * @author Kyriakos Chatzidimitriou, email: kyrcha [at] issel (dot) ee (dot) auth (dot) gr
 */
public class FourStatesCMACAgent implements AgentInterface {
	
	private static final int REPLACING = 1;
	
	private static final int ACCUMULATING = 2;
	
	private static final int SARSA = 1;
	
	private static final int QLEARNING = 2;
	
	/** Logger named after the class */
	private static final Logger logger = Logger.getLogger(FourStatesCMACAgent.class.getName());
    
    private Action lastAction;
    
    private Point lastState;
    
    private int numberOfTilings = 10;
    
    private int tilesPerDimension = 10;
    
    private int numberOfDimensions;
    
    private int numberOfActions;
    
    private double lambda = 0.95;
    
    private double gamma = 1.0;
    
	private double epsilon = 0.0;
    
    private double alpha = 0.05 * (0.1/numberOfTilings);
    
    private double offset = 0.1;
    
//    private int typeOfTrace = ACCUMULATING;
    private int typeOfTrace = REPLACING;
    
//    private int approach = QLEARNING;
    private int approach = SARSA;
    
    private double[] lowerBounds;
    
    private double[] upperBounds;
    
    private ArrayList<Point> centers = new ArrayList<Point>();
    
    private double[][][][][][] tilings;
    
    private double[][][][][][] traces;
    
    private boolean initialized = false;
    
    private double step = 1.0/tilesPerDimension;
    
    private boolean randomValues = false;
    
    private FileWriter writer = null;
    
    private Scanner reader = null;
    
    private Scanner mapping = null;
    
    private boolean write = true;
    
    private boolean read = true;
    
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
    
    public FourStatesCMACAgent(String fileName) {
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
			tilings = new double[numberOfTilings][tilesPerDimension][tilesPerDimension][tilesPerDimension][tilesPerDimension][numberOfActions];
			traces = new double[numberOfTilings][tilesPerDimension][tilesPerDimension][tilesPerDimension][tilesPerDimension][numberOfActions];
			System.out.println("initialize tiles");
			// Initialize Tiles Randomly (offsets and values)
			if(randomValues) {
				for(int i1 = 0; i1 < tilings.length; i1++) {
					for(int i2 = 0; i2 < tilings[i1].length; i2++) {
						for(int i3 = 0; i3 < tilings[i1][i2].length; i3++) {
							for(int i4 = 0; i4 < tilings[i1][i2][i3].length; i4++) {
								for(int i5 = 0; i5 < tilings[i1][i2][i3][i4].length; i5++) {
									for(int i6 = 0; i6 < tilings[i1][i2][i3][i4][i5].length; i6++) {
										tilings[i1][i2][i3][i4][i5][i6] = Utils.rand.nextDouble();
									}
								}
							}
						}
					}
				}
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
			// Read Mappings
			if(read) {
				try {
					mapping = new Scanner(new File(filenameMapping));
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				}
			    int numOfMappings = Integer.parseInt(mapping.nextLine());
				// Read the state mapping
				String states = mapping.nextLine();
				String[] stateTokens = states.split(";");
				int sourceStates = Integer.parseInt(stateTokens[0]);
				int targetStates = Integer.parseInt(stateTokens[1]);
				int[][][] stateMappings = new int[numOfMappings][sourceStates][targetStates];
				String actions = mapping.nextLine();
				String[] actionsTokens = actions.split(";");
				int sourceActions = Integer.parseInt(actionsTokens[0]);
				int targetActions = Integer.parseInt(actionsTokens[1]);
				int[][][] actionMappings = new int[numOfMappings][sourceActions][targetActions];
				int countTheMappings = 0;
				while(countTheMappings < numOfMappings) {
					// Read state mappings
					while(mapping.hasNextLine()) {
						String line = mapping.nextLine();
						if(line.equalsIgnoreCase("-")) {
							break;
						} else {
							String[] tokens = line.split(";");
							int source = Integer.parseInt(tokens[0]);
							int target = Integer.parseInt(tokens[1]);
							stateMappings[countTheMappings][source][target] = 1;
						}
					}
					// Read actions mappings
					while(mapping.hasNextLine()) {
						String line = mapping.nextLine();
						if(line.equalsIgnoreCase("-")) {
							break;
						} else {
							String[] tokens = line.split(";");
							int source = Integer.parseInt(tokens[0]);
							int target = Integer.parseInt(tokens[1]);
							actionMappings[countTheMappings][source][target] = 1;
						}
					}
					// Read the samples and make the updates
				    Action action1 = new Action(1, 0, 0);
				    Action action2 = new Action(1, 0, 0);
				    Point state1 = new Point(targetStates);
				    Point state2 = new Point(targetStates);
				    int samplesCounter = 0;
				    for(int r = 0; r < tlRuns; r++) {
				    	try {
							reader = new Scanner(new File(filenameReader));
						} catch (FileNotFoundException e) {
							e.printStackTrace();
						}
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
									if(stateMappings[countTheMappings][j][i] == 1) {
										index = j;
										break;
									}
								}
								state1.s[i] = sourceState.s[index];
							}
							int sourceAction = Integer.parseInt(tokens[sourceStates]);
							for(int i = 0; i < targetActions; i++) {
								if(actionMappings[countTheMappings][sourceAction][i] == 1) {
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
										if(stateMappings[countTheMappings][j][i] == 1) {
											index = j;
											break;
										}
									}
									state2.s[i] = sourceState.s[index];
								}
								sourceAction = Integer.parseInt(tokens[offset + sourceStates]);
								for(int i = 0; i < targetActions; i++) {
									if(actionMappings[countTheMappings][sourceAction][i] == 1) {
										action2.intArray[0] = i;
										break;
									}
								}
							}
							double Qvalue = 0;
							for(int j = 0; j < numberOfTilings; j++) {
								// Find the indices for the four dimensions
								int[] indices = new int[targetStates];
								Point center = centers.get(j);
								if(outside(state1, center)) continue;
								for(int k = 0; k < numberOfDimensions; k++) {
									indices[k] = (int)((state1.s[k] - center.s[k]) / step);
									indices[k] = Math.max(0, Math.min(indices[k], tilesPerDimension - 1));
								}
								Qvalue += tilings[j][indices[0]][indices[1]][indices[2]][indices[3]][action1.intArray[0]];
							}
							double QvaluePrime = 0;
							if(tokens.length != offset) {
								for(int j = 0; j < numberOfTilings; j++) {
									// Find the indices for the four dimensions
									int[] indices = new int[targetStates];
									Point center = centers.get(j);
									if(outside(state2, center)) continue;
									for(int k = 0; k < numberOfDimensions; k++) {
										indices[k] = (int)((state2.s[k] - center.s[k]) / step);
										indices[k] = Math.max(0, Math.min(indices[k], tilesPerDimension - 1));
									}
									QvaluePrime += tilings[j][indices[0]][indices[1]][indices[2]][indices[3]][action2.intArray[0]];
								}
							}
					    	// find delta
							double delta = reward + gamma*QvaluePrime - Qvalue;
							// Update tiles
							for(int j = 0; j < numberOfTilings; j++) {
								// Find the indices for the four dimensions
								int[] indices = new int[targetStates];
								Point center = centers.get(j);
								if(outside(state1, center)) continue;
								for(int k = 0; k < numberOfDimensions; k++) {
									indices[k] = (int)((state1.s[k] - center.s[k]) / step);
									indices[k] = Math.max(0, Math.min(indices[k], tilesPerDimension - 1));
								}
								double tempValue = tilings[j][indices[0]][indices[1]][indices[2]][indices[3]][action1.intArray[0]];
								tempValue += alpha * delta;
								tilings[j][indices[0]][indices[1]][indices[2]][indices[3]][action1.intArray[0]] = tempValue;
							}
				    	}
				    }
				    countTheMappings++;
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
    	for(int i1 = 0; i1 < tilings.length; i1++) {
			for(int i2 = 0; i2 < tilings[i1].length; i2++) {
				for(int i3 = 0; i3 < tilings[i1][i2].length; i3++) {
					for(int i4 = 0; i4 < tilings[i1][i2][i3].length; i4++) {
						for(int i5 = 0; i5 < tilings[i1][i2][i3][i4].length; i5++) {
							for(int i6 = 0; i6 < tilings[i1][i2][i3][i4][i5].length; i6++) {
								traces[i1][i2][i3][i4][i5][i6] *= gamma * lambda;
							}
						}
					}
				}
			}
		}
    	// update e and find Qvalue
    	double Qvalue = 0;
    	for(int j = 0; j < numberOfTilings; j++) {
			// Find the indices for the four dimensions
			int[] indices = new int[numberOfDimensions];
			Point center = centers.get(j);
			if(outside(lastState, center)) continue;
			for(int k = 0; k < numberOfDimensions; k++) {
				indices[k] = (int)((lastState.s[k] - center.s[k]) / step);
				indices[k] = Math.max(0, Math.min(indices[k], tilesPerDimension - 1));
			}
			if(typeOfTrace == ACCUMULATING) {
				traces[j][indices[0]][indices[1]][indices[2]][indices[3]][lastAction.intArray[0]] += 1.0;
			} else {
				traces[j][indices[0]][indices[1]][indices[2]][indices[3]][lastAction.intArray[0]] = 1.0;
			}
			Qvalue += tilings[j][indices[0]][indices[1]][indices[2]][indices[3]][lastAction.intArray[0]];
			
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
    	for(int i1 = 0; i1 < tilings.length; i1++) {
			for(int i2 = 0; i2 < tilings[i1].length; i2++) {
				for(int i3 = 0; i3 < tilings[i1][i2].length; i3++) {
					for(int i4 = 0; i4 < tilings[i1][i2][i3].length; i4++) {
						for(int i5 = 0; i5 < tilings[i1][i2][i3][i4].length; i5++) {
							for(int i6 = 0; i6 < tilings[i1][i2][i3][i4][i5].length; i6++) {
								tilings[i1][i2][i3][i4][i5][i6] += alpha * delta * traces[i1][i2][i3][i4][i5][i6];
							}
						}
					}
				}
			}
		}
        return returnAction;
    }
    
    private Action qlearning(double reward, Point p) {
    	// update e and find Qvalue
    	double Qvalue = 0;
    	for(int j = 0; j < numberOfTilings; j++) {
			// Find the indices for the four dimensions
			int[] indices = new int[numberOfDimensions];
			Point center = centers.get(j);
			if(outside(lastState, center)) continue;
			for(int k = 0; k < numberOfDimensions; k++) {
				indices[k] = (int)((lastState.s[k] - center.s[k]) / step);
				indices[k] = Math.max(0, Math.min(indices[k], tilesPerDimension - 1));
			}
			if(typeOfTrace == ACCUMULATING) {
				traces[j][indices[0]][indices[1]][indices[2]][indices[3]][lastAction.intArray[0]] += 1.0;
			} else {
				traces[j][indices[0]][indices[1]][indices[2]][indices[3]][lastAction.intArray[0]] = 1.0;
			}
			Qvalue += tilings[j][indices[0]][indices[1]][indices[2]][indices[3]][lastAction.intArray[0]];
			
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
		for(int i1 = 0; i1 < tilings.length; i1++) {
			for(int i2 = 0; i2 < tilings[i1].length; i2++) {
				for(int i3 = 0; i3 < tilings[i1][i2].length; i3++) {
					for(int i4 = 0; i4 < tilings[i1][i2][i3].length; i4++) {
						for(int i5 = 0; i5 < tilings[i1][i2][i3][i4].length; i5++) {
							for(int i6 = 0; i6 < tilings[i1][i2][i3][i4][i5].length; i6++) {
								tilings[i1][i2][i3][i4][i5][i6] += alpha * delta * traces[i1][i2][i3][i4][i5][i6];
							}
						}
					}
				}
			}
		}
		Action returnAction = new Action(1, 0, 0);
	    Q q = egreedy(p);
	    int action = q.action;
	    returnAction.intArray[0] = action;
	    lastAction = returnAction;
	    lastState = p;
     	// set e
	    for(int i1 = 0; i1 < tilings.length; i1++) {
			for(int i2 = 0; i2 < tilings[i1].length; i2++) {
				for(int i3 = 0; i3 < tilings[i1][i2].length; i3++) {
					for(int i4 = 0; i4 < tilings[i1][i2][i3].length; i4++) {
						for(int i5 = 0; i5 < tilings[i1][i2][i3][i4].length; i5++) {
							for(int i6 = 0; i6 < tilings[i1][i2][i3][i4][i5].length; i6++) {
								if(q.isRandom) {
									traces[i1][i2][i3][i4][i5][i6] = 0;
								} else {
									traces[i1][i2][i3][i4][i5][i6] *= gamma * lambda;
								}
							}
						}
					}
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
    	for(int i1 = 0; i1 < tilings.length; i1++) {
			for(int i2 = 0; i2 < tilings[i1].length; i2++) {
				for(int i3 = 0; i3 < tilings[i1][i2].length; i3++) {
					for(int i4 = 0; i4 < tilings[i1][i2][i3].length; i4++) {
						for(int i5 = 0; i5 < tilings[i1][i2][i3][i4].length; i5++) {
							for(int i6 = 0; i6 < tilings[i1][i2][i3][i4][i5].length; i6++) {
								traces[i1][i2][i3][i4][i5][i6] *= gamma * lambda;
							}
						}
					}
				}
			}
		}
    	// update e and find Qvalue
    	double Qvalue = 0;
    	for(int j = 0; j < numberOfTilings; j++) {
			// Find the indices for the four dimensions
			int[] indices = new int[numberOfDimensions];
			Point center = centers.get(j);
			if(outside(lastState, center)) continue;
			for(int k = 0; k < numberOfDimensions; k++) {
				indices[k] = (int)((lastState.s[k] - center.s[k]) / step);
				indices[k] = Math.max(0, Math.min(indices[k], tilesPerDimension - 1));
			}
			if(typeOfTrace == ACCUMULATING) {
				traces[j][indices[0]][indices[1]][indices[2]][indices[3]][lastAction.intArray[0]] += 1.0;
			} else {
				traces[j][indices[0]][indices[1]][indices[2]][indices[3]][lastAction.intArray[0]] = 1.0;
			}
			Qvalue += tilings[j][indices[0]][indices[1]][indices[2]][indices[3]][lastAction.intArray[0]];
		}
    	// find delta
		double delta = reward - Qvalue;
		// final update
		for(int i1 = 0; i1 < tilings.length; i1++) {
			for(int i2 = 0; i2 < tilings[i1].length; i2++) {
				for(int i3 = 0; i3 < tilings[i1][i2].length; i3++) {
					for(int i4 = 0; i4 < tilings[i1][i2][i3].length; i4++) {
						for(int i5 = 0; i5 < tilings[i1][i2][i3][i4].length; i5++) {
							for(int i6 = 0; i6 < tilings[i1][i2][i3][i4][i5].length; i6++) {
								tilings[i1][i2][i3][i4][i5][i6] += alpha * delta * traces[i1][i2][i3][i4][i5][i6];
							}
						}
					}
				}
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
        traces = new double[numberOfTilings][tilesPerDimension][tilesPerDimension][tilesPerDimension][tilesPerDimension][numberOfActions];
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
		for(int j = 0; j < numberOfTilings; j++) {
			// Find the indices for the four dimensions
			int[] indices = new int[numberOfDimensions];
			Point center = centers.get(j);
			if(outside(state, center)) continue;
			for(int k = 0; k < numberOfDimensions; k++) {
				indices[k] = (int)((state.s[k] - center.s[k]) / step);
				indices[k] = Math.max(0, Math.min(indices[k], tilesPerDimension - 1));
			}
			for(int i = 0; i < numberOfActions; i++) {
				actionValues[i] += tilings[j][indices[0]][indices[1]][indices[2]][indices[3]][i];
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
