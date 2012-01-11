package info.kyrcha.fiterr.ne.neat;

import java.util.ArrayList;


import java.util.Iterator;
import java.util.Properties;

import info.kyrcha.fiterr.Utils;
import info.kyrcha.fiterr.ne.neat.Innovation;
import info.kyrcha.fiterr.ne.MetaNEAT;

/**
 * NEAT is a class implementing the NeuroEvolution of Augmented topologies algorithm.
 * 
 * @author Kyriakos C. Chatzidimitriou (EMAIL - kyrcha [at] gmail (dot) com, WEB - http://kyrcha.info)
 *
 */
public class NEAT extends MetaNEAT {
	
	// Class variables
	
	// Scope: private
	
	/** Global innovation number */
	private int innovationNumber;
	
	/** Innovation list */
	private ArrayList<Innovation> innovationList = new ArrayList<Innovation>();
	
	// Scope: package
	
	static final int NOT_FOUND = -1;
	
	/** The maximum change that the weights can mutate */
	static double weightRange = 2.5;
	
	/** The probability of mutating the weights */
	static double weightMutationProbability = 0.9;
	
	static double C1 = 1.0;
	
	static double C2 = 1.0;
	
	static double C3 = 0.4;
	
	static double recurrentProb = 0.0;
	
	static double mateChoose = 0.6;
	
	static double mateAverage = 0.4;
	
	static double addNodeProb = 0.03;
	
	static double addLinkProb = 0.05;
	
	/** Mutation type */
	static final String MUT_TYPE = "Gaussian";
	
	// Constructors
	
	/** 
	 * Create a NEAT implementation.
	 */
	public NEAT(String propsfile) {
		setUpNetworks(propsfile);
	}
	
	// End of constructors
	
	// Class methods
	
	// Scope: private
	
	private void setUpNetworks(String propsfile) {
		Properties props = Utils.loadProperties(propsfile);
		// Set up the population.
		populationSize = Integer.parseInt(props.getProperty("populationSize"));
		population = new NEATGenome[populationSize];
		// Set up NEAT parameters
		weightRange = Double.parseDouble(props.getProperty("weight-mut-power"));
		recurrentProb = Double.parseDouble(props.getProperty("recur-prop"));
		C1 = Double.parseDouble(props.getProperty("disjoint-coeff.c1"));
		C2 = Double.parseDouble(props.getProperty("excess-coeff.c2"));
		C3 = Double.parseDouble(props.getProperty("mutdiff-coeff.c3"));
		Ct = Double.parseDouble(props.getProperty("compat-threshold"));
		survivalThreshold = Double.parseDouble(props.getProperty("survival-thresh"));
		mutateOnly = Double.parseDouble(props.getProperty("mutate-only-prob"));
		weightMutationProbability = Double.parseDouble(props.getProperty("mutate-link-weights-prob"));
		addNodeProb = Double.parseDouble(props.getProperty("mutate-add-node-prob.mn"));
		addLinkProb = Double.parseDouble(props.getProperty("mutate-add-link-prob.ml"));
		interspeciesMatingRange = Double.parseDouble(props.getProperty("interspecies-mate-rate"));
		mateOnly = Double.parseDouble(props.getProperty("mate-only-prob"));
		mateChoose = Double.parseDouble(props.getProperty("mate-multipoint-prob"));
		mateAverage = Double.parseDouble(props.getProperty("mate-multipoint-avg-prob"));
		numSpeciesTarget = Integer.parseInt(props.getProperty("num-species-target"));
		numCompatMode = Double.parseDouble(props.getProperty("num-compat-mod"));
		useCompatMode = Boolean.parseBoolean(props.getProperty("use-compat-mod"));
		// Initialize genomes
		// Check to see if an initial genome template has been given
		if(props.getProperty("genome") != null) {
			//TODO Initialize with template
		} else { // else create the standard
			nInputUnits = Integer.parseInt(props.getProperty("nInputUnits"));
			nOutputUnits = Integer.parseInt(props.getProperty("nOutputUnits"));
			innovationNumber = nInputUnits * nOutputUnits;
			for(int i = 0; i < populationSize; i++) {
				population[i] = new NEATGenome(this);
			}
		}
		// Initialize the current generation
		currGen = 1;
	}
	
	// Scope: public
	@Override
	public void preproc(){}
	
	/** 
	 * Return the current innovation number *
	 * 
	 * @return Current innovation number
	 */
	public int getInnovationNumber() {
		return innovationNumber;
	}
	
	/**
	 * Increments and return the innovation number
	 * 
	 * @return Innovation number after the increment
	 */
	public int incrementInnovationNumber() {
		return ++innovationNumber;
	}
	
	/**
	 * The method returns the innovation number if an innovation (connection gene) 
	 * has already been occurred in the current generation. If the connection is not
	 * found it returns -1.
	 * 
	 * @param from The node id of the beginning of a connection.
	 * @param to The node id of the end of a connection.
	 * @return The innovation number if a connection exists and -1 if not.
	 */
	public int exists(int from, int to) {
		// Calculate key
		Iterator<Innovation> iter = innovationList.iterator();
		while(iter.hasNext()) {
			Innovation next = iter.next();
			if(next.getFrom() == from && next.getTo() == to) {
				return next.getInnovationNumber();
			}
		}
		return NOT_FOUND;
	}
	
	/**
	 * Add new innovation to the innovation list
	 * @param from
	 * @param to
	 * @param gin
	 */
	public void addInnovation(int from, int to, int gin) {
		innovationList.add(new Innovation(from, to, gin));
	}

	@Override
	public void cleanup() {
		innovationList = new ArrayList<Innovation>();
	}
	
	// End of class methods
	
}
