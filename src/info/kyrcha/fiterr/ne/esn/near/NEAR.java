package info.kyrcha.fiterr.ne.esn.near;

import java.util.Properties;

import info.kyrcha.fiterr.Function;
import info.kyrcha.fiterr.LearningMode;
import info.kyrcha.fiterr.Utils;
import info.kyrcha.fiterr.esn.WeightComputeMethod;
import info.kyrcha.fiterr.ne.MetaNEAT;

public class NEAR extends MetaNEAT {
	
	/** Density upper bound */
	public static final double D_UB = 1.0;
	
	/** Density lower bound */
	public static final double D_LB = 0.05;
	
	/** Spectral radius upper bound */
	public static final double RHO_UB = 0.99;
	
	/** Spectral radius lower bound */
	public static final double RHO_LB = 0.55;
	
	protected Function reservoirActivationFunction;
	
	protected Function outputActivationFunction;
	
	protected double C1;
	
	protected double C2;

	protected double C3;
	
	protected double[] inputScaling;
	
	protected double[] inputShift;
	
	protected double[] teacherScaling;
	
	protected double[] teacherShift;
	
	protected double[] feedbackScaling;
	
	protected double noiseLevel;
	
	protected double mutateRHO;
	
	protected double mutateD;
	
	protected double mutateLinkWeights;
	
	protected double mn;
	
	protected double ml;
	
	protected double weightMutationPower;
	
	protected WeightComputeMethod weightComputeMethod;
	
	protected LearningMode learningMode;
	
	public static double RLS_lambda;
	
	public static double RLS_delta;
	
	protected boolean darwinian;
	
	protected boolean crossoverAdaptation;
	
	protected int initIntNodes;
	
	protected double fittestProbability = 0.5;
	
	protected double largestProbability = 0.5;
	
	protected double fittestQuality;
	
	protected double largestQuality;
	
	protected double s1 = 1;
	
	protected double s2 = 1;

	@Override
	public void cleanup() { }
	
	@Override
	public void preproc(){
		crossoverAdaptation();
	}

	public NEAR(String propsfile) {
		setUpNetworks(propsfile);
	}
	
	private void setUpNetworks(String propsfile) {
		Properties props = Utils.loadProperties(propsfile);
		populationSize = Integer.parseInt(props.getProperty("populationSize"));
		population = new NEARGenome[populationSize];
		useCompatMode = Boolean.parseBoolean(props.getProperty("use-compat-mode"));
		numCompatMode = Double.parseDouble(props.getProperty("num-compat-mod"));
		numSpeciesTarget = Integer.parseInt(props.getProperty("num-species-target"));
		C1 = Double.parseDouble(props.getProperty("c1"));
		C2 = Double.parseDouble(props.getProperty("c2"));
		C3 = Double.parseDouble(props.getProperty("c3"));
		Ct = Double.parseDouble(props.getProperty("compat-threshold"));
		survivalThreshold = Double.parseDouble(props.getProperty("survival-thresh"));
		interspeciesMatingRange = Double.parseDouble(props.getProperty("interspecies-mate-rate"));
		noiseLevel = Double.parseDouble(props.getProperty("noiseLevel"));
		mutateRHO = Double.parseDouble(props.getProperty("mutate-SR"));
		mutateD = Double.parseDouble(props.getProperty("mutate-sparse"));
		mutateOnly = Double.parseDouble(props.getProperty("mutate-only-prob"));
		mateOnly = Double.parseDouble(props.getProperty("mate-only-prob"));
		mutateLinkWeights = Double.parseDouble(props.getProperty("mutate-link-weights-prob"));
		mn = Double.parseDouble(props.getProperty("mutate-add-node-prob.mn")); 
		ml = Double.parseDouble(props.getProperty("mutate-add-link-prob.ml"));
		weightMutationPower = Double.parseDouble(props.getProperty("weight-mut-power"));
		learningMode = LearningMode.valueOf(props.getProperty("learningMode"));
		weightComputeMethod = WeightComputeMethod.valueOf(props.getProperty("methodWeightCompute"));
		RLS_lambda = Double.parseDouble(props.getProperty("RLS_lambda"));
		RLS_delta = Double.parseDouble(props.getProperty("RLS_delta"));
		darwinian = Boolean.parseBoolean(props.getProperty("darwinian"));
		initIntNodes = Integer.parseInt(props.getProperty("init-int-nodes"));
		crossoverAdaptation = Boolean.parseBoolean(props.getProperty("crossover-adaptation"));
		System.out.println(crossoverAdaptation);
		// Initialize genomes
		// Check to see if an initial genome template has been given
		if(props.getProperty("genome") != null) {
			//TODO Initialize with template
		} else { // else create the standard
			nInputUnits = Integer.parseInt(props.getProperty("nInputUnits"));
			nOutputUnits = Integer.parseInt(props.getProperty("nOutputUnits"));
			for(int i = 0; i < populationSize; i++) {
				population[i] = new NEARGenome(this);
			}
		}
		reservoirActivationFunction = Function.valueOf(props.getProperty("reservoir-activation-function"));
		outputActivationFunction = Function.valueOf(props.getProperty("output-activation-function"));
		if(nInputUnits > 0) {
			inputScaling = Utils.getMultiProps(props, "inputScaling", nInputUnits);
			inputShift = Utils.getMultiProps(props, "inputShift", nInputUnits); 
    	}
		teacherScaling = Utils.getMultiProps(props, "teacherScaling", nOutputUnits);
		teacherShift = Utils.getMultiProps(props, "teacherShift", nOutputUnits);
		feedbackScaling = Utils.getMultiProps(props, "feedbackScaling", nOutputUnits);
		// Initialize the current generation
		currGen = 1;
	}
	
	public double getFittestProb() {
		return fittestProbability;
	}
	
	private void crossoverAdaptation() {
		if(currGen > 0) {
			// update sums and counters
			fittestQuality = 0;
			int fittestCounter = 0;
			largestQuality = 0;
			int largestCounter = 0;
			for(int i = 0; i < populationSize; i++) {
				NEARGenome g = (NEARGenome)population[i];
				double diff = Math.max(0, g.getFitness() - Math.max(g.getParentFitness1(), g.getParentFitness2()));
				if(g.getOutOfCrossoverFittest()) {
					fittestQuality += diff;
					fittestCounter++;
				} else if(g.getOutOfCrossoverLargest()) {
					largestQuality += diff;
					largestCounter++;
				}
			}
			if(fittestCounter > 0) {
				fittestQuality /= fittestCounter;
			}
			if(largestCounter > 0) {
				largestQuality /= largestCounter;
			}
			
			// calculations
			double qall = fittestQuality + largestQuality;
			double cOmega = 0.5;
			double pmin = 0.1;
			if(qall > 0) {
				s1 = ((cOmega * fittestQuality) / qall) + (1 - cOmega) * s1;
				s2 = ((cOmega * largestQuality) / qall) + (1 - cOmega) * s2;
			} else {
				s1 = (cOmega/2) + (1-cOmega) * s1;
				s2 = (cOmega/2) + (1-cOmega) * s2;
			}
			fittestProbability = pmin + (((1 - 2 * pmin) * s1) /(s1 + s2));
			largestProbability = pmin + (((1 - 2 * pmin) * s2) /(s1 + s2));
			System.out.println(fittestProbability + " " + largestProbability);
		}
	}

}
