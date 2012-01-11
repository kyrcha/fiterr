package info.kyrcha.fiterr.esn;

import info.kyrcha.fiterr.LearningMode;
import info.kyrcha.fiterr.Utils;

import info.kyrcha.fiterr.Function;
import info.kyrcha.fiterr.exceptions.MatrixException;

import java.util.Properties;

import org.apache.log4j.Logger;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;

/**
 * ESN is the class that constructs Echo State Networks (ESNs) as those defined in the following publications:
 * <ul>
 *   <li>H. Jaeger, The "echo state" approach to analysing and training recurrent neural networks - with an Erratum note, German National Research Center for Information Technology, 2001</li>
 *   <li>H. Jaeger, Tutorial on training recurrent neural networks, covering BPTT, RTRL, EKF and the "echo state network" approach, German National Research Center for Information Technology, 2002</li>
 * </ul> 
 * 
 * Code is mainly based on Dr. Herbert Jaeger's Matlab toolbox for ESN.
 * 
 * The functionality of this class enables it to handle:
 * <ul>
 *   <li>MIMO off-line training with Widrow-Hopf and Pseudo-Inverse</li>
 *   <li>MIMO on-line training with RLS</li>
 *   <li>Leaky integrator neurons</li>
 *   <li>Step-by-step execution of the learning machine</li>
 * </ul> 
 *
 * @author Kyriakos C. Chatzidimitriou (EMAIL - kyrcha [at] gmail (dot) com, WEB - http://kyrcha.info)
 * @version "%I%, %G%
 * 
 * TODO Leaky neurons (look Jaeger Matlab Code)
 * TODO Time-warping (look Jaeger Matlab Code)
 * TODO Exception handling in certain places
 * TODO Step by step execution (mainly for RL tasks, but also for TS tasks), flush state
 * TODO What happens when no inputs are available (teacher forcing only)
 * TODO make enumerations ESN-type, Functions, learning methods, on-line/off-line
 * TODO Test with just teacher forcing and with teacher forcing and input as well 
 * TODO Online RLS learning
 * TODO add support for output feedback weights in activateInput
 * TODO load/store trained ESN to disk
 */
public class ESN {
	
	// Private class variables

	/** Logger named after the class */
	private static final Logger logger = Logger.getLogger(ESN.class.getName());
	
	/** The number of inputs*/
	private int nInputUnits;
	
	/** The number of reservoir units */
	private int nInternalUnits;
	
	/** The number of output units */
	private int nOutputUnits;
	
	/** The total number of units */
	private int nTotalUnits;
	
	/** The learning method for computing the output units */
	private LearningMode learningMode;
	
	/** The method for computing the output weights in off-line mode */
	private WeightComputeMethod weightComputeMethod;
	
	/** The type of the ESN. Default type is define as <a href={@link #PLAIN_ESN}>PLAIN_ESN</a> */
	private ESNType type = ESNType.PLAIN_ESN;
	
	/** A flag indicating whether the network has been trained already */
	private boolean trained;
	
	private boolean minusPlusOneInput = false;
	
	private boolean reservoirStandard = false;
	
	private double reservoirDefault;
	
	private Function reservoirActivationFunction;
	
	private Function outputActivationFunction;
	
	private double[] inputScaling;
	
	private double[] inputShift;
	
	private double[] teacherScaling;
	
	private double[] teacherShift;
	
	private double[] feedbackScaling;
	
	private double[] timeConstants;
	
	private double noiseLevel;
	
	private double RLS_lambda;
	
	private double RLS_delta;
	
	/** The connectivity D of the reservoir */
	private double D;
	
	/** The required spectral radius of the ESN */
	private double spectralRadius;
	
	private double leakage;
	
	private Matrix internalWeights;
	
	private Matrix inputWeights;
	
	private Matrix inScale;
	
	private Matrix inShift;
	
	private Matrix outputWeights;
	
	private Matrix outScale;
	
	private Matrix outShift;
	
	private Matrix feedbackWeights;
	
	private Matrix internalWeightsUnitSR;
	
	private Matrix stateMatrix;
	
	private Matrix totalState;
	
	private Matrix internalState;
	
	private Matrix feedbackDiag;
	
	private Matrix feedbackMatrix;
	
	private Matrix weightMatrix;
	
	// Constructors
	
	/**
	 * Create an empty ESN. Later use get/set methods to define it. 
	 */
	public ESN() { }
	
	/**
	 * Creates a random ESN with initial parameters as defined in a parameter file.
	 */
    public ESN(String fileName) {
    	// Set up parameters
    	setUpNetwork(fileName);
    }
	
	// Public methods
    
    // Get-set
    
    
    public void flush() {
    	totalState = new Matrix(nTotalUnits, 1, 0);
    	internalState = new Matrix(nInternalUnits, 1, 0);
    }
	
	public int getNumberOfInputUnits() {
		return nInputUnits;
	}
	
	public int getNumberOfOutputUnits() {
		return nOutputUnits;
	}
	
	public void setNumberOfInputUnits(int units) {
		nInputUnits = units;
	}
	
	public void setNumberOfOutputUnits(int units) {
		nOutputUnits = units;
	}
	
	public void setNumberOfInternalUnits(int units) {
		nInternalUnits = units;
	}
	
	public void setNumberOfTotalUnits(int units) {
		nTotalUnits = units;
	}
	
	public void setLearningMode(LearningMode alm) {
		learningMode = alm;
	}
	
	public void setWeightComputeMethod(WeightComputeMethod wcm) {
		weightComputeMethod = wcm;
	}
	
	public void setESNType(ESNType esnType) {
		type = esnType;
	}
	
	public void setReservoirActivationFunction(Function raf) {
		reservoirActivationFunction = raf;
	}
	
	public void setOutputActivationFunction(Function oaf) {
		outputActivationFunction = oaf;
	}
	
	public void setDensity(double d) {
		D = d;
	}
	
	public void setSpectralRadius(double rho) {
		spectralRadius = rho;
	}
	
	public void setInputScaling(double[] ainputScaling) {
		inputScaling = ainputScaling;
		inScale = new Matrix(inputScaling, nInputUnits);
	}
	
	public void setInputShift(double[] ainputShift) {
		inputShift = ainputShift;
		inShift = new Matrix(inputShift, nInputUnits);
	}
	
	public void setOutputScaling(double[] aoutputScaling) {
		teacherScaling = aoutputScaling;
		outScale = new Matrix(teacherScaling, nOutputUnits);	
	}
	
	public void setOutputShift(double[] aoutputShift) {
		teacherShift = aoutputShift;
		outShift = new Matrix(teacherShift, nOutputUnits);
	}
	
	public void setFeedbackScaling(double[] afs) {
		feedbackScaling = afs;
    	feedbackDiag = diagonal(feedbackScaling);
    	feedbackMatrix = feedbackWeights.times(feedbackDiag);
	}
	
	public void setNoiseLevel(double anoiseLevel) {
		noiseLevel = anoiseLevel;
	}
	
	public void setInputWeights(double[][] ainputWeights) {
		inputWeights = new Matrix(Utils.cloneMatrix(ainputWeights));
	}
	
	public void setOutputWeights() {
		outputWeights = generateOutputWeights();
	}
	
	public void setOutputWeights(double[][] aoutputWeights) {
		outputWeights = new Matrix(Utils.cloneMatrix(aoutputWeights));
	}
	
	public void setFeedbackWeights() {
		feedbackWeights = generateFeedbackWeights();
	}
	
	public void setInternalWeights(double[][] aw) {
		internalWeights = new Matrix(aw);
	}
	
	public void buildWeightMatrix() {
		weightMatrix = new Matrix(nInternalUnits, nTotalUnits);
		weightMatrix.setMatrix(0, nInternalUnits - 1, 0, nInternalUnits - 1, internalWeights);
		weightMatrix.setMatrix(0, nInternalUnits - 1, nInternalUnits, nInternalUnits + nInputUnits - 1, inputWeights);
		weightMatrix.setMatrix(0, nInternalUnits - 1, nInternalUnits + nInputUnits, nTotalUnits - 1, feedbackMatrix);
    	trained = false;
	}
	
	
	public double getOutputWeightsRMS() {
		int sum = 0;
		for(int i = 0; i < outputWeights.getRowDimension(); i++) {
			for(int j = 0; j < outputWeights.getColumnDimension(); j++) {
				sum += outputWeights.get(i, j) * outputWeights.get(i, j);
			}
		}
		return Math.sqrt(sum / (outputWeights.getRowDimension() * outputWeights.getColumnDimension()));
	}
	
	public double getOutputWeightsMA() {
		int sum = 0;
		for(int i = 0; i < outputWeights.getRowDimension(); i++) {
			for(int j = 0; j < outputWeights.getColumnDimension(); j++) {
				sum += Math.abs(outputWeights.get(i, j));
			}
		}
		return Math.sqrt(sum / (outputWeights.getRowDimension() * outputWeights.getColumnDimension()));
	}
    
    // Private methods
    
    /**
     * Set up a random ESN defined in the parameter file.  
     */
    private void setUpNetwork(String fileName) {
    	// Load props to variable
		Properties props = Utils.loadProperties(fileName);
		// Input required arguments
		nInputUnits = Integer.parseInt(props.getProperty("nInputUnits"));
		nInternalUnits = Integer.parseInt(props.getProperty("nInternalUnits"));
		nOutputUnits = Integer.parseInt(props.getProperty("nOutputUnits"));
		D = Double.parseDouble(props.getProperty("D"));
		nTotalUnits = nInternalUnits + nInputUnits + nOutputUnits;
		// Optional arguments
		if(nInputUnits > 0) {
			inputScaling = Utils.getMultiProps(props, "inputScaling", nInputUnits);
			inScale = new Matrix(inputScaling, nInputUnits);
			inputShift = Utils.getMultiProps(props, "inputShift", nInputUnits); 
    		inShift = new Matrix(inputShift, nInputUnits);
    	}
		teacherScaling = Utils.getMultiProps(props, "teacherScaling", nOutputUnits);
		teacherShift = Utils.getMultiProps(props, "teacherShift", nOutputUnits);
		outScale = new Matrix(teacherScaling, nOutputUnits);
		outShift = new Matrix(teacherShift, nOutputUnits);
		if(props.getProperty("minusPlusOneInput") != null) {
			minusPlusOneInput = Boolean.parseBoolean(props.getProperty("minusPlusOneInput"));
		} else {
			minusPlusOneInput = false;
		}
		if(props.getProperty("reservoirStandard") != null) {
			reservoirStandard = Boolean.parseBoolean(props.getProperty("reservoirStandard"));
		} else {
			reservoirStandard = false;
		}
		if(reservoirStandard) {
			reservoirDefault = Double.parseDouble(props.getProperty("reservoirDefault"));
		}
		feedbackScaling = Utils.getMultiProps(props, "feedbackScaling", nOutputUnits);
		learningMode = Enum.valueOf(LearningMode.class, props.getProperty("learningMode"));
		weightComputeMethod = Enum.valueOf(WeightComputeMethod.class, props.getProperty("methodWeightCompute"));
		reservoirActivationFunction = Enum.valueOf(Function.class, props.getProperty("reservoirActivationFunction"));
		outputActivationFunction = Enum.valueOf(Function.class, props.getProperty("outputActivationFunction"));
		noiseLevel = Double.parseDouble(props.getProperty("noiseLevel"));
		spectralRadius = Double.parseDouble(props.getProperty("spectralRadius"));
		if(type == ESNType.LEAKY_ESN || type == ESNType.TWI_ESN) {
			timeConstants = Utils.getMultiProps(props, "timeConstants", nInternalUnits);
			leakage = Double.parseDouble(props.getProperty("leakage"));
		}
		if(learningMode == LearningMode.ONLINE) {
			RLS_lambda = Double.parseDouble(props.getProperty("RLS_lambda"));
			RLS_delta = Double.parseDouble(props.getProperty("RLS_delta"));
		}
    	inputWeights = generateInputWeights();
    	outputWeights = generateOutputWeights();
    	feedbackWeights = generateFeedbackWeights();
    	feedbackDiag = diagonal(feedbackScaling);
    	feedbackMatrix = feedbackWeights.times(feedbackDiag);
    	internalWeightsUnitSR = generateInternalWeights(reservoirStandard);
    	internalWeights = internalWeightsUnitSR.times(spectralRadius);
    	buildWeightMatrix();
    	trained = false;
    }
    
    /**
     * Generate a weighted matrix with elements in [-1,1]
     * @param row Number of rows
     * @param col Number of columns
     * @return The weighted matrix
     */
    private Matrix generateWeightsMinusPlusOneInt(int row, int col) {
    	Matrix b = new Matrix(row, col);
    	int[] picks = {-1,1};
    	for(int r = 0; r < row; r++) {
    		for(int c = 0; c < col; c++) {
    			b.set(r, c, picks[Utils.rand.nextInt(1)]);
    		}
    	}
    	return  b;
    }
    
    /**
     * Generate a weighted matrix with elements in [-1,1]
     * @param row Number of rows
     * @param col Number of columns
     * @return The weighted matrix
     */
    private Matrix generateWeightsMinusPlusOneRand(int row, int col) {
    	Matrix b = new Matrix(row, col, 1d);
    	Matrix r = Matrix.random(row, col);
    	return  r.times(2.0).minus(b);
    }
    
    /**
     * Generate the matrix containing the feedback weights.
     * @return The Matrix containing the feedback weights
     */
    private Matrix generateFeedbackWeights() {
    	return generateWeightsMinusPlusOneRand(nInternalUnits, nOutputUnits);
    }
    
    /** 
     * Generate the matrix containing the output weights. Initially they are set to 0.
     * @return The Matrix containing the output weights
     */
    private Matrix generateOutputWeights() {
    	return new Matrix(nOutputUnits, nInternalUnits + nInputUnits, 0);
    }
    
    /**
     * Generate the matrix containing the input weights.
     * @return The Matrix containing the input weights
     */
    private Matrix generateInputWeights() {
    	if(minusPlusOneInput) {
    		return generateWeightsMinusPlusOneInt(nInternalUnits, nInputUnits);
    	} else {
    		return generateWeightsMinusPlusOneRand(nInternalUnits, nInputUnits);
    	}
    }
    
    /**
     * Generates the matrix containing the internal weights after the 
     * @param connectivity The required connectivity or sparseness of the matrix
     * @return The reservoir connection weights
     */
    private Matrix generateInternalWeights(boolean reservoirStandard) {
    	Matrix internal = null;
    	try {
	    	internal = new Matrix(nInternalUnits, nInternalUnits, 0);
	    	if( reservoirStandard) {
	    		double[] weights = {-reservoirDefault, reservoirDefault};
		    	// Generate a random Matrix
		    	for(int i = 0; i < nInternalUnits; i++) {
		    		for(int j = 0; j < nInternalUnits; j++) {
		    			if(Utils.rand.nextDouble() < D)
		    				internal.set(i, j, weights[Utils.rand.nextInt(1)]);
		    		}
		    	}
	    	} else {
		    	// Generate a random Matrix
		    	for(int i = 0; i < nInternalUnits; i++) {
		    		for(int j = 0; j < nInternalUnits; j++) {
		    			if(Utils.rand.nextDouble() < D)
		    				internal.set(i, j, 2 * Utils.rand.nextDouble() - 1.0);
		    		}
		    	}
	    	}
	    	EigenvalueDecomposition evd = internal.eig();
	    	double[] reEigVals = evd.getRealEigenvalues();
	    	double[] imEigVals = evd.getImagEigenvalues();
	    	double[] eigVals = new double[nInternalUnits];
	    	double maxVal = Double.MIN_VALUE;
	    	for(int i = 0; i < reEigVals.length; i++) {
	    		eigVals[i] = Math.sqrt(reEigVals[i] * reEigVals[i] + imEigVals[i] * imEigVals[i]);
	    		if(maxVal < eigVals[i]) {
	    			maxVal = eigVals[i];
	    		}
	    	}
	    	if(maxVal < 0.001) {
	    		throw new MatrixException("maxVal < 0.001");
	    	}
	    	internal = internal.times(1/maxVal);
    	} catch(Exception e) {
    		logger.debug(e);
    		e.printStackTrace();
    	}
    	return internal;
    }

    /**
     * Method for training an ESN in batch mode, that is all the inputs are fed into the network and
     * the training is done as a batch and not in a single step mode.
     * 
     * @param input
     * @param output
     * @param nForgetPoints
     * @return
     */
    public boolean batchTraining(double[][] input, double[][] output, int nForgetPoints) {
    	switch(learningMode) {
    	case OFFLINE:
    		/* input and output each represent a single time series in an 
    		 * array of size sequence length x sequence dimension */
    		if(type == ESNType.TWI_ESN) {
    			// TODO For time-warping ESN calculate avgDist (train_esn.m) 
    		}
    		Matrix stateCollection = computeStateMatrix(input, output, nForgetPoints);
    		Matrix teacherCollection = computeTeacher(output, nForgetPoints);
    		outputWeights = computeWeights(stateCollection, teacherCollection, weightComputeMethod);
    		break;
    	case ONLINE:
//    		int nSampleInput = input.length;
//    		int internalInputUnits =  nInternalUnits + nInputUnits;
//    		stateCollection = new Matrix(nSampleInput, internalInputUnits);
//    		Matrix sInverse = Matrix.identity(internalInputUnits, internalInputUnits).times(1/this.RLS_delta);
//    		Matrix totalState = new Matrix(nTotalUnits, 1);
//    		Matrix internalState = new Matrix(nInternalUnits, 1);
//    		Matrix error = new Matrix(nSampleInput, 1);
//    		Matrix weights = new Matrix(nSampleInput, 1);
//    		for(int i = 0; i < nSampleInput; i++) {
//    	    	Matrix inScale = null;
//    	    	Matrix inShift = null;
//    	    	Matrix inputSeq = null;
//    	    	Matrix in = null;
//    	    	if(nInputUnits > 0) {
//    	    		inScale = diagonal(inputScaling);
//    	    		inShift = new Matrix(inputShift, nInputUnits);
//    	    		inputSeq = new Matrix(input);
//        			in = inputSeq.getMatrix(i, i, 0, nInputUnits-1);
//        			in = inScale.times(in.transpose()).plus(inShift);
//        			// Write input into totalstate
//        			totalState.setMatrix(nInternalUnits, (nInternalUnits + nInputUnits -1), 0, 0, in);
//        		}
//    		
//    	    	Matrix temp = new Matrix(nInternalUnits, nTotalUnits);
//    		
//	    		// Create a temporary feedback matrix
//	    		Matrix feedbackDiag = Matrix.identity(nOutputUnits, nOutputUnits);
//	    		for(int f = 0; f < feedbackScaling.length; f++) {
//	    			feedbackDiag.set(f, f, feedbackScaling[f]);
//	    		}
//	    		Matrix feedbackMatrix = feedbackWeights.times(feedbackDiag);
//	    		
//	    		temp.setMatrix(0, nInternalUnits - 1, 0, nInternalUnits - 1, internalWeights);
//	    		temp.setMatrix(0, nInternalUnits - 1, nInternalUnits, nInternalUnits + nInputUnits - 1, inputWeights);
//	    		temp.setMatrix(0, nInternalUnits - 1, nInternalUnits + nInputUnits, nTotalUnits - 1, feedbackMatrix);
//	    		
//	    		// Apply reservoir activation function
//	    		internalState = applyFunctionOnMatrix(temp.times(totalState), reservoirActivationFunction);
//	    		
//	    		// Create noise
//	    		Matrix noise = Matrix.random(nInternalUnits, 1).minusEquals((new Matrix(nInternalUnits, 1, 0.5))).timesEquals(noiseLevel);
//	    		// Add noise
//	    		internalState.plusEquals(noise);
//	    		
//	    		Matrix tempInpMat = new Matrix(internalState.getRowDimension() + in.getRowDimension(), internalState.getColumnDimension());
//				tempInpMat.setMatrix(0, internalState.getRowDimension() - 1, 0, internalState.getColumnDimension() - 1, internalState);
//				tempInpMat.setMatrix(internalState.getRowDimension(), internalState.getRowDimension() + in.getRowDimension() - 1, 0, in.getColumnDimension() - 1, in);
//	    		Matrix netOut = applyFunctionOnMatrix(outputWeights.times(tfeedbackWeightsempInpMat) , outputActivationFunction);
//	    		
//	    		totalState.setMatrix(0, nInternalUnits -1, 0, 0, internalState);
//	    		totalState.setMatrix(nInternalUnits, (nInternalUnits + nInputUnits -1), 0, 0, in);
//	    		totalState.setMatrix(nInternalUnits + nInputUnits, (nTotalUnits -1), 0, 0, netOut);
//    		
//	    		Matrix state = new Matrix(nInternalUnits + nInputUnits,1);
//	    		state.setMatrix(0, nInternalUnits -1, 0, 0, internalState);
//	    		state.setMatrix(nInternalUnits, (nInternalUnits + nInputUnits -1), 0, 0, in);
//	    		
//	    		stateCollection.setMatrix(i, i, 0, nInternalUnits + nInputUnits - 1, state.transpose());
//	    		
//	    		Matrix phi = state.transpose().times(sInverse);
//	    		
//	    		double denominator = (this.RLS_lambda + phi.times(state).get(0, 0));
//	    		Matrix k = phi.transpose().times(1/denominator);
//	    		
//	    		double e = this.teacherScaling[0] * output[i][0] + this.teacherShift[0] - netOut.get(0, 0);
//	    		error.set(i, 0, e * e);
//	    		outputWeights = outputWeights.plusEquals(k.times(e).transpose());
//	    		sInverse = sInverse.minus(k.times(phi)).times(1/this.RLS_lambda);
//    		}
//    		break;	
    	default:
    		System.out.println("Error during training in learning mode!");
    		logger.debug("Error during training in learning mode!");
    		return false;
    	}
    	trained = true;
    	return trained;
    }
    
    /**
     * Custom diagonal matrix 
     * @param array Array containing the elements of the diagonal
     * @return A custom diagonal matrix
     */
    public Matrix diagonal(double[] array) {
    	int matrixDimension = array.length;
    	Matrix diagonal = new Matrix(matrixDimension, matrixDimension, 0);
    	for(int t = 0; t < matrixDimension; t++) {
    		diagonal.set(t, t, array[t]);
		}
    	return diagonal;
    }
    
    /**
     * @param input
     * @param output
     * @param nForgetPoints an integer, may be negative, positive or zero. 
     * @return
     */
    private Matrix computeStateMatrix(double[][] input, double[][] output, int nForgetPoints) {
    	if((input == null) && (output == null)) {
    		// TODO throw exception
    		logger.error("Error in computeStateMatrix: two empty input args");
    		System.exit(1);
    	}
    	// See if there is teacher forcing or not
    	boolean teacherForcing = false;
    	int nDataPoints;
    	if(output == null) {
    		teacherForcing = false;
    		nDataPoints = input.length;
    	} else {
    		teacherForcing = true;
    		nDataPoints = output.length;
    	}
    	
    	// Define the required matrices
    	if(nForgetPoints >= 0) {
    		stateMatrix = new Matrix(nDataPoints - nForgetPoints, nInputUnits + nInternalUnits, 0);
    	} else {
    		stateMatrix = new Matrix(nDataPoints, nInputUnits + nInternalUnits, 0);
    	}
    	
    	// TODO Add the starting state (compute_statematrix.m)
    	
    	// TODO if nForgetPoints is negative, ramp up ESN by feeding first input |nForgetPoints| many times (compute_statematrix.m)
    	
    	flush();
		
    	// Scale and shift the input
    	
    	Matrix inputSeq = null;
    	if(nInputUnits > 0) {
    		inputSeq = new Matrix(input);
    	}
    	
    	// Start collecting reservoir states
    	int collectIndex = 0;
    	for(int i = 0; i < nDataPoints; i++) {
    		// Scale and shift the value of the input
    		Matrix in = null;
    		if(nInputUnits > 0) {
    	    	in = inputSeq.getMatrix(i, i, 0, nInputUnits-1).transpose().arrayTimes(inScale).plusEquals(inShift);
    			// Write input into totalState
    			totalState.setMatrix(nInternalUnits, (nInternalUnits + nInputUnits -1), 0, 0, in);
    		}
    		
    		// TODO Here the differences between plain, leaky1, leaky and twi_esn (compute_statematrix.m)
    		
    		// Apply reservoir activation function
    		internalState = Utils.applyFunctionOnMatrix(weightMatrix.times(totalState), reservoirActivationFunction, false);
    		
    		// Create noise
    		Matrix noise = Matrix.random(nInternalUnits, 1).minusEquals((new Matrix(nInternalUnits, 1, 0.5))).timesEquals(noiseLevel);
    		// Add noise
    		internalState.plusEquals(noise);
    		
    		// Compute Output
    		Matrix out = null;
        	Matrix outputSeq = null;
    		if(teacherForcing) {
        		outputSeq = new Matrix(output);
    			out = outputSeq.getMatrix(i, i, 0, nOutputUnits-1).transpose().arrayTimes(outScale).plus(outShift);
    		} else {
    			Matrix tempInpMat = new Matrix(internalState.getRowDimension() + in.getRowDimension(), internalState.getColumnDimension());
    			tempInpMat.setMatrix(0, internalState.getRowDimension() - 1, 0, internalState.getColumnDimension() - 1, internalState);
    			tempInpMat.setMatrix(internalState.getRowDimension(), internalState.getRowDimension() + in.getRowDimension() - 1, 0, in.getColumnDimension() - 1, in);
    			out = Utils.applyFunctionOnMatrix(outputWeights.times(tempInpMat) , outputActivationFunction, false);
    		}
    		totalState.setMatrix(0, nInternalUnits -1, 0, 0, internalState);
    		if(nInputUnits > 0) {
    			totalState.setMatrix(nInternalUnits, (nInternalUnits + nInputUnits -1), 0, 0, in);
    		}
    		totalState.setMatrix(nInternalUnits + nInputUnits, (nTotalUnits -1), 0, 0, out);
    		if(nForgetPoints >= 0 && i >= nForgetPoints) {    			
    			stateMatrix.setMatrix(collectIndex, collectIndex, 0, nInternalUnits - 1, internalState.transpose());
    			if(nInputUnits > 0) {
    				stateMatrix.setMatrix(collectIndex, collectIndex, nInternalUnits, nInternalUnits + nInputUnits - 1, in.transpose());
    			}
    			collectIndex++;
    		} else if(nForgetPoints < 0){ // TODO Not in use yet
    			stateMatrix.setMatrix(collectIndex, collectIndex, 0, nInternalUnits - 1, internalState.transpose());
    			if(nInputUnits > 0) {
    				stateMatrix.setMatrix(collectIndex, collectIndex, nInternalUnits, nInternalUnits + nInputUnits - 1, in.transpose());
    			}
    			collectIndex++;
    		}
    	}
    	return stateMatrix;
    }
    
    private Matrix computeTeacher(double[][] output, int nForgetPoints) {
    	Matrix outputSequence = null;
    	if(nForgetPoints >= 0) {
    		Matrix originalOutputSequence = new Matrix(output);
    		int rowBegin = nForgetPoints;
    		int rowEnd = originalOutputSequence.getRowDimension() - 1;
    		int colBegin = 0;
    		int colEnd = originalOutputSequence.getColumnDimension() - 1;
    		outputSequence = originalOutputSequence.getMatrix(rowBegin, rowEnd, colBegin, colEnd);
    	}
		// Create teacher scaling diagonal matrix
		Matrix teacherScalingDiag = diagonal(teacherScaling);
    	// Create the teacher collection matrix
    	Matrix teachCollectMat = teacherScalingDiag.times(outputSequence.transpose()).transpose();
    	
        for(int r = 0; r < teachCollectMat.getRowDimension(); r++) {
        	for(int c = 0; c < teachCollectMat.getColumnDimension(); c++) {
        		teachCollectMat.set(r, c, teachCollectMat.get(r, c) + teacherShift[(int)c]); 
        	}
        }
    	return Utils.applyFunctionOnMatrix(teachCollectMat, outputActivationFunction, true);
    }
    
    public double[][] test(double[][] input, int nForgetPoints) {
    	Matrix stateCollection = computeStateMatrix(input, null, nForgetPoints);
    	Matrix outputSequence = stateCollection.times(outputWeights.transpose());
    	// Scale and shift the output sequence back to its 
    	outputSequence = Utils.applyFunctionOnMatrix(outputSequence, outputActivationFunction, false);
    	for(int r = 0; r < outputSequence.getRowDimension(); r++) {
        	for(int c = 0; c < outputSequence.getColumnDimension(); c++) {
        		outputSequence.set(r, c, outputSequence.get(r, c) - teacherShift[c]); 
        	}
        }
		for(int r = 0; r < outputSequence.getRowDimension(); r++) {
        	for(int c = 0; c < outputSequence.getColumnDimension(); c++) {
        		outputSequence.set(r, c, outputSequence.get(r, c) / teacherScaling[c]); 
        	}
        }
    	return outputSequence.getArray();
    }
    
    // TODO add support for output feedback weights
    public double[] activateInput(double[] inputVector) {
    	Matrix inputMatrix = new Matrix(inputVector, nInputUnits);
		Matrix in = inputMatrix.arrayTimes(inScale).plus(inShift);
		totalState.setMatrix(nInternalUnits, nInternalUnits + nInputUnits -1, 0, 0, in);
		internalState = Utils.applyFunctionOnMatrix(weightMatrix.times(totalState), reservoirActivationFunction, false);
		Matrix noise = Matrix.random(nInternalUnits, 1).minusEquals((new Matrix(nInternalUnits, 1, 0.5))).timesEquals(noiseLevel);
		internalState.plusEquals(noise);
		Matrix tempInpMat = new Matrix(internalState.getRowDimension() + in.getRowDimension(), internalState.getColumnDimension());
		tempInpMat.setMatrix(0, internalState.getRowDimension() - 1, 0, internalState.getColumnDimension() - 1, internalState);
		tempInpMat.setMatrix(internalState.getRowDimension(), internalState.getRowDimension() + in.getRowDimension() - 1, 0, in.getColumnDimension() - 1, in);
		Matrix netOut = Utils.applyFunctionOnMatrix(outputWeights.times(tempInpMat), outputActivationFunction, false);
		totalState.setMatrix(0, nInternalUnits -1, 0, 0, internalState);
		totalState.setMatrix(nInternalUnits, nInternalUnits + nInputUnits -1, 0, 0, in);
		totalState.setMatrix(nInternalUnits + nInputUnits, nTotalUnits -1, 0, 0, netOut);
		Matrix unscaledOutput = netOut.minus(outShift).arrayRightDivide(outScale);
		return unscaledOutput.getRowPackedCopy();
    }
    
    private Matrix computeWeights(Matrix stateCollection, Matrix teacherCollection, WeightComputeMethod weightComputeMethod) {
    	if(weightComputeMethod == WeightComputeMethod.PSEUDOINVERSE) {
    		return pseudoinverse(stateCollection, teacherCollection);
    	} else if(weightComputeMethod == WeightComputeMethod.WIENER_HOPF) {
    		return wienerHopf(stateCollection, teacherCollection);
    	} else {
    		logger.error("Error for method " + weightComputeMethod + ". No such weight compute method exists! ");
    		System.exit(1);
    		return null;
    	}
    }
    
    private Matrix wienerHopf(Matrix stateCollection, Matrix teacherCollection) {
    	try {
    		int runlength = stateCollection.getRowDimension();
    		Matrix covMat = stateCollection.transpose().times(stateCollection).times(1.0/runlength);
    		Matrix pVec = stateCollection.transpose().times(teacherCollection).times(1.0/runlength);
    		return covMat.inverse().times(pVec).transpose();
    	} catch(Exception e) {
    		e.printStackTrace();
    		return null;
    	}
    }
    
    private Matrix pseudoinverse(Matrix stateCollection, Matrix teacherCollection) {
    	try {
    		return stateCollection.inverse().times(teacherCollection).transpose();
    	} catch(Exception e) {
    		e.printStackTrace();
    		return null;
    	}
    }
    
}
