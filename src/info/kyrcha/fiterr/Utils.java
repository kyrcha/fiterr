package info.kyrcha.fiterr;

import java.io.File;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;
import java.util.Random;

import org.apache.commons.math.stat.StatUtils;

import Jama.Matrix;

/**
 * A class containing methods used throughout the project. 
 * 
 * @author Kyriakos C. Chatzidimitriou
 *
 */
public class Utils {
	
	// Class variables
	
	// Scope: public
	
	/** A factor that defines the curve of the sigmoid.  */
	private static final double SIGMOIDAL_FACTOR = -4.9;
	
	/** 
	 * Random Number Generator (RNG) for the stochastic parts of the algorithms, 
	 * seeded using the current time of the system. 
	 */
	public static Random rand = new Random(System.currentTimeMillis());
	
	// End of class variables
	
	// Class methods
	
	// Scope public
	
	/** 
	 * Sets the seed for the simulation for replicating experiments with the same random
	 * number sequence.
	 */
	public static void setRandomSeed(long seed) {
		rand.setSeed(seed);
	}

	/**
	 * Return the value of sigmoid function, given a curve shape factor and the activation.
	 * 
	 * @param activation x
	 * 
	 * @param factor defines the curve of the sigmoid
	 * 
	 * @return 1 / (1 + exp(factor * x))
	 */
	public static double sigmoid(double activation, double factor) {
		return (1.0 / ( 1.0 + Math.exp(factor * activation)));
	}
	
	/**
	 * Returns the value of the tanh function, scaled and shifted, given an activation.
	 * 
	 * @param activation x
	 * 
	 * @param div d
	 * 
	 * @param plus p
	 * 
	 * @return tanh(x)/d + p
	 */
	public static double tanhScaleShift(double activation, double div, double plus) {
		return Math.tanh(activation) / div + plus; 
	}
	
	/**
	 * Returns the value of the tanh function, scaled and shifted, given an activation.
	 * 
	 * @param activation x
	 * 
	 * @param div d
	 * 
	 * @param plus p
	 * 
	 * @return atanh(x-p)/d
	 */
	public static double atanhScaleShift(double activation, double div, double plus) {
		return atanh(activation - plus) * div;
	}
	
	/**
	 * The inverse tanh function
	 * 
	 * @param activation x
	 * 
	 * @return atanh(x)
	 */
	public static double atanh(double activation) {
		return Math.log((1 + activation) / (1 - activation)) / 2;
	}
	
    /**
     * Apply a function on a matrix. The function returns a new matrix and does not change the one in the
     * argument.
     * 
     * @param mat The application matrix
     * 
     * @param function A Function declarations "tanh", "sigmoidal" and "identity" are supported
     * 
     * @param inverse Defines whether the normal or the inverse function should be applied
     * 
     * @return The matrix after the application of the function.
     */
    public static Matrix applyFunctionOnMatrix(Matrix mat, Function f, boolean inverse) {
    	Matrix after = new Matrix(mat.getRowDimension(), mat.getColumnDimension(), 0);
    	for(int r = 0; r < mat.getRowDimension(); r++) {
        	for(int c = 0; c < mat.getColumnDimension(); c++) {
        		if(!inverse) {
        			switch(f) {
        				case TANH:		after.set(r, c, Math.tanh(mat.get(r, c)));
        								break;
        				case IDENTITY: 	after.set(r, c, mat.get(r, c));
        						   		break;
        				case TANHSS: 	after.set(r, c, Utils.tanhScaleShift(mat.get(r, c), 2, 0.5));
        						   		break;
        				case SIGMOID:	after.set(r, c, Utils.sigmoid(mat.get(r, c), SIGMOIDAL_FACTOR));
        								break;
        			}
        		} else {
        			switch(f) {
    					case TANH:		after.set(r, c, Utils.atanh(mat.get(r, c)));
    									break;
    					case IDENTITY: 	after.set(r, c, mat.get(r, c));
    									break;
    					case TANHSS: 	after.set(r, c, Utils.atanhScaleShift(mat.get(r, c), 2, 0.5));
    									break;
        			}
    			}
        	}
    	}
    	return after;
    }
	
	/**
	 * Accounts for forget points difference between estimatedOutput and correctOutput, i.e. 
	 * points in estimated output <= points in correctOutput
	 * 
	 * @param estimatedOutput
	 * 
	 * @param correctOutput
	 * 
	 * @param NRMSE
	 * 
	 * @return
	 */
	public static double[] computeError(double[][] estimatedOutput, double[][] correctOutput, Error errorType, double[] priors) {
		
		// Number of output units
		int numOfOutputs = correctOutput[0].length;
		// Number of points
		int nEstimatePoints = estimatedOutput.length;
		// Number of forget points
		int nForgetPoints = correctOutput.length - nEstimatePoints;
		// Truncate correct output to match estimated output
		double[][] truncCorrectOutput =  new double[nEstimatePoints][numOfOutputs];
		for(int i = 0; i < nEstimatePoints; i++) {
			for(int j = 0; j < numOfOutputs; j++) {
				truncCorrectOutput[i][j] = correctOutput[nForgetPoints + i][j];
			}
		}
		// Calculate variances for each output
		double[] correctVars = new double[numOfOutputs];
		for(int j = 0; j < numOfOutputs; j++) {
			double[] values = new double[nEstimatePoints];
			for(int i = 0; i < nEstimatePoints; i++) {
				values[i] = truncCorrectOutput[i][j];
			}
			correctVars[j] = StatUtils.variance(values);
		}
		// Calculate error means (squared or absolute)
		double[] errors = new double[numOfOutputs];
		double[] denom = new double[numOfOutputs];
		double[] SStot = new double[numOfOutputs];
		double[] SSerr = new double[numOfOutputs];
		for(int j = 0; j < numOfOutputs; j++) {
			errors[j] = 0.0;
			for(int i = 0; i < nEstimatePoints; i++) {
				if(errorType == Error.MAE || errorType == Error.RAE) {
					errors[j] += Math.abs(estimatedOutput[i][j] - truncCorrectOutput[i][j]);
					denom[j]  += Math.abs(truncCorrectOutput[i][j] - priors[j]);
				} else if(errorType == Error.CC) {
					SSerr[j] += Math.pow((estimatedOutput[i][j] - truncCorrectOutput[i][j]), 2.0);
					SStot[j] += Math.pow((truncCorrectOutput[i][j] - priors[j]), 2.0);
				} else {
					errors[j] += Math.pow((estimatedOutput[i][j] - truncCorrectOutput[i][j]), 2.0);
				}
			}
			if(errorType == Error.RAE) {
				errors[j] = 100 * (errors[j] / denom[j]);
			} else if(errorType == Error.CC) {
				errors[j] = 1 - (SSerr[j]/SStot[j]);
			} else {
				errors[j] /= nEstimatePoints;
			}
		}
		// Return or root and return
		if(errorType == Error.MAE || errorType == Error.RAE || errorType == Error.CC) {
			return errors;
		} else {
			if(errorType == Error.NRMSE) {
				for(int e = 0; e < numOfOutputs; e++) {
					errors[e] = Math.sqrt(errors[e] / correctVars[e]);
				}
			} else {
				for(int e = 0; e < numOfOutputs; e++) {
					errors[e] = Math.sqrt(errors[e]);
				}
			}
			return errors;
		}
	}
	
	/**
	 * Returns a random value inside the space [-magnitude, magnitude)
	 * 
	 * @param maxMagnitude m
	 * 
	 * @return uniform random value in [-m, m)
	 */
	public static double pertubation(double magnitude) {
		return rand.nextDouble() * 2 * magnitude - magnitude;
	}
	
	/**
	 * Rounding function
	 * 
	 * @param Rval x
	 * 
	 * @param Rpl d
	 * 
	 * @return the double value x rounded using d decimal places
	 */
    public static double round(double Rval, int Rpl) {
    	double p = (double)Math.pow(10,Rpl);
  	  	Rval = Rval * p;
  	  	double tmp = Math.round(Rval);
  	  	return (double)tmp/p;
    }

	public static Properties loadProperties(String fileName) {
		File file = new File(fileName);
		// Test if I can read the file
		if (!file.canRead()) {
			fatalError("Cannot read parameter file: " + fileName);
		}
		// Create and load default properties
		Properties props = new Properties();
		try {
			FileInputStream in = new FileInputStream(fileName);
			props.load(in);
			in.close();
		} catch(IOException ioe) {
			ioe.printStackTrace();
		}
		return props;
	}
	
	public static void fatalError(String message) {
		System.err.println("FATAL ERROR: " + message);
		System.exit(1);
	}
	
	public static void quicksort(double[] main, int[] index) {
	    quicksort(main, index, 0, index.length - 1);
	}

	// quicksort a[left] to a[right]
	public static void quicksort(double[] a, int[] index, int left, int right) {
	    if (right <= left) return;
	    int i = partition(a, index, left, right);
	    quicksort(a, index, left, i-1);
	    quicksort(a, index, i+1, right);
	}

	// partition a[left] to a[right], assumes left < right
	private static int partition(double[] a, int[] index, 
	int left, int right) {
	    int i = left - 1;
	    int j = right;
	    while (true) {
	        while (less(a[++i], a[right]))      // find item on left to swap
	            ;                               // a[right] acts as sentinel
	        while (less(a[right], a[--j]))      // find item on right to swap
	            if (j == left) break;           // don't go out-of-bounds
	        if (i >= j) break;                  // check if pointers cross
	        exch(a, index, i, j);               // swap two elements into place
	    }
	    exch(a, index, i, right);               // swap with partition element
	    return i;
	}

	// is x < y ?
	private static boolean less(double x, double y) {
	    return (x < y);
	}

	// exchange a[i] and a[j]
	private static void exch(double[] a, int[] index, int i, int j) {
	    double swap = a[i];
	    a[i] = a[j];
	    a[j] = swap;
	    int b = index[i];
	    index[i] = index[j];
	    index[j] = b;
	}
	
    /** Parse properties that come in tuples */
    public static double[] getMultiProps(Properties props, String key, int dimension) {
    	String defaultValue = props.getProperty(key + ".1");
    	double[] values = new double[dimension];
    	values[0] = Double.parseDouble(defaultValue);
    	for(int i = 1; i < dimension; i++) {
    		values[i] = Double.parseDouble(props.getProperty(key + "." + (i+1), defaultValue));
    	}
    	return values;
    }
    
	public static double[][] randomMatrixPlusMinus(int rows, int cols, double limit) {
		double[][] randMat = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				randMat[i][j] = (2 * Utils.rand.nextDouble() - 1) * limit;
			}	
		}
		return randMat;
	}
	
	public static boolean[][] cloneMatrix(boolean[][] mat) {
		boolean[][] newW = new boolean[mat.length][mat[0].length];
		for(int i = 0; i < mat.length; i++) {
			for(int j = 0; j < mat[i].length; j++) {
				newW[i][j] = mat[i][j];
			}
		}
		return newW;	
	}
	
	public static double[][] cloneMatrix(double[][] mat) {
		double[][] newW = new double[mat.length][mat[0].length];
		for(int i = 0; i < mat.length; i++) {
			for(int j = 0; j < mat[i].length; j++) {
				newW[i][j] = mat[i][j];
			}
		}
		return newW;	
	}

}
