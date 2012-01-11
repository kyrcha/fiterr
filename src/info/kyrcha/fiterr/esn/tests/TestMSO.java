package info.kyrcha.fiterr.esn.tests;


import org.apache.commons.math.stat.StatUtils;

import org.apache.log4j.Logger;

import info.kyrcha.fiterr.Utils;
import info.kyrcha.fiterr.Error;
import info.kyrcha.fiterr.esn.ESN;
import info.kyrcha.fiterr.testbeds.timeseries.MSO;
import info.kyrcha.fiterr.testbeds.timeseries.MackeyGlass;

public class TestMSO {
	
	/** Logger named after the class */
	private static final Logger logger = Logger.getLogger(TestMSO.class.getName());

	/**
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		
		MSO mso = new MSO();
		
		int W = 100; // Wash-out
		
		int T = 1000; // Training
		
		int F = 1000; // Testing
		
		int V = 1000; // Validation
		
		int J = (int)Math.floor((W + T) / (W + F)) - 1; // Testing pairs
		
		double input[] = new double[W+T+F];
		// Create the input
		for(int i = 0; i < 2100; i++) {
			input[i] = i+1;
		}
		
		double[] original = mso.createSequence(input);
		// Caclulate statistics
		double min = StatUtils.min(original);
		double max = StatUtils.max(original);
		double mean = StatUtils.mean(original);
		double var = StatUtils.variance(original, mean);
		System.out.println("Statistics");
		System.out.println("Min: " + Utils.round(min, 2) + 
				" Max: " + Utils.round(max, 2) +
				" Mean: " + Utils.round(mean,2) + 
				" Var: " + Utils.round(var,2));
		double[] sequence = new double[T+W+V];
		
		for(int i = 0; i < sequence.length; i++) {
			sequence[i] = original[i];
		}
		
		double[][] inputSequence = new double[T+W][2];
		double[][] outputSequence = new double[T+W][1];
		
		// Training input sequence 0-99/100-2999 (3000 samples)
		for(int i = 0; i < T+W; i++) {
			inputSequence[i][0] = sequence[i];
			inputSequence[i][1] = 1.0;
		}
		
		// Training output sequence 1-3000 (3000 samples)
		for(int i = 0; i < T+W; i++) {
			outputSequence[i][0] = sequence[i+1];
		}
		
		int numOfExperiments = 1;
		double[] trainNRMSE1 = new double[numOfExperiments];
		double[] trainNRMSE2 = new double[numOfExperiments];
		double[] testNRMSE = new double[numOfExperiments];
		
		String paramsFile = args[0]; // TODO add exception if a file is not present
		for(int e = 0; e < numOfExperiments; e++) {
		
			ESN esn = new ESN(paramsFile);
				
			int nForgetPoints = W;
			if(!esn.batchTraining(inputSequence, outputSequence, nForgetPoints)) {
				System.exit(1);
			}

			// Test it
			// Train errors
			double[][] predictedTrainOutput = esn.test(inputSequence, nForgetPoints);
			double[] errorsTrain = Utils.computeError(predictedTrainOutput, outputSequence, Error.NRMSE, null);
			trainNRMSE1[e] = errorsTrain[0];
			// Test and Validation errors
			esn.flush();
			double out = 0;
			double seTest = 0;
			double seTrain = 0;
			double[] sample = new double[esn.getNumberOfInputUnits()];
			for(int i = 0; i < sequence.length - 1; i++) {
				// Use real input to wash out the network
				if(i < (T+W)) {
					sample[0] = sequence[i];
				} else { // Use the output
					sample[0] = out;
				}
				sample[1] = 1;
				double[] output = esn.activateInput(sample);
				out = output[0];
				if(i >= (T+W)) {
					seTest += Math.pow((out - sequence[i+1]), 2.0);
				} else if(i >= W) {
					seTrain += Math.pow((out - sequence[i+1]), 2.0);
				}
			}
			trainNRMSE2[e] = Math.sqrt(seTrain / (T * var));
			testNRMSE[e] = Math.sqrt(seTest / (V * var));
		}
		System.out.println("Train NRMSE 1: " + StatUtils.mean(trainNRMSE1));
		System.out.println("Train NRMSE 2: " + StatUtils.mean(trainNRMSE2));
		System.out.println("Test NRMSE: " + StatUtils.mean(testNRMSE));
	}
}
