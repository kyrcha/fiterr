package info.kyrcha.fiterr.esn.tests;

import info.kyrcha.fiterr.Utils;
import info.kyrcha.fiterr.Error;
import info.kyrcha.fiterr.esn.ESN;
import info.kyrcha.fiterr.testbeds.timeseries.NARMA;

import org.apache.log4j.Logger;

/**
 * Test ESN in a NARMA sequence
 * 
 * @author Kyriakos C. Chatzidimitriou (EMAIL - kyrcha [at] gmail (dot) com, WEB - http://kyrcha.info)
 *
 */
public class TestNARMA {
	
	/** Logger named after the class */
	private static final Logger logger = Logger.getLogger(TestNARMA.class.getName());

	/**
	 * Test my implementation of H. Jaeger's matlab code
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		
		// Generating training data
		int sequenceLength = 1000;
		System.out.println("Generating Data...");
		System.out.println("Sequence Length " + sequenceLength);
		int systemOrder = 3; // Set the order of the NARMA equation
		NARMA narma = new NARMA(systemOrder);
		double[][] inputSequence = narma.generateNARMAInputSequence(sequenceLength);
		double[][] outputSequence = narma.generateNARMAOutputSequence(inputSequence);
		
		// Split the data into train and test
		double trainFraction = 0.5;
		double[][] inputTrainSequence = narma.split(true, trainFraction, inputSequence);
		double[][] inputTestSequence = narma.split(false, 1 - trainFraction, inputSequence);
		double[][] outputTrainSequence = narma.split(true, trainFraction, outputSequence);
		double[][] outputTestSequence = narma.split(false, 1 - trainFraction, outputSequence);
		
		// Generate the ESN
		String paramsFile = args[0]; // TODO add exception if a file is not present
		ESN esn = new ESN(paramsFile);
		int nForgetPoints = 100;
		if(!esn.batchTraining(inputTrainSequence, outputTrainSequence, nForgetPoints)) {
			logger.debug("Error in Training!");
			System.exit(1);
		}
		
		// Now that I have a trained ESN test it
		double[][] predictedTrainOutput = esn.test(inputTrainSequence, nForgetPoints);
		double[][] predictedTestOutput = esn.test(inputTestSequence, nForgetPoints);
		double[] errorsTrain = Utils.computeError(predictedTrainOutput, outputTrainSequence, Error.NRMSE, null);
		double[] errorsTest = Utils.computeError(predictedTestOutput, outputTestSequence, Error.NRMSE, null);
		for(int i = 0; i < errorsTrain.length; i++) {
			System.out.println("NRMSE: " + errorsTrain[i] + " " + errorsTest[i]);
		}	
	}

}
