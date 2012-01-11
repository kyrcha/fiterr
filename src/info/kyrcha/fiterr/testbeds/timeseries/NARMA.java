package info.kyrcha.fiterr.testbeds.timeseries;

/**
 * NARMA sequence
 * 
 * @author Kyriakos C. Chatzidimitriou (EMAIL - kyrcha [at] gmail (dot) com, WEB - http://kyrcha.info)
 *
 */
public class NARMA {
	
	private int systemOrder;
	
	public NARMA(int asystemOrder) {
		systemOrder = asystemOrder;
	}
	
	public double[][] generateNARMAInputSequence(int sequenceLength) {
		double[][] inputSequence = new double[sequenceLength][2];
		for(int i = 0; i < sequenceLength; i++) {
			inputSequence[i][0] = 1.0;
			inputSequence[i][1] = Math.random();
		}
		return inputSequence;
	}
	
	public double[][] generateNARMAOutputSequence(double[][] inputSequence) {
		int sequenceLength = inputSequence.length;
		double[][] outputSequence = new double[sequenceLength][1];
		for(int i = 0; i < sequenceLength; i++) {
			outputSequence[i][0] = 0.1;
		}
		for(int i= systemOrder; i < sequenceLength; i++) {
			outputSequence[i][0] = 0.7 * inputSequence[i - systemOrder][1] + 0.1 
			+ (1 - outputSequence[i-1][0]) * outputSequence[i-1][0];
		}
		return outputSequence;
	}
	
	public double[][] split(boolean train, double percent, double[][] sequence) {
		int samplePoints = sequence.length;
		int percentPoint = (int)(percent * samplePoints);
		double[][] splitSequence;
		if(train) {
			splitSequence= new double[percentPoint][sequence[0].length];
			for(int i = 0; i < percentPoint; i++) {
				for(int j = 0; j < sequence[0].length; j++) {
					splitSequence[i][j] = sequence[i][j];
				}
			}
		} else {
			splitSequence= new double[samplePoints - percentPoint][sequence[0].length];
			for(int i = percentPoint; i < samplePoints; i++) {
				for(int j = 0; j < sequence[0].length; j++) {
					splitSequence[i-percentPoint][j] = sequence[i][j];
				}
			}
		}
		return splitSequence;
	}

}
