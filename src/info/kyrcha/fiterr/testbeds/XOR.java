package info.kyrcha.fiterr.testbeds;

public class XOR {
	
	/** Two sensor inputs, plus a bias term */
	private static final int numberOfInputs = 2 + 1;
	
	private static final int numberOfOutputs = 1;
	
	private static final int numberOfCases = 4;
	
	private int state = 0;
	
	private double[][] inputs = {{0,0,1}, {0,1,1}, {1,0,1}, {1,1,1}};
	
	private double[][] outputs = {{0}, {1}, {1}, {0}};
	
	public int getNumberOfInputs() {
		return numberOfInputs;
	}
	
	public int getNumberOfCases() {
		return numberOfCases;
	}
	
	public int getNumberOfOutputs() {
		return numberOfOutputs;
	}
	
	public double[] input() {
		return inputs[(state)%4];
	}
	
	public double[] output() {
		return outputs[(state++)%4];
	}
	
	public static void main(String[] args) {
		XOR xor = new XOR();
		for(int i = 0; i < numberOfCases; i++) {
			System.out.print("Input: ");
			for(int j = 0; j < numberOfInputs; j++) {
				System.out.print(xor.input()[j] + " ");
			}
			System.out.print("- Output: ");
			for(int j = 0; j < numberOfOutputs; j++) {
				System.out.print(xor.output()[j] + " ");
			}
			System.out.println();
		}
	}

}
