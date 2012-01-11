package info.kyrcha.fiterr.testbeds.timeseries;

public class MSO {
	
	private static final double l1 = 0.2;
	
	private static final double l2 = 0.311;
	
	private static final double l3 = 0.42;
	
	private static final double l4 = 0.51;
	
	private static final double l5 = 0.63;
	
	private static final double l6 = 0.74;
	
	private static final double l7 = 0.85;
	
	private static final double l8 = 0.97;
	
	public double[] createSequence(double[] input) {
		double[] output = new double[input.length];
		for(int i = 0; i < input.length; i++) {
			output[i] = Math.sin(l1 * input[i]) +
				Math.sin(l2 * input[i]) +
				Math.sin(l3 * input[i]) + 
				Math.sin(l4 * input[i]) + 
				Math.sin(l5 * input[i]) +
				Math.sin(l6 * input[i]) + 
				Math.sin(l7 * input[i]) + 
				Math.sin(l8 * input[i]);
		}
		return output;
	}
	

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		MSO mso = new MSO();
		double[] in = new double[1000];
		for(int i = 0; i < in.length; i++) {
			in[i] = i+1;
		}
		double[] out = mso.createSequence(in);
		for(int i = 0; i < in.length; i++) {
			System.out.println(out[i]);
		}
	}

}
