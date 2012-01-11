package info.kyrcha.fiterr.rlglue.agents;

import java.util.ArrayList;

import info.kyrcha.fiterr.Function;
import info.kyrcha.fiterr.Utils;

import org.rlcommunity.rlglue.codec.types.Observation;

public class NESJSAgent extends NEAgent {

	public NESJSAgent(String fileName) {
		super(fileName);
	}
	
	@Override
	protected double[] encodeInput(Observation obsv) {
    	int length = obsv.getNumInts();
//    	double[] encodedInput = new double[length + 1];
//		encodedInput[0] = 1d;
    	double[] encodedInput = new double[length];
		double denom = (double) 100 / ((double)obsv.getNumInts() / 4.0);
    	for(int i = 0; i < obsv.getNumInts(); i++) {
    		if(network.getHiddenLayerFunction().compareTo(Function.TANH) == 0) {
    			encodedInput[i] = 2 * (obsv.getInt(i)/denom) - 1.0;
    		} else {
    			encodedInput[i] = (obsv.getInt(i)/denom);
    		}
    	}
    	return encodedInput;
    }
	
	@Override
	protected int egreedy(double[] values, Observation observation) {
    	if(Utils.rand.nextDouble() < epsilon) {
    		ArrayList<Integer> indices = new ArrayList<Integer>();
    		for(int i = 0; i < observation.getNumInts(); i++) {
    			if(observation.getInt(i) != 0) {
    				indices.add(new Integer(i));
    			}
    		}
    		int randomIndex = Utils.rand.nextInt(indices.size()); 
    		return indices.get(randomIndex).intValue();
    	} else {
    		double maxValue = -100000000d;
    		int index = -1;
    		for(int i = 0; i < values.length; i++) {
//        		System.out.println(values[i] + " " + observation.getInt(i));
    			if(maxValue < values[i] && observation.getInt(i) != 0) {
    				maxValue = values[i];
    				index = i;
    			}
    		}
//    		System.out.print("\n");
    		return index;
    	}
	}

}
