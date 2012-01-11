package info.kyrcha.fiterr;

/**
 * Class to hold instances of a dataset while parsing a file
 * 
 * @author Kyriakos C. Chatzidimitriou (EMAIL - kyrcha [at] gmail (dot) com, WEB - http://kyrcha.info)
 *
 */
public class Instance {
	
	double[] instance;
	
	public Instance(int size) {
		instance = new double[size];
	}
	
	public void setInstance(String[] ainstance) {
		for(int i = 0; i < ainstance.length; i++) {
			instance[i] = Double.parseDouble(ainstance[i]); 
		}
	}
	
	public double[] getInstance() {
		return instance;
	}
	
}
