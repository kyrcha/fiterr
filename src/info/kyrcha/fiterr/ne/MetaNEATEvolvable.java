package info.kyrcha.fiterr.ne;

/**
 * All algorithms that are using the NEAT as a meta search method for neuroevolution should
 * implement this interface.
 * 
 * @author Kyriakos C. Chatzidimitriou (EMAIL - kyrcha [at] gmail (dot) com, WEB - http://kyrcha.info)
 *
 */
public interface MetaNEATEvolvable {
	
	public int getPopulationSize();
	
	public MetaNEATGenome getGenome(int index);
	
	public void evolveNextGeneration();
	
	public int getNumberOfSpecies();

}
