package info.kyrcha.fiterr.ne;

public abstract class MetaNEATGenome {
	
	protected MetaNEATSpecies species;
	
	protected static final double MAX_WEIGHT = 100d;
	
	protected static final double MAX_OUTPUT = 100000d;
	
	protected double fitness;
	
	protected double adjFitness;
	
	/** The number of input units including bias */
	protected int nInputUnits = 0;
	
	/** The number of hidden units */
	protected int nInternalUnits = 0;
	
	/** The number of output units */
	protected int nOutputUnits = 0;
	
	protected int episodes = 0;
	
	// End of class variables
	
	// Class methods
	
	public abstract Network toPhenotype();
	
	public abstract int getNumberOfNodes();
	
	public abstract int getNumberOfConnections();
	
	public abstract int getActiveConnections();
	
	public abstract double calculateDistance(MetaNEATGenome agenome);
	
	public abstract MetaNEATGenome clone();
	
	public abstract String message(String astring);
	
	public abstract void mutate();
	
	public abstract MetaNEATGenome xover(MetaNEATGenome parent);
	
	public abstract void verify();
	
	public double getFitness() {
		return fitness;
	}
	
	public void setFitness(double afitness) {
		fitness = afitness;
	}
	
	public MetaNEATSpecies getSpecies() {
		return species;
	}
	
	public void setSpecies(MetaNEATSpecies aspecies) {
		species = aspecies;
	}
	
	public void calculateAdjFitness() {
		adjFitness = fitness / species.getSize();
	}
	
	public double getAdjFitness() {
		return adjFitness;
	}
	
	public void setAdjFitness(double aadjFitness) {
		adjFitness = aadjFitness;
	}
	
	public void setEpisodes(int aepisodes) {
		episodes = aepisodes;
	}
	
	public void addFitness(double afitness) {
		fitness += afitness;
	}
	
	public void incrEpisodes() {
		episodes += 1.0;
	}
	
	public int getEpisodes() {
		return episodes;
	}
	
}
