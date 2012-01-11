package info.kyrcha.fiterr.ne;

import info.kyrcha.fiterr.Utils;

import java.util.ArrayList;
import java.util.Iterator;

public class MetaNEATSpecies {
	
	// Class variables
	
	// Scope: private
	
	/**
	 * Species ID number
	 */
	private int id;
	
	/**
	 * List of species genomes.
	 */
	private ArrayList<MetaNEATGenome> genomes = new ArrayList<MetaNEATGenome>();
	
	/**
	 * Bookkeeping how many times the species average fitness stayed the same.
	 */
	private int stagnationCounter = 0;
	
	/**
	 * Number of offsprings the species will have in the next generation.
	 */
	private int numberOfOffsprings;
	
	/**
	 * A permanent representative of the species.
	 */
	private MetaNEATGenome representative;
	
	/**
	 * Average fitness of the species. It is the average fitness of the 
	 * genomes adjusted fitness.
	 */
	private double averageFitness = 0;
	
	/**
	 * Average fitness in the previous generation.
	 */
	private double previousAverageFitness = Double.MIN_VALUE;
	
	// End of class variables
	
	// Class methods
	
	public MetaNEATSpecies(int aid) {
		id = aid;
	}
	
	// Scope: public
	
	/**
	 * Get method for the stagnation counter.
	 * 
	 * @return The stagnation counter
	 */
	public int getStagnationCounter() {
		return stagnationCounter;
	}
	
	/**
	 * Resets the genomes of the species to none.
	 */
	public void reset() {
		genomes = new ArrayList<MetaNEATGenome>();
		numberOfOffsprings = 0;
	}
	
	
	public void add(MetaNEATGenome g) {
		genomes.add(g);
		g.setSpecies(this);
	}
	
	public void setRepresentative(MetaNEATGenome g) {
		representative = g.clone();
	}
	
	public MetaNEATGenome getRepresentative() {
		return representative;
	}
	
	public int getSize() {
		return genomes.size();
	}
	
	public void calculateAverageFitness() {
		if(previousAverageFitness < averageFitness) {
			previousAverageFitness = averageFitness;
		}
		averageFitness = 0;
		if(getSize() > 0) {
			Iterator<MetaNEATGenome> iter = genomes.iterator();
			double totalFitness = 0.0;
			while(iter.hasNext()) {
				totalFitness += iter.next().getAdjFitness();
			}
			averageFitness = Utils.round(totalFitness / getSize(), 5);
		} else {
			Utils.fatalError("Species with 0 genomes. Should have been removed!!!");
		}
		if(previousAverageFitness >= averageFitness) {
			stagnationCounter++;
		} else {
			stagnationCounter = 0;
		}
	}
	
	public double getAverageFitness() {
		return averageFitness;
	}
	
	public int getNumberOfOffsprings() {
		return numberOfOffsprings;
	}
	
	public void setNumberOfOffsprings(int anumber) {
		numberOfOffsprings = anumber;
	}
	
	public int getId() {
		return id;
	}
	
	public MetaNEATGenome getGenome(int index) {
		return genomes.get(index);
	}
	
	public ArrayList<MetaNEATGenome> getGenomes() {
		return genomes;
	}
	
	public String toString() {
		String theString = new String();
		theString += "* Sp.ID:";
		theString += id;
		theString += "-Gens:";
		theString += genomes.size();
		theString += "-AvgFit:";
		theString += getAverageFitness();
		return theString;
	}

}
