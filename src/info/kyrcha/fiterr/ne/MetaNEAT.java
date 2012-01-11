package info.kyrcha.fiterr.ne;

import info.kyrcha.fiterr.Utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

/**
 * MetaNEAT is a class that contains the steps of NEAT as a meta-search algorithm for 
 * constructing neural networks through evolution.
 * 
 * @author Kyriakos C. Chatzidimitriou (EMAIL - kyrcha [at] gmail (dot) com, WEB - http://kyrcha.info)
 *
 */
public abstract class MetaNEAT implements MetaNEATEvolvable {
	
	// Class variables
	
	/** Global species number */
	protected int speciesNumbering = 0;
	
	/** Minimum value for compatibility threshold */
	private static final double Ct_MIN = 0.3;
	
	/** 
	 * Maximum number of generations a species is allowed 
	 * to stay the same fitness before it is removed 
	 */
	private static final int MAXIMUM_STAGNATION = 15;
	
	/** Current generation */
	protected int currGen = 0;
	
	/** Size of the population */
	protected int populationSize;
	
	/** Array to hole the population of genomes */
	protected MetaNEATGenome[] population;
	
	/** Dynamic container to hold the species */
	private ArrayList<MetaNEATSpecies> species = new ArrayList<MetaNEATSpecies>();
	
	/** The number of input neurons in the network */
	protected int nInputUnits;
	
	/** The number of output neurons in the network */
	protected int nOutputUnits;
	
	/** Compatibility threshold */
	protected double Ct = 3.0;
	
	/** Boolean variable to check if we will use compatibility mode or not */
	protected boolean useCompatMode = false;
	
	/** Variation to change the compatibility threshold */
	protected double numCompatMode = 0.3;
	
	/** The desired number of species */
	protected int numSpeciesTarget = 5;
	
	/** Percentage of organisms allowed to reproduce */
	protected double survivalThreshold = 0.2;
	
	/** Probability for doing interspecies mating */
	protected double interspeciesMatingRange = 0.001;
	
	/** Probability for just doing mutation */
	protected static double mutateOnly = 0.25;
	
	/** Probability for just doing mating */
	protected static double mateOnly = 0.2;
	
	public int getNumberOfSpecies() {
		return speciesNumbering;
	}
	
	// Class methods
	
	// Scope: private
	
	/**
	 * Adjusts compatibility thresholds
	 */
	private void adjustCompatibilityThresholds() {
		if(currGen > 1 && useCompatMode) {
			if(species.size() < numSpeciesTarget) {
				Ct -= numCompatMode;
			} else if(species.size() > numSpeciesTarget) {
				Ct += numCompatMode;
			}
			Ct = Math.max(Ct, Ct_MIN);
		}
	}
	
	/**
	 * Removes stagnated species
	 */
	private void removeStagnatedSpecies() {
		Iterator<MetaNEATSpecies> iter = species.iterator();
		while(iter.hasNext()) {
			MetaNEATSpecies tmpSpecies = iter.next();
			if(tmpSpecies.getStagnationCounter() > MAXIMUM_STAGNATION) {
				iter.remove();
			}
		}
	}
	
	/**
	 * Reset the species
	 */
	private void resetSpecies() {
		// Reset species
		Iterator<MetaNEATSpecies> iter = species.iterator();
		while(iter.hasNext()) {
			MetaNEATSpecies aspecies = iter.next();
			aspecies.reset();
		}
	}
	
	/**
	 * Cluster population into species.
	 */
	private void clusterPopulation() {
		// If no species exist create the first species and place the first genome in it
		int start = 0;
		if(species.isEmpty()) {
			MetaNEATSpecies firstSpecies = new MetaNEATSpecies(incrementSpeciesNumbering());
			firstSpecies.setRepresentative(population[start]);
			firstSpecies.add(population[start]);
			species.add(firstSpecies);
			start++;
		}
		// Cluster the population
		for(int i = start; i < population.length; i++) {
			MetaNEATGenome currGenome = population[i];
			/* Search the species and put the genome to the one that satisfies
			 * the compatibility threshold.
			 */
			boolean isPartOfExistingSpecies = false;
			Iterator<MetaNEATSpecies> iter = species.iterator();
			while(iter.hasNext()) {
				MetaNEATSpecies currSpecies = iter.next(); 
				double dist = currGenome.calculateDistance(currSpecies.getRepresentative());
				if(dist <= Ct) {
					currSpecies.add(currGenome);
					isPartOfExistingSpecies = true;
					break;
				}
			}
			// If none of the species is sufficient create a new one
			if(!isPartOfExistingSpecies) {
				MetaNEATSpecies newSpecies = new MetaNEATSpecies(incrementSpeciesNumbering());
				newSpecies.setRepresentative(currGenome);
				newSpecies.add(currGenome);
				species.add(newSpecies);
			}
		}
	}
	
	/** 
	 * Remove empty species
	 */
	private void removeEmptySpeciesOffsprings() {
		Iterator<MetaNEATSpecies> iter = species.iterator();
		while(iter.hasNext()) {
			MetaNEATSpecies tmpSpecies = iter.next();
			if(tmpSpecies.getNumberOfOffsprings() == 0) {
				iter.remove();
			}
		}
	}
	
	/**
	 * Remove empty species
	 */
	private void removeEmptySpecies() {
		Iterator<MetaNEATSpecies> iter = species.iterator();
		while(iter.hasNext()) {
			MetaNEATSpecies tmpSpecies = iter.next();
			if(tmpSpecies.getSize() == 0) {
				iter.remove();
			}
		}
	}
	
	private void calculatePopulationFitness() {
		for(int i = 0; i < population.length; i++) {
			population[i].calculateAdjFitness();
		}
	}
	
	private double calculateSpeciesFitness() {
		double totalFitness = 0.0;
		Iterator<MetaNEATSpecies> iter = species.iterator();
		while(iter.hasNext()) {
			MetaNEATSpecies tmpSpecies = iter.next();
			tmpSpecies.calculateAverageFitness();
			totalFitness += tmpSpecies.getAverageFitness();
		}
		return totalFitness;
	}
	
	private void calculateNumberOfOffsprings(double totalFitness) {
		int counter = 0;
		double remainingFitness = totalFitness;
		for(int i = 0; i < species.size(); i++) {
			MetaNEATSpecies tmpSpecies = species.get(i);
			tmpSpecies.setNumberOfOffsprings(0);
		}
		
		while(remainingFitness > 0 && counter < population.length) {
			for(int i = 0; i < species.size(); i++) {
				MetaNEATSpecies tmpSpecies = species.get(i);
				double speciesFitness = tmpSpecies.getAverageFitness();
				if(speciesFitness <= 0) {
					Utils.fatalError("Fitness of 0");
				}
				double percentage =  Utils.round(speciesFitness / (double) population.length,3);
				if(remainingFitness > 0 && counter < population.length) {
					int previousNumOfOffsprings = tmpSpecies.getNumberOfOffsprings();
					previousNumOfOffsprings++;
					tmpSpecies.setNumberOfOffsprings(previousNumOfOffsprings);
					counter++;
					remainingFitness -= percentage;
				}
			}
		}
	}
	
	private void sortSpecies() {
		List<MetaNEATSpecies> speciesList = species;
		Collections.sort(speciesList, new Comparator<MetaNEATSpecies>() {
			public int compare(MetaNEATSpecies s1, MetaNEATSpecies s2) {
				if(s1.getAverageFitness() < s2.getAverageFitness()) {
					return 1;
				} else if(s1.getAverageFitness() > s2.getAverageFitness()) {
					return -1;
				} else {
					return 0;
				}
			}
		});
	}
	
	private List<MetaNEATGenome> sortGenomes(MetaNEATSpecies species) {
		// Sort genomes based on their fitness
		List<MetaNEATGenome> genomeList = species.getGenomes(); 
		Collections.sort(genomeList, new Comparator<MetaNEATGenome>() {
			public int compare(MetaNEATGenome g1, MetaNEATGenome g2) {
				if(g1.getAdjFitness() < g2.getAdjFitness()) {
					return 1;
				} else if(g1.getAdjFitness() > g2.getAdjFitness()) {
					return -1;
				} else {
					return 0;
				}
			}
		});
		return genomeList;
	}
	
	// Scope: public
	
	/**
	 * Returns the number of input units
	 */
	public int getNInputUnits() {
		return nInputUnits;
	}
	
	/**
	 * Returns the number of input units
	 */
	public int getNOutputUnits() {
		return nOutputUnits;
	}
	
	public int getPopulationSize() {
		return populationSize;
	}
	
	public MetaNEATGenome getGenome(int index) {
		return population[index];
	}
	
	public int incrementSpeciesNumbering() {
		speciesNumbering++;
		return speciesNumbering;
	}
	
	public abstract void preproc();
	
	public abstract void cleanup();
	
	/**
	 * Evolve next generation. The algorithm is as follows:
	 * <ol>
	 * <li>Adjust compatibility threshold</li>
	 * <li>Remove stagnated species so that I don't deal with them again</li>
	 * <li>Cluster the genomes into species in order to calculate the genomes and species fitness</li>
	 * <li>Remove species that were left without any genomes</li>
	 * <li>Calculate fitness and statistics from genomes and species</li>
	 * <li>Sort species based on average fitness</li>
	 * <li>Calculate number of offsprings</li>
	 * <li>Remove species that will not produce any offsprings</li>
	 * <li>Select-Mate-Mutate</li>
	 * </ol>
	 */
	public void evolveNextGeneration() {
		// Adjust compatibility threshold
		adjustCompatibilityThresholds();
		// Remove stagnated species
		removeStagnatedSpecies();
		// Reset species
		resetSpecies();
		// Cluster genomes into species 
		clusterPopulation();
		// Remove empty species (their average fitness cannot be calculated)
		removeEmptySpecies();
		// Calculate fitness of individuals and calculate statistics
		calculatePopulationFitness();
		// Calculate fitness of species and total fitness
		double totalFitness = calculateSpeciesFitness();
		// Sort species based on their fitness 
		sortSpecies();
		// Calculate number of offsprings for each species for the next generation
		calculateNumberOfOffsprings(totalFitness);
		// Remove empty species: Were not allocated any number of offsprings due to really small average fitness! 
		removeEmptySpeciesOffsprings();
		// Cleanup
		cleanup();
		// Preprocessing handle
		preproc();
		// Selection - Mate - Mutate
		int offspringCounter = 0;
		for(int i = 0; i < species.size(); i++) {
			MetaNEATSpecies tmpSpecies = species.get(i);
			int remainingOffsprings = tmpSpecies.getNumberOfOffsprings();
			if(remainingOffsprings > 0) {
				List<MetaNEATGenome> sortedGenomes = sortGenomes(tmpSpecies);
				// Maintain the highest performing individual
				MetaNEATGenome best = sortedGenomes.get(0).clone();
				best.setSpecies(null);
				population[offspringCounter] = best;
				remainingOffsprings--;
				offspringCounter++;
				int genomesSize = sortedGenomes.size();
				// While I have not finished producing offsprings out of the current species continue
				while(remainingOffsprings > 0) {
					// Produce offsprings
					int randGenomeIdx = Utils.rand.nextInt(Math.max((int)(survivalThreshold * (double)genomesSize),1));
					MetaNEATGenome parent1 = sortedGenomes.get(randGenomeIdx);
					randGenomeIdx = Utils.rand.nextInt(Math.max((int)(survivalThreshold * (double)genomesSize),1));
					MetaNEATGenome parent2 = sortedGenomes.get(randGenomeIdx);
					// Interspecies mutation (if there are more than 1 species)
					if(Utils.rand.nextDouble() < interspeciesMatingRange && species.size() > 1) {
						// A small hack not to select the same species
						int randSpeciesIdx = Utils.rand.nextInt(species.size());
						if(species.get(randSpeciesIdx).getId() == tmpSpecies.getId()) {
							randSpeciesIdx = (randSpeciesIdx+1) % species.size();
						}
						int interspeciesSize = species.get(randSpeciesIdx).getSize();
						randGenomeIdx = Utils.rand.nextInt(Math.max((int)(survivalThreshold * (double)interspeciesSize),1));
						parent2 = species.get(randSpeciesIdx).getGenome(randGenomeIdx);
					}
					double offspringProb = Utils.rand.nextDouble();
					MetaNEATGenome offspring;
					if(offspringProb < mutateOnly) { // Perform only mutation
						offspring = parent1.clone();
						offspring.mutate();
					} else if(offspringProb < (mateOnly + mutateOnly)) { // Perform only mating
						offspring = parent1.xover(parent2); 
					} else { // Perform both
						offspring = parent1.xover(parent2);
						offspring.mutate();
					}
					offspring.setSpecies(null);
					offspring.verify();
					population[offspringCounter] = offspring; 
					remainingOffsprings--;
					offspringCounter++;
				}
			}
		}
		currGen++;
	}

}
