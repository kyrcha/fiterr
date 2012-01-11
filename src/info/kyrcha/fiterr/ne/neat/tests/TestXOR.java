package info.kyrcha.fiterr.ne.neat.tests;

import info.kyrcha.fiterr.ne.neat.NEAT;
import info.kyrcha.fiterr.ne.MetaNEATEvolvable;
import info.kyrcha.fiterr.ne.Network;
import info.kyrcha.fiterr.testbeds.XOR;

public class TestXOR {

	/**
	 * The main method testing NEAT implementation with the XOR testbed.
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		// The testbed
		XOR xor = new XOR();
		// The parameter file
		String paramsFile = args[0]; // TODO add exception if a file is not present
		// Define some the runs and the generations
		int runs = 100;
		int generations = 100;
		// Variable to keep statistics
		int[] numberOfEvals = new int[runs];
		int[] numberOfNodes = new int[runs];
		int[] numberOfActiveConnections = new int[runs];
		int fiveNodesCounter = 0;
		int totNumberOfHiddenNodes = 0;
		int activeConnections = 0;
		int countGens = 0;
		int netsEvaluated = 0;
		int solutionsFound = 0;
		// Runs
		for(int run = 0; run < runs; run++) {
			// Create new NEAT
			MetaNEATEvolvable neat = new NEAT(paramsFile);
			boolean end = false;
			// Generations
			for(int gen = 1; gen <= generations && !end; gen++) {
				double maxFitGen = Double.MIN_NORMAL;
				// Population
				for(int pop = 0; pop < neat.getPopulationSize() && !end; pop++) {
					double sum = 0.0; // For fitness calculation
					int roundedSum = 0; // For stopping criterion 
					// Check the four cases
					for(int i = 0; i < xor.getNumberOfCases(); i++) {
						Network net = neat.getGenome(pop).toPhenotype();
						double[] output = net.activate(xor.input());
						net.flush();
						double real = xor.output()[0];
						double predicted = Math.round(output[0]);
						sum += Math.abs(real - predicted);
						roundedSum += Math.abs(Math.round(real) - Math.round(predicted));
					}
					double fitness = Math.pow(xor.getNumberOfCases() - sum, 2.0);
					if(maxFitGen < fitness) {
						maxFitGen = fitness;
					}
					neat.getGenome(pop).setFitness(fitness);
					numberOfEvals[run]++;
					// Found a solution
					if(roundedSum == 0) {
						end = true;
						// Keep stats
						numberOfNodes[run] = neat.getGenome(pop).getNumberOfNodes();
						numberOfActiveConnections[run] = neat.getGenome(pop).getActiveConnections();
						System.out.print("Solution found! Generation: " + gen);
						System.out.println("Nodes: " + numberOfNodes[run]);
						System.out.println("ActConns: " + numberOfActiveConnections[run]);
						System.out.println("Run: " + run);
						System.out.println(neat.getGenome(pop));
						if(neat.getGenome(pop).getNumberOfNodes() == 5) {
							fiveNodesCounter++;
						}
						totNumberOfHiddenNodes += neat.getGenome(pop).getNumberOfNodes() - 4;
						activeConnections += neat.getGenome(pop).getActiveConnections();
						if(neat.getGenome(pop).getNumberOfNodes() == 5) {
							if(neat.getGenome(pop).getActiveConnections() > 7 ) {
								System.out.println("********" + neat.getGenome(pop).toString());
							}
						}
						countGens += gen;
						netsEvaluated += numberOfEvals[run];
						solutionsFound++;
					}
				}// End population loop
				neat.evolveNextGeneration();
				
			}// End generation loop
		}// End run loop
		int counter = 0;
		for(int i = 0; i < runs; i++) {
			if(numberOfNodes[i] != 0) {
			System.out.println("Counter: " + (++counter) +
					" Run: " + (i+1) +
					" Evaluations: " + numberOfEvals[i] + 
					" Nodes: " + numberOfNodes[i] +
					" Connections: " + numberOfActiveConnections[i]);
			}
		}
		System.out.println("Solutions Found: " + solutionsFound);
		System.out.println("Perfect: " + fiveNodesCounter);
		System.out.println("AVG of Hidden: " + (double)totNumberOfHiddenNodes / (double)counter);
		System.out.println("AVG of Act. Conne: " + (double)activeConnections / (double)counter);
		System.out.println("AVG of generations: " + (double)countGens / (double)counter);
		System.out.println("AVG of Net Eval: " + (double)netsEvaluated / (double)counter);

	}

}
