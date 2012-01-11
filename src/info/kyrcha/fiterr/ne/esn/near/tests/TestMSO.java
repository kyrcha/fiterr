package info.kyrcha.fiterr.ne.esn.near.tests;

import info.kyrcha.fiterr.Utils;

import info.kyrcha.fiterr.esn.ESN;
import info.kyrcha.fiterr.ne.MetaNEATEvolvable;
import info.kyrcha.fiterr.ne.esn.near.NEAR;
import info.kyrcha.fiterr.ne.esn.near.NEARGenome;
import info.kyrcha.fiterr.testbeds.timeseries.MSO;

import org.apache.commons.math.stat.descriptive.moment.Mean;
import org.apache.commons.math.stat.descriptive.rank.Max;
import org.apache.commons.math.stat.descriptive.rank.Min;
import org.apache.commons.math.stat.descriptive.moment.StandardDeviation;

public class TestMSO {

	public static void main(String[] args) {
		String paramsFile = args[0];
		int runs = 10;
		int generations = 100;
//		int generations = 1;
		double[][] testNRMSE = null;
		double[][] validationNRMSE = null;
		double[][] nodes = null;
		double[][] sparsity = null;
		double[][] spectral = null;
		double[][] species = null;
		for(int r = 0; r < runs; r++) {
			System.out.println("*** " + (r + 1) + " th Experiment for NEAR and MSO Begins!");
			
			MSO mso = new MSO();
			
			int W = 100; // Wash-out
			
			int T = 1000; // Training
			
			int F = 1000; // Testing
			
			int V = 1000; // Validation
			
			int J = (int)Math.floor((W + T) / (W + F)); // Testing pairs
			
			System.out.println("Testing pairs: " + J);
			
			double drive[] = new double[W+T+V];
			
			for(int i = 0; i < (W+T+V); i++) {
				drive[i] = i+1;
			}
			
			double[] original = mso.createSequence(drive);
			
			double[] uTrain = new double[T];
			double[][] inputSequence = new double[W + T][2];
			double[][] outputSequence = new double[W + T][1];
			
			Mean m = new Mean();
			Min min = new Min();
			Max max = new Max();
			
			double originalMean = m.evaluate(original);
			double originalMax = max.evaluate(original);
			double originalMin = min.evaluate(original);
			
			System.out.println(originalMean + " " + originalMax + " " + originalMin);
			
			for(int i = 0; i < original.length; i++) {
				original[i] = (original[i] - originalMean) / (originalMax - originalMin);
			}
			
			// Training input sequence 0-99/100-2999 (3000 samples)
			for(int i = 0; i < (W + T); i++) {
				inputSequence[i][0] = original[i];
				inputSequence[i][1] = 1.0;
				if(i >= W) uTrain[i - W] = original[i];
			}
			
			// Get sequence statistics
			
			double mean = m.evaluate(uTrain);
			StandardDeviation sd = new StandardDeviation();
			double std = sd.evaluate(uTrain, mean);
			
			
			// Training output sequence 1-3000 (3000 samples)
			for(int i = 0; i < (W + T); i++) {
				outputSequence[i][0] = original[i+1];
			}
	
			// Instantiate the system
			MetaNEATEvolvable near = new NEAR(paramsFile);

			testNRMSE = new double[runs][generations];
			validationNRMSE = new double[runs][generations];
			nodes = new double[runs][generations];
			sparsity = new double[runs][generations];
			spectral = new double[runs][generations];
			species = new double[runs][generations];
			
			// Evolve over generations
			for(int gen = 1; gen <= generations; gen++) {
				double minTestNRMSE = Double.MAX_VALUE;
				int individual = -1;
				for(int pop = 0; pop < near.getPopulationSize(); pop++) {
					// Train the net
					NEARGenome genome = (NEARGenome)near.getGenome(pop);
					ESN esn = genome.toESN();
					if(!esn.batchTraining(inputSequence, outputSequence, W)) {
						System.exit(1);
					}
					// Test the net and establish fitness
					double[] NRMSEs = new double[J];
					double totalNRMSEs = 0;
					for(int j = 0; j < J; j++) {
						// Drive the net using uWash
						double[] out = new double[esn.getNumberOfOutputUnits()];
						double[] input = new double[esn.getNumberOfInputUnits()];
						for(int i = (j * W + j * F); i < ((j+1) * W + j * F); i++) {
							out = esn.activateInput(inputSequence[i]);
						}
						for(int i = 0; i < F; i++) {
							input[0] = out[0];
							input[1] = 1.0;
							out = esn.activateInput(input);
							int index = ((j + 1) * W) + (j * F) + i;
							double error = out[0] - outputSequence[index][0];
							NRMSEs[j] += Math.pow(error, 2.0);
						}
						NRMSEs[j] = Math.sqrt(NRMSEs[j] / (F * std * std));
						totalNRMSEs += NRMSEs[j];
					}
					totalNRMSEs /= J;
					totalNRMSEs = Utils.round(totalNRMSEs, 10);
					if(minTestNRMSE > totalNRMSEs) {
						minTestNRMSE = totalNRMSEs;
						individual = pop;
					}
					genome.setFitness(Utils.round(1.0/totalNRMSEs, 10));
				}
				NEARGenome genome = (NEARGenome)near.getGenome(individual);
				ESN esn = genome.toESN();
				testNRMSE[r][gen-1] = minTestNRMSE;
				nodes[r][gen-1] = genome.getNInternalUnits();
				sparsity[r][gen-1] = Utils.round(genome.getSparseness(), 10);
				spectral[r][gen-1] = Utils.round(genome.getSpectralRadius(), 10);
				// Validation			
			    double[] sample = new double[esn.getNumberOfInputUnits()];
			    double out = 0;
			    double sum = 0;
			    double sum2 = 0;
			    double msq = 0;
		    	if(!esn.batchTraining(inputSequence, outputSequence, W)) {
					System.exit(1);
				}
				for(int i = 0; i < original.length - 1; i++) {
					// Use real input to wash out the network
					if(i < (W+T)) {
						sample[0] = original[i];
					} else { // Use the output
						sample[0] = out;
					}
					sample[1] = 1;
					out = esn.activateInput(sample)[0];
					double unormedOut = out;
					if(i >= (W+T)) {
//						System.out.println(out + " " + original[i+1]);
						sum += Math.pow((out - original[i+1]), 2.0);
					} else if(i >= W) {
						sum2 += Math.pow((out - original[i+1]), 2.0);
						msq += Math.pow((out - original[i+1]), 2.0);
					}
				}
				double validationError = Math.sqrt(sum/(V * std * std));
				validationNRMSE[r][gen-1] = Utils.round(validationError, 10); 
				// Print out statistics for each generation
				System.out.print("Gen: " + gen);
				System.out.print(" Test: " + testNRMSE[r][gen-1]);
				System.out.print(" Valid: " + validationNRMSE[r][gen-1]);
				System.out.print(" Nodes: " + nodes[r][gen-1]);
				System.out.print(" Sparse: " + sparsity[r][gen-1]);
				System.out.print(" Spectral: " + spectral[r][gen-1]);
				// Evolve
				near.evolveNextGeneration();
				species[r][gen-1] = Utils.round(near.getNumberOfSpecies(), 10);
				System.out.println(" Species: " + species[r][gen-1]);
			}
			System.out.println("Best validation NRMSE result for run " + r + " : " + min.evaluate(validationNRMSE[r]));
			System.out.println("Best test NRMSE result for run " + r + " : " + min.evaluate(testNRMSE[r]));
		}
		System.out.println("*************************");
		for(int r = 0; r < runs; r++) {
			for(int g = 0; g < generations; g++) {
				System.out.print(" Test: " + testNRMSE[r][g]);
				System.out.print(" Valid: " + validationNRMSE[r][g]);
				System.out.print(" Nodes: " + nodes[r][g]);
				System.out.print(" Sparse: " + sparsity[r][g]);
				System.out.print(" Spectral: " + spectral[r][g]);
				System.out.println(" Species: " + species[r][g]);
			}
		}
	}

}
