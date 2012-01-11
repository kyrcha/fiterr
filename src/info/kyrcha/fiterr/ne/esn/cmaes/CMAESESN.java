package info.kyrcha.fiterr.ne.esn.cmaes;

import info.kyrcha.fiterr.Function;
import info.kyrcha.fiterr.Utils;
import info.kyrcha.fiterr.ne.Network;

import java.util.Properties;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;

public class CMAESESN {
	
	// Network
	
	private static final double RAND_INP = 1d;
	
	private static final double RAND_INT = 1d;
	
	private int nInputUnits;
	
	private int nOutputUnits;
	
	private int nInternalUnits;
	
	protected Function reservoirActivationFunction;
	
	protected Function outputActivationFunction;
	
	private double rho = 0.85;
	
	private double noise = 0.00000001;
	
	private double density = 0.25;
	
	private double[][] wIn;
	
	private double[][] w;
	
	private Network net;
	
	// CMA-ES related
	
	/**
	 * Parent population size
	 */
	private int mu; 
	
	/**
	 * Offspring population size
	 */
	private int lambda;
	
	private int n;
	
	private Matrix B;
	
	private Matrix D;
	
	private Matrix BD;
	
	private Matrix C;
	
	private Matrix pc;
	
	private Matrix ps;
	
	private Matrix xmeanw;
	
	private static final double MIN_SIGMA = 1e-15;
	
	private double sigma;
	
	private double cc;
	
	private double ccov;
	
	private double cs;
	
	private double damp;
	
	private double chiN;
	
	private double cw;
	
	private Matrix arweights;
	
	private double[] arfitness;
	
	private double[] fitness;
	
	private int[] episodes;
	
	private Matrix arz;
	
	private Matrix arx;
	
	private int counteval = 0;
	
	public int getPopulationSize() {
		return lambda;
	}
	
	public CMAESESN(String propsfile) {
		Properties props = Utils.loadProperties(propsfile);
		nInputUnits = Integer.parseInt(props.getProperty("nInputUnits"));
		nOutputUnits = Integer.parseInt(props.getProperty("nOutputUnits"));
		nInternalUnits = Integer.parseInt(props.getProperty("nInternalUnits"));
		reservoirActivationFunction = Function.valueOf(props.getProperty("reservoir-activation-function"));
		outputActivationFunction = Function.valueOf(props.getProperty("output-activation-function"));
		rho = Double.parseDouble(props.getProperty("rho"));
		noise = Double.parseDouble(props.getProperty("noise"));
		density = Double.parseDouble(props.getProperty("density"));
		
		// Gene length
		double stepSize = Double.parseDouble(props.getProperty("step-size")); 
		n = nInputUnits * nOutputUnits + nInternalUnits * nOutputUnits;
		double[] gene = new double[n];
		for(int i = 0; i < gene.length; i++) {
			gene[i] = 0.0;
		}
		xmeanw = new Matrix(gene, n);
		
		// Initialize dynamic strategy parameters and constants
		B = Matrix.identity(n, n); 
		D = Matrix.identity(n, n); 
		BD = B.times(D); 
		C = BD.times(BD.transpose());
		pc = new Matrix(n,1); 
		ps = new Matrix(n,1);
		sigma = stepSize; 
		lambda = 4 + (int)Math.floor(3 * Math.log(n)); 
		mu = (int)Math.floor(lambda/2);
		
		// parameter setting: adaptation
		cc = 4 /(n + 4); 
		ccov = 2 / Math.pow((n + Math.pow(2, 0.5)), 2);
		cs = 4/(n+4); 
		damp = 1/cs + 1;
		
		// Initialize weights
		arweights = new Matrix(mu, 1);
		for(int i = 0; i < mu; i++) {
			arweights.set(i, 0, Math.log(((double)lambda + 1) / 2) - Math.log((double)(i+1)));
		}
		cw = arweights.norm1() / arweights.norm2();
		chiN = Math.sqrt((double)n) * (1 - (1/(4 * (double)n)) + (1/(21 * Math.pow(n , 2))));
		arfitness = new double[lambda];
		arz = new Matrix(n, lambda);
		arx = new Matrix(n, lambda);
		counteval = 0;
		
		fitness = new double[lambda];
		episodes = new int[lambda];
		
		for(int i = 0; i < lambda; i++) {
			Matrix randn = new Matrix(n, 1);
			for(int j = 0; j < n; j++) {
				randn.set(j, 0, Utils.rand.nextGaussian());
			}
			arz.setMatrix(0, n-1, i, i, randn);
			Matrix temp = xmeanw.plus(BD.times(arz.getMatrix(0, n-1, i, i)).times(sigma));
			arx.setMatrix(0, n-1, i, i, temp); 
		}
		
		// construct network
		net = new Network(nInputUnits, nInternalUnits, nOutputUnits, reservoirActivationFunction, outputActivationFunction);
		wIn = Utils.randomMatrixPlusMinus(nInternalUnits, nInputUnits, RAND_INP);
		net.setWin(wIn);
		// Initialize the reservoir
		w = new double[nInternalUnits][nInternalUnits];
		for(int i = 0; i < nInternalUnits; i++) {
			for(int j = 0; j < nInternalUnits; j++) {
				if(Utils.rand.nextDouble() < density) {
					w[i][j] = (2 * Utils.rand.nextDouble() - 1) * RAND_INT;
				} else {
					w[i][j] = 0.0;
				}
			}
		}
		Matrix internal = new Matrix(Utils.cloneMatrix(w));
		try {
			EigenvalueDecomposition evd = internal.eig();
	    	double[] reEigVals = evd.getRealEigenvalues();
	    	double[] imEigVals = evd.getImagEigenvalues();
	    	double[] eigVals = new double[nInternalUnits];
	    	double maxVal = Double.MIN_VALUE;
	    	for(int i = 0; i < reEigVals.length; i++) {
	    		eigVals[i] = Math.sqrt(reEigVals[i] * reEigVals[i] + imEigVals[i] * imEigVals[i]);
	    		if(maxVal < eigVals[i]) {
	    			maxVal = eigVals[i];
	    		}
	    	}
	    	if(maxVal < 0.001) {
	    		maxVal = 1.0;
	    	}
	    	internal = internal.times(1/maxVal);
	    	internal.times(rho);
	    	net.setW(internal.getArray());
	    	net.setNoiseLevel(noise);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void setFitness(int index, double afitness) {
		fitness[index] = afitness;
	}
	
	public void addFitness(int index, double afitness) {
		fitness[index] = afitness;
	}
	
	public double getFitness(int index) {
		return fitness[index];
	}
	
	public int getEpisodes(int index) {
		return episodes[index];
	}
	
	public void incrEpisodes(int index) {
		episodes[index]++;
	}
	
	public void setEpisodes(int index, int aepisodes) {
		episodes[index] = aepisodes;
	}
	
	public String getGenomeString(int selection) {
		double[] array = arx.getMatrix(0, n-1, selection, selection).getRowPackedCopy();
		Matrix temp = new Matrix(array, nOutputUnits);
		Matrix wout = new Matrix(nOutputUnits, nInputUnits + nInternalUnits + nOutputUnits);
		wout.setMatrix(0, nOutputUnits - 1, 0, nInputUnits + nInternalUnits - 1, temp);
		net.setWout(wout.getArrayCopy());
		return net.toString();
	}
	
	public void evolveNextGeneration() {
		// Generate and evaluate lambda offspring
		for(int i = 0; i < lambda; i++) {
			arfitness[i] = fitness[i];
			counteval++;
		}
		int[] index = new int[arfitness.length];
		for(int i = 0; i < arfitness.length; i++) {
			index[i] = i;
		}
		// Sort by fitness and compute weighted mean
		Utils.quicksort(arfitness, index);
		int[] arIndex = new int[mu];
		for(int i = 0; i < mu; i++) {
			arIndex[i] = index[i];
		}
		xmeanw = arx.getMatrix(0, n-1, arIndex).times(arweights).times(1/arweights.norm1());
		Matrix zmeanw = arz.getMatrix(0, n-1, arIndex).times(arweights).times(1/arweights.norm1());
		// Adapt covariance matrix
		pc = pc.times((1-cc)).plus(BD.times(zmeanw)).times((Math.sqrt(cc*(2-cc))*cw));
		C = C.times((1-ccov)).plus(pc.times(pc.transpose()).times(ccov));
		// adapt sigma
		ps = ps.times(1-cs).plus(B.times(zmeanw).times(Math.sqrt(cs*(2-cs))*cw));
		sigma = sigma * Math.exp((ps.norm2()-chiN)/chiN/damp);
		// Update B and D from C
		if((counteval/lambda) % (n/10) < 1) {
			// enforce symmetry
			for(int i = 1; i < C.getRowDimension(); i++) {
				for(int j = 0; j < i; j++) {
					C.set(i, j, C.get(j, i));
				}
			}
			EigenvalueDecomposition evd = C.eig();
			B = evd.getV();
			D = evd.getD();
			// limit condition of C to 1e14 + 1
			double max = Double.MIN_VALUE;
			double min = Double.MAX_VALUE;
			for(int i = 0; i < D.getRowDimension(); i++) {
				if(max < D.get(i, i)) {
					max = D.get(i, i);
				}
				if(min > D.get(i, i)) {
					min = D.get(i, i);
				}	
			}
			if(max > 1e14 * min) {
				double tmp = max/1e14 - min;
				C = C.plus(Matrix.identity(n, n).times(tmp));
				D = D.plus(Matrix.identity(n, n).times(tmp));
			}
			Matrix newD = new Matrix(D.getRowDimension(), D.getColumnDimension());
			for(int i = 0; i < D.getRowDimension(); i++) {
				newD.set(i, i, Math.sqrt(D.get(i, i)));
			}
			D = newD;
			BD = B.times(D); // for speed up only
		}

		double max = Double.MIN_VALUE;
		double min = Double.MAX_VALUE;
		for(int i = 0; i < D.getRowDimension(); i++) {
			if(max < D.get(i, i)) {
				max = D.get(i, i);
			}
			if(min > D.get(i, i)) {
				min = D.get(i, i);
			}	
		}
		// Adjust minimal step size
		int indexa = (int)Math.floor((counteval/lambda) % n);
		Matrix a = BD.getMatrix(0, BD.getRowDimension()-1, indexa, indexa);
		Matrix b = xmeanw.plus(a.times(0.2 * sigma)); 
		boolean flag = true;
		for(int i = 0; i < xmeanw.getRowDimension(); i++){
			for(int j = 0; j < xmeanw.getColumnDimension(); j++) {
				if(xmeanw.get(i, j) != b.get(i, j)) {
					flag = false;
				}
			}
		}
		if(sigma*min < MIN_SIGMA || arfitness[0] == arfitness[(int)Math.min(mu+1,lambda)] || flag)  {
			sigma = 1.4*sigma;
		}
		for(int i = 0; i < lambda; i++) {
			Matrix randn = new Matrix(n, 1);
			for(int j = 0; j < n; j++) {
				randn.set(j, 0, Utils.rand.nextGaussian());
			}
			arz.setMatrix(0, n-1, i, i, randn);
			Matrix temp = xmeanw.plus(BD.times(arz.getMatrix(0, n-1, i, i)).times(sigma));
			arx.setMatrix(0, n-1, i, i, temp); 
		}
	}

}
