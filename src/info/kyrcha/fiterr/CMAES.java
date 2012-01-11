package info.kyrcha.fiterr;

import Jama.EigenvalueDecomposition;

import Jama.Matrix;

public class CMAES {
	
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
	
	private Matrix arz;
	
	private Matrix arx;
	
	private int counteval = 0;
	
	public CMAES(double[] gene, double stepSize) {
		n = gene.length;
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
	}
	
	public double getFitness() {
		return arfitness[0];
	}
	
	public int getCounteval() {
		return counteval;
	}
	
	public void nextGen() {
		// Generate and evaluate lambda offspring
		for(int i = 0; i < lambda; i++) {
			Matrix randn = new Matrix(n, 1);
			for(int j = 0; j < n; j++) {
				randn.set(j, 0, Utils.rand.nextGaussian());
			}
			arz.setMatrix(0, n-1, i, i, randn);
			Matrix temp = xmeanw.plus(BD.times(arz.getMatrix(0, n-1, i, i)).times(sigma));
			arx.setMatrix(0, n-1, i, i, temp); 
			arfitness[i] = cigar(arx.getMatrix(0, n-1, i, i).getRowPackedCopy());
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
		System.out.println(indexa + " " + counteval + " " + lambda);
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
	}
	
	public static void main(String[] args) {
		int N = 10; // vector/solution length
		double[] x = new double[N];
		for(int i = 0; i < x.length; i++) {
			x[i] = 1.0;
		}
		CMAES cmaes = new CMAES(x, 1.0);
		int maxeval = 300 * (int)Math.pow(N + 2, 2);
		double stopfitness = 1e-10;
		// Generation loop
		while((cmaes.getFitness() > stopfitness && cmaes.getCounteval() < maxeval) || cmaes.getCounteval() == 0) {
			cmaes.nextGen();
			System.out.println(cmaes.getCounteval() + ": " + cmaes.getFitness());
		}
	}
	
	private static double cigar(double[] x) {
		double a = Math.pow(x[0], 2);
		double b = 0;
		for(int i = 1; i < x.length; i++) {
			b += Math.pow(x[i], 2);
		}
		return a + 1e6 * b;
	}

}
