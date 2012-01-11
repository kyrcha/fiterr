package info.kyrcha.fiterr.ne;

import java.util.ArrayList;

import info.kyrcha.fiterr.Function;
import info.kyrcha.fiterr.Utils;

import Jama.Matrix;
import Jama.SingularValueDecomposition;

public class Network {
	
	/** Number of input units */
	private int K;
	
	/** Number of internal units */
	private int N;
	
	/** Number of output units */
	private int L;
	
	/** Output matrix: L x (K + N + L) */
	private Matrix wout;
	
	/** Input matrix: N x K */
	private Matrix win;
	
	/** Hidden matrix: N x N */
	private Matrix w;
	
	/** Backprojection matrix: N x L */
	private Matrix wback;
	
	/** Keeps the state of the hidden units */
	private Matrix internalState;
	
	/** Keep the state of the output units */
	private Matrix outputState;
	
	/** Internal activation function */
	private Function internalFunction = Function.SIGMOID;
	
	/** Output activation function */
	private Function outputFunction = Function.SIGMOID;
	
	/** Keeps the state of the input and hidden units */
	private Matrix featureVector;
	
	/** Eligibility traces */
	private Matrix eligibilityTraces;
	
	private double noiseLevel = 0;
	
	private ArrayList<Matrix> A = new ArrayList<Matrix>();
	
	private ArrayList<Matrix> b = new ArrayList<Matrix>();
	
	public void setNoiseLevel(double anoiseLevel) {
		noiseLevel = anoiseLevel;
	}
	
	public Network(int aK, int aN, int aL, Function aIntFunct, Function aOutFunct) {
		internalFunction = aIntFunct;
		outputFunction = aOutFunct;
		K = aK;
		N = aN;
		L = aL;
		wout = new Matrix(L, K + N + L);
		if(N > 0) {
			win = new Matrix(N, K);
			w = new Matrix(N, N);
			wback = new Matrix(N, L);
			internalState = new Matrix(N, 1);
		}
		outputState = new Matrix(L, 1);
		for(int i = 0; i < L; i++) {
			A.add(new Matrix(K + N, K + N));
			b.add(new Matrix(K + N, 1));
		}
	}
	
	public void setWin(int m, int n, double weight) {
		win.set(m, n, weight);
	}
	
	public void setWin(double[][] awin) {
		win = new Matrix(awin);
	}
	
	public void setWout(int m, int n, double weight) {
		wout.set(m, n, weight);
	}
	
	public void setWout(double[][] awout) {
		wout = new Matrix(awout);
	}
	
	public void setW(int m, int n, double weight) {
		w.set(m, n, weight);
	}
	
	public void setW(double[][] aw) {
		w = new Matrix(aw);
	}
	
	public void setWback(int m, int n, double weight) {
		wback.set(m, n, weight);
	}
	
	public void setWback(double[][] awback) {
		wback = new Matrix(awback);
	}
	
	public Function getHiddenLayerFunction() {
		return internalFunction;
	}
	
	public Matrix getFeatureVector() {
		return featureVector;
	}
	
	public Function getOutputFunction() {
		return outputFunction;
	}

	public int getN() {
		return N;
	}
	
	public int getK() {
		return K;
	}
	
	public int getL() {
		return L;
	}
	
	public double getWout(int m, int n) {
		return wout.get(m, n);
	}
	
	/**
	 * Flushes the state the network was in
	 */
	public void flush() {
		if(N > 0) {
			internalState = new Matrix(N, 1);
		}
		outputState = new Matrix(L, 1);
		eligibilityTraces = new Matrix(L, K + N + L);
	}
	
	public void updateLSTDMatrices(Matrix F, Matrix FP, double reward) {
		for(int i = 0; i < L; i++) {
			A.get(i).plusEquals(F.times(F.transpose().minus(FP.transpose())));
			b.get(i).plusEquals(F.times(reward));
		}
	}
	
	public void updateWeights() {
		for(int i = 0; i < L; i++) {
			SingularValueDecomposition svd = new SingularValueDecomposition(A.get(i));
			A.get(i).print(5, 4);
			Matrix Ainv = svd.getV().times(svd.getS().inverse()).times(svd.getU().transpose());
			Matrix weights = Ainv.times(b.get(i)).transpose();
			wout.setMatrix(i, i, 0, K + N - 1, weights);
		}
	}
	
	public void updateTraces(double gamma, double lambda) {
		eligibilityTraces = eligibilityTraces.timesEquals(gamma * lambda);
	}
	
	public void updateTraces(boolean accumulatedTraces, Matrix gradient, int index) {
		Matrix update = gradient.transpose().copy();
		if(accumulatedTraces) {
			update = eligibilityTraces.getMatrix(index, index, 0, K + N - 1).plus(gradient.transpose());
		}
		eligibilityTraces.setMatrix(index, index, 0, K + N - 1, update);
	}
	
	public double[] activate(double[] activation) {
		Matrix input = new Matrix(activation, K);
		if(N > 0) {
			Matrix internalActivation = win.times(input).plus(w.times(internalState)).plus(wback.times(outputState));
			internalState = Utils.applyFunctionOnMatrix(internalActivation, internalFunction, false);
    		// Add noise
			Matrix noise = Matrix.random(N, 1).minusEquals((new Matrix(N, 1, 0.5))).timesEquals(noiseLevel);
    		internalState.plusEquals(noise);
		}
		featureVector = new Matrix(K + N, 1);
		Matrix totalState = new Matrix(K + N + L, 1);
		totalState.setMatrix(0, K - 1, 0, 0, input);
		featureVector.setMatrix(0, K - 1, 0, 0, input);
		if(N > 0) {
			totalState.setMatrix(K, K + N - 1, 0, 0, internalState);
			featureVector.setMatrix(K, K + N - 1, 0, 0, internalState);
			totalState.setMatrix(K + N, K + N + L -1, 0, 0, outputState);
		}
		outputState = Utils.applyFunctionOnMatrix(wout.times(totalState), outputFunction, false);
		return outputState.getRowPackedCopy();
	}
	
	public void GDTDLearning(double learningRate, double delta) {
		wout = wout.plus(eligibilityTraces.timesEquals(learningRate * delta));
	}
	
	/**
	 * We need to transfer all the K, L, M, all the matrices, and the functions
	 */
	public String toString() {
		int capacity = internalFunction.toString().length() + 
		outputFunction.toString().length() + 3 * 3 + 
		(L * (K + N + L)) * 10 + N * K * 10 + N * N * 10 + N * L * 10;
		StringBuilder sbnet = new StringBuilder(capacity);
		sbnet.append(internalFunction);
		sbnet.append('/');
		sbnet.append(outputFunction);
		sbnet.append('/');
		sbnet.append(K);
		sbnet.append('/');
		sbnet.append(N);
		sbnet.append('/');
		sbnet.append(L);
		sbnet.append('/');
		sbnet.append(doubleArrayToString(wout.getRowPackedCopy()));
		sbnet.append('/');
		if(N > 0) {
			sbnet.append(doubleArrayToString(win.getRowPackedCopy()));
			sbnet.append('/');
			sbnet.append(doubleArrayToString(w.getRowPackedCopy()));
			sbnet.append('/');
			sbnet.append(doubleArrayToString(wback.getRowPackedCopy()));
		}
		sbnet.trimToSize();
		return sbnet.toString();
	}
	
	private StringBuilder doubleArrayToString(double[] array) {
		StringBuilder sb = new StringBuilder(array.length * 15);
		for(double x: array) {
			sb.append(x);
			sb.append(";");
		}
		sb.trimToSize();
		return sb;
	}

}
