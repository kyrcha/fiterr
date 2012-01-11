package info.kyrcha.fiterr;

import Jama.Matrix;

public class XMatrix extends Matrix {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4323028543320862970L;

	public XMatrix(double[] vals, int m) {
		super(vals, m);
	}
	
	public XMatrix(double[][] vals) {
		super(vals);
	}
    
    public XMatrix(double[][] vals, int m, int n) {
    	super(vals, m, n);
    }
    
    public XMatrix(int m, int n) {
    	super(m, n);
    }
    
    public XMatrix(int m, int n, double s) {
    	super(m, n, s);
    }
    
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}
