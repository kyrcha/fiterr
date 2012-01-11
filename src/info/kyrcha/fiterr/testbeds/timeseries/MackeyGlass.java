package info.kyrcha.fiterr.testbeds.timeseries;

import info.kyrcha.fiterr.Utils;

/**
 * Generates a Mackey-Glass time series using the 4th order Runge-Kutta method
 * 
 * @author kyrcha
 *
 */
public class MackeyGlass {
	
	private double a = 0.2;
	
	private double b = 0.1;
	
	private double tau = 17;
	
	private double x0 = 1.2;
	
	private double deltat = 1;
	
	public MackeyGlass(){
		
	}
	
	public MackeyGlass(double a, double b, double tau, double x0, double deltat) {
		this.a = a;
		this.b = b;
		this.tau = tau;
		this.x0 = x0;
		this.deltat = deltat;
	}
	
	public double[] createSequence(int sample_n, double variation) {
		x0 += variation * (Utils.rand.nextDouble() - 0.5);
		double time = 0.0;
		int index = 0;
		int historyLength = (int)Math.floor(tau / deltat);
		double[] xHistory = new double[historyLength];
		double x_t = x0;
		double[] X = new double[sample_n + 1];
		double[] T = new double[sample_n + 1];
		double x_t_minus_tau;
		double x_t_plus_deltat;
		for(int i = 0; i < sample_n+1; i++) {
			X[i] = x_t;
			
			if(tau == 0.0) {
				x_t_minus_tau = 0.0;
			} else {
				x_t_minus_tau = xHistory[index];
			}
			
			x_t_plus_deltat = mackeyglassRK4(x_t, x_t_minus_tau, deltat, a, b);
			
			if(tau != 0) {
				xHistory[index] = x_t_plus_deltat;
				index = (index % (historyLength - 1)) + 1;
			}
			
			time = time + deltat;
			T[i] = time;
			x_t = x_t_plus_deltat;
			
		}
		return X;
	}
	
	
	public static void main(String[] args) {
		MackeyGlass mg = new MackeyGlass();
		double[] seq = mg.createSequence(1000, 0.0);
		for(int i = 0; i < seq.length; i++) {
			System.out.println(seq[i]);
		}
	}
	
	private static double mackeyglassRK4(double x_t, double x_t_minus_tau, double deltat, double a, double b) {
		double k1 = deltat * mackeyglassEq(x_t, x_t_minus_tau, a, b);
		double k2 = deltat * mackeyglassEq(x_t + 0.5 * k1, x_t_minus_tau, a, b);
		double k3 = deltat * mackeyglassEq(x_t + 0.5 * k2, x_t_minus_tau, a, b);
		double k4 = deltat * mackeyglassEq(x_t + k3, x_t_minus_tau, a, b);
		return x_t + k1/6 + k2/3 + k3/3 + k4/6;
	}
	
	private static double mackeyglassEq(double x_t, double x_t_minus_tau, double a, double b) {
		return (-b * x_t) + ((a * x_t_minus_tau) / (1 + Math.pow(x_t_minus_tau, 10.0)));
	}

}
