package info.kyrcha.fiterr.testbeds.timeseries;

/**
 * Generates a Mackey-Glass time series using the 4th order Runge-Kutta method
 * 
 * @author kyrcha
 *
 */
public class Lorentz {
	
	private double a = 10.0;
	
	private double b = 28;
	
	private double c = 8.0d/3.0d;
	
	private double delta = 0.03;
	
	private double x0 = -1;
	
	private double y0 = 0;
	
	private double z0 = 15;
	
	public Lorentz(){}
	
	public Lorentz(double a, double b, double c, double x0, double y0, double z0, double delta) {
		this.a = a;
		this.b = b;
		this.c = c;
		this.x0 = x0;
		this.y0 = y0;
		this.z0 = z0;
		this.delta = delta;
	}
	
	public double[][] createSequence(int sample_n) {
		double x_t = x0;
		double y_t = y0;
		double z_t = z0;
		double[] X = new double[sample_n];
		double[] Y = new double[sample_n];
		double[] Z = new double[sample_n];
		for(int i = 0; i < sample_n; i++) {
			X[i] = x_t;
			Y[i] = y_t;
			Z[i] = z_t;
			double[] k1 = lorentzEq(x_t, y_t, z_t, a, b, c);
			double[] k2 = lorentzEq(x_t + 0.5 * delta * k1[0], 
					y_t + 0.5 * delta * k1[1],
					z_t + 0.5 * delta * k1[2], a, b, c);
			double[] k3 = lorentzEq(x_t + 0.5 * delta * k2[0], 
					y_t + 0.5 * delta * k2[1],
					z_t + 0.5 * delta * k2[2], a, b, c);
			double[] k4 = lorentzEq(x_t + delta * k3[0], 
					y_t + delta * k3[1],
					z_t + delta * k3[2], a, b, c);
			x_t = x_t + delta * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6.0;
			y_t = y_t + delta * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6.0;
			z_t = z_t + delta * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6.0;
		}
		double[][] XYZ = new double[3][sample_n];
		for(int i = 0; i < sample_n; i++) {
			XYZ[0][i] = X[i];
			XYZ[1][i] = Y[i];
			XYZ[2][i] = Z[i];
		}
		return XYZ;
	}
	
	public double[][] createSequence2(int sample_n) {
		double x_t = x0;
		double y_t = y0;
		double z_t = z0;
		double[] X = new double[sample_n];
		double[] Y = new double[sample_n];
		double[] Z = new double[sample_n];
		for(int i = 0; i < sample_n; i++) {
			X[i] = x_t;
			Y[i] = y_t;
			Z[i] = z_t;
			double[] k1 = lorentzEq(x_t, y_t, z_t, a, b, c);
			x_t = x_t + delta * k1[0];
			y_t = y_t + delta * k1[1];
			z_t = z_t + delta * k1[2];
		}
		double[][] XYZ = new double[3][sample_n];
		for(int i = 0; i < sample_n; i++) {
			XYZ[0][i] = X[i];
			XYZ[1][i] = Y[i];
			XYZ[2][i] = Z[i];
		}
		return XYZ;
	}
	
	private static double[] lorentzEq(double x_t, double y_t, double z_t, double a, double b, double c) {
		double[] next = new double[3];
		next[0] = a * (y_t - x_t);
		next[1] = x_t * (b - z_t) - y_t;
		next[2] = x_t * y_t - c * z_t;
		return next;
	}
	
	public static void main(String[] args) {
		Lorentz lorentz = new Lorentz();
		double[][] seq = lorentz.createSequence(1000);
		for(int i = 0; i < seq[0].length; i++) {
			System.out.println(seq[0][i]);
		}
		System.out.println("changed");
	}

}
