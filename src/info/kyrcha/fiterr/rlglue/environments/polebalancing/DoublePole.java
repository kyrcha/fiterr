package info.kyrcha.fiterr.rlglue.environments.polebalancing;

import info.kyrcha.fiterr.Utils;

import java.util.Properties;
import java.util.Random;

import org.rlcommunity.rlglue.codec.EnvironmentInterface;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpec;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpecVRLGLUE3;
import org.rlcommunity.rlglue.codec.taskspec.ranges.DoubleRange;
import org.rlcommunity.rlglue.codec.taskspec.ranges.IntRange;
import org.rlcommunity.rlglue.codec.types.Action;
import org.rlcommunity.rlglue.codec.types.Observation;
import org.rlcommunity.rlglue.codec.types.Reward_observation_terminal;
import org.rlcommunity.rlglue.codec.util.EnvironmentLoader;

public class DoublePole implements EnvironmentInterface {

	
	
    /** Gravitational Acceleration */
	final static double GRAVITY = 9.81;
    
	/** Mass of cart */
    final static double MASSCART = 1.0;
    
    /** Mass of pole 1 */
    final static double MASSPOLE_1 = 0.1;
    
    /** Mass of pole 2 */
    final static double MASSPOLE_2 = 0.01;
    
    /** Pole length 1 */
    final static double LENGTH_1 = 0.5;	  /* actually half the pole's length */
    
    /** Pole length 2 */
    final static double LENGTH_2 = 0.05;	  /* actually half the pole's length */
    
    /** Force magnitude */
    final static double FORCE_MAG = 10.0;
    
    /** Coefficient of friction - pole hinge */
    final static double MUP = 0.000002;
    
    /** Coefficient of friction - cart on track */
    final static double MUC = 0.0005;
//    final static double MUC = 0.0000;
    
    /** Time step */
    final static double TAU = 0.01;	  /* seconds between state updates */
    
    final static double EULER_TAU = TAU / 4;

    final static double FOURTHIRDS = 4.0d / 3.0d;
    
    final static double DEFAULTLEFTCARTBOUND = -2.4;
    
    final static double DEFAULTRIGHTCARTBOUND = 2.4;
    
    final static double DEFAULTLEFTANGLEBOUND = -Math.toRadians(12.0d);
    
    final static double DEFAULTRIGHTANGLEBOUND = Math.toRadians(12.0d);
    
    double leftCartBound;
    
    double rightCartBound;
    
    double leftAngleBound;
    
    double rightAngleBound;   
    
    boolean randomStarts;
    
    private Random randomGenerator;
    
    //State variable
    
    double[] state = new double[6];
    
    double[] customStartState = new double[6];
    
    boolean customStart = false;
    
    boolean rk4;

    boolean nonMarkovian = false;

    public DoublePole() {
        leftCartBound = DEFAULTLEFTCARTBOUND;
        rightCartBound = DEFAULTRIGHTCARTBOUND;
        leftAngleBound = DEFAULTLEFTANGLEBOUND;
        rightAngleBound =  DEFAULTRIGHTANGLEBOUND;
        randomStarts = false;
        rk4 = true;
    }

    public DoublePole(String fileName) {
		Properties props = Utils.loadProperties(fileName);
		leftCartBound = Double.parseDouble(props.getProperty("leftCartBound"));
		rightCartBound = Double.parseDouble(props.getProperty("rightCartBound"));
		leftAngleBound = -Math.toRadians(Double.parseDouble(props.getProperty("leftAngleBound")));
		rightAngleBound = Math.toRadians(Double.parseDouble(props.getProperty("rightAngleBound")));
//		System.out.println(leftAngleBound + " " + rightAngleBound);
		randomStarts = Boolean.parseBoolean(props.getProperty("randomStarts"));
		rk4 = Boolean.parseBoolean(props.getProperty("rk4"));
		nonMarkovian = Boolean.parseBoolean(props.getProperty("nonMarkovian"));
		initRandom();
    }
    
    private void initRandom() {
		randomGenerator = new Random();
		randomGenerator.nextDouble();
		randomGenerator.nextDouble();
		randomGenerator.nextDouble();
    }

    /*RL GLUE METHODS*/
    public String env_init() {
        return makeTaskSpec();
    }


    public Observation env_start() {
    	if(randomStarts) {
    		state[0] = 4.8 * randomGenerator.nextDouble() - 2.4;
    		state[1] = 2 * randomGenerator.nextDouble() - 1;
    		state[2] = 0.4 * randomGenerator.nextDouble() - 0.2;
    		state[4] = 3 * randomGenerator.nextDouble() - 1.5;
    		state[3] = 0.4 * randomGenerator.nextDouble() - 0.2;
    		state[5] = 3 * randomGenerator.nextDouble() - 1.5;
    	} if(customStart) {
    		state[0] = customStartState[0];
    		state[1] = customStartState[1];
    		state[2] = customStartState[2];
    		state[3] = customStartState[3];
    		state[4] = customStartState[4];
    		state[5] = customStartState[5];
    	} else {
    		state[0] = 0;
    		state[1] = 0;
    		state[2] = 0.07; // four degrees
    		state[3] = 0;
    		state[4] = 0;
    		state[5] = 0;
    	}
        return makeObservation();
    }
    
	private static final double ML_1 = LENGTH_1 * MASSPOLE_1;
	private static double ML_2 = LENGTH_2 * MASSPOLE_2;
    
    private double[] step(double output, double[] st, double[] derivs) {
    	double force =  (output - 0.5) * FORCE_MAG * 2;
    	double costheta_1 = Math.cos(st[2]);
    	double sintheta_1 = Math.sin(st[2]);
    	double gsintheta_1 = GRAVITY * sintheta_1;
    	double costheta_2 = Math.cos(st[4]);
    	double sintheta_2 = Math.sin(st[4]);
    	double gsintheta_2 = GRAVITY * sintheta_2;
	    
    	double temp_1 = MUP * st[3] / ML_1;
    	double temp_2 = MUP * st[5] / ML_2;
    	double fi_1 = (ML_1 * st[3] * st[3] * sintheta_1) + (0.75 * MASSPOLE_1 * costheta_1 * (temp_1 + gsintheta_1));
    	double fi_2 = (ML_2 * st[5] * st[5] * sintheta_2) +	(0.75 * MASSPOLE_2 * costheta_2 * (temp_2 + gsintheta_2));
    	double mi_1 = MASSPOLE_1 * (1 - (0.75 * Math.pow(costheta_1,2)));
    	double mi_2 = MASSPOLE_2 * (1 - (0.75 * Math.pow(costheta_2,2)));
	    
	    
    	derivs[1] = (force + fi_1 + fi_2) / (mi_1 + mi_2 + MASSCART);
	    
    	derivs[3] = -0.75 * (derivs[1] * costheta_1 + gsintheta_1 + temp_1) / LENGTH_1;
    	derivs[5] = -0.75 * (derivs[1] * costheta_2 + gsintheta_2 + temp_2) / LENGTH_2;
    	
//    	double force = (output - 0.5) * FORCE_MAG * 2.0;
//    	double costheta_1 = Math.cos(st[2]);
//    	double sintheta_1 = Math.sin(st[2]);
//    	double gsintheta_1 = GRAVITY * sintheta_1;
//    	double costheta_2 = Math.cos(st[4]);
//    	double sintheta_2 = Math.sin(st[4]);
//    	double gsintheta_2 = GRAVITY * sintheta_2;
//    	double ml_1 = LENGTH_1 * MASSPOLE_1;
//    	double ml_2 = LENGTH_2 * MASSPOLE_2;
//    	double temp_1 = MUP * st[3] / ml_1;
//    	double temp_2 = MUP * st[5] / ml_2;
//    	double fi_1 = (ml_1 * st[3] * st[3] * sintheta_1) + (0.75 * MASSPOLE_1 * costheta_1 * (temp_1 + gsintheta_1));
//    	double fi_2 = (ml_2 * st[5] * st[5] * sintheta_2) + (0.75 * MASSPOLE_2 * costheta_2 * (temp_2 + gsintheta_2));
//    	double mi_1 = MASSPOLE_1 * (1 - (0.75 * costheta_1 * costheta_1));
//    	double mi_2 = MASSPOLE_2 * (1 - (0.75 * costheta_2 * costheta_2));
//    	derivs[1] = (force -( MUC * sign(st[1]))+ fi_1 + fi_2) / (mi_1 + mi_2 + MASSCART);
//    	derivs[3] = -0.75 * (derivs[1] * costheta_1 + gsintheta_1 + temp_1) / LENGTH_1;
//    	derivs[5] = -0.75 * (derivs[1] * costheta_2 + gsintheta_2 + temp_2) / LENGTH_2;
    	return derivs;
    }
    
    private double sign(double value) {
    	if(value>=0) 
    		return 1;
    	else
    		return 0;
    }

    public Reward_observation_terminal env_step(Action action) {
    	
		int i;
		double[]  dydx = new double[6];
		 
		/*random start state for long pole*/
		/*state[2]= drand48();   */
    	double output = action.getInt(0);
		    
		/*--- Apply action to the simulated cart-pole ---*/

		if(rk4) {
    		for(i=0;i<2;++i){
    		dydx[0] = state[1];
    		dydx[2] = state[3];
    		dydx[4] = state[5];
    		step(output,state,dydx);
    		rk4(output,state,dydx,state);
    		}
		} else {
    		for(i=0;i<8;++i){
    			step(output,state,dydx);
    			state[0] += EULER_TAU * dydx[0];
    			state[1] += EULER_TAU * dydx[1];
    			state[2] += EULER_TAU * dydx[2];
    			state[3] += EULER_TAU * dydx[3];
    			state[4] += EULER_TAU * dydx[4];
    			state[5] += EULER_TAU * dydx[5];
    		}
		}
    	
//    	double[] dydx = new double[6];
//    	double output = action.getInt(0);
//    	if(rk4) {
//    		for(int i = 0; i < 2; i++) {
//    			dydx = step(output, state, dydx);
//    			dydx[0] = state[1];
//    			dydx[2] = state[3];
//    			dydx[4] = state[5];
//    			double hh = TAU * 0.5;
//    			double h6 = TAU / 6.0;
//    			double[] yt = new double[6];
//    			for(int j = 0; j <=5; j++) yt[j] = state[j] + hh * dydx[j];
//    			double[] dyt = new double[6];
//    			dyt = step(output, yt, dyt);
//    			dyt[0] = yt[1];
//    			dyt[2] = yt[3];
//    			dyt[4] = yt[5];
//    			for(int j = 0; j <=5; j++) yt[j] = state[j] + hh * dyt[j];
//    			double[] dym = new double[6];
//    			dym = step(output, yt, dym);
//    			dym[0] = yt[1];
//    			dym[2] = yt[3];
//    			dym[4] = yt[5];
//    			for(int j = 0; j <=5; j++) {
//    				yt[j] = state[j] + TAU * dym[j];
//    				dym[j] += dyt[j];
//    			}
//    			dyt = step(output, yt, dyt);
//    			dyt[0] = yt[1];
//    			dyt[2] = yt[3];
//    			dyt[4] = yt[5];
//    			for(int j = 0; j <=5; j++) state[j] = state[j] + h6 * (dydx[j] + dyt[j] + 2.0 * dym[j]); 
//    			
//    		}
//    	} else {
//    		for(int i = 0; i < 8; i++) {
//    			dydx = step(output, state, dydx);
//    			state[0] += EULER_TAU * dydx[0];
//    			state[1] += EULER_TAU * dydx[1];
//    			state[2] += EULER_TAU * dydx[2];
//    			state[3] += EULER_TAU * dydx[3];
//    			state[4] += EULER_TAU * dydx[4];
//    			state[5] += EULER_TAU * dydx[5];
//    		}
//    	}
//
//        while (state[2] >= Math.PI) {
//            state[2] -= 2.0d * Math.PI;
//        }
//        while (state[2] < -Math.PI) {
//            state[2] += 2.0d * Math.PI;
//        }
//        
//        while (state[4] >= Math.PI) {
//            state[4] -= 2.0d * Math.PI;
//        }
//        while (state[4] < -Math.PI) {
//            state[4] += 2.0d * Math.PI;
//        }

        if (inFailure()) {
            return new Reward_observation_terminal(-1.0d, makeObservation(), 1);
        } else {
            return new Reward_observation_terminal(1.0d, makeObservation(), 0);
        }
    }

	void rk4( double f, double[] y, double[] dydx, double[] yout ) {
		int i;

		double hh;
		double h6;
		double[] dym = new double[6];
		double[] dyt = new double[6];
		double[] yt = new double[6];


		hh=TAU*0.5;
		h6=TAU/6.0;
		for (i=0;i<=5;i++) yt[i]=y[i]+hh*dydx[i];
		step(f,yt,dyt);
		dyt[0] = yt[1];
		dyt[2] = yt[3];
		dyt[4] = yt[5];
		for (i=0;i<=5;i++) yt[i]=y[i]+hh*dyt[i];
		step(f,yt,dym);
		dym[0] = yt[1];
		dym[2] = yt[3];
		dym[4] = yt[5];
		for (i=0;i<=5;i++) {
			yt[i]=y[i]+TAU*dym[i];
			dym[i] += dyt[i];
		}
		step(f,yt,dyt);
		dyt[0] = yt[1];
		dyt[2] = yt[3];
		dyt[4] = yt[5];
		for (i=0;i<=5;i++)
			yout[i]=y[i]+h6*(dydx[i]+dyt[i]+2.0*dym[i]);
	}
	
    public void env_cleanup() {
    	customStart = false;
    }

    public String env_message(String message) {
        if(message.startsWith("custom:")) {
        	customStart = true;
        	String values = message.substring(7);
        	String[] initialState = values.split("\\;");
        	customStartState[0] = Double.parseDouble(initialState[0]);
        	customStartState[1] = Double.parseDouble(initialState[1]);
        	customStartState[2] = Double.parseDouble(initialState[2]);
        	customStartState[3] = Double.parseDouble(initialState[3]);
        	customStartState[4] = 0;
        	customStartState[5] = 0;
            return null;
        }
        return null;
    }

    protected Observation makeObservation() {
    	if(!nonMarkovian) {
	        Observation returnObs = new Observation(0, 6);
	        returnObs.doubleArray[0] = state[0];
	        returnObs.doubleArray[1] = state[1];
	        returnObs.doubleArray[2] = state[2];
	        returnObs.doubleArray[3] = state[3];
	        returnObs.doubleArray[4] = state[4];
	        returnObs.doubleArray[5] = state[5];
	        return returnObs;
    	} else {
    		Observation returnObs = new Observation(0, 3);
	        returnObs.doubleArray[0] = state[0];
	        returnObs.doubleArray[1] = state[2];
	        returnObs.doubleArray[2] = state[4];
	        return returnObs;
    	}
    }

    private boolean inFailure() {
        if (state[0] < leftCartBound || state[0] > rightCartBound || state[2] < leftAngleBound || state[2] > rightAngleBound || state[4] < leftAngleBound || state[4] > rightAngleBound) {
//        	System.out.println("left: " + this.leftAngleBound);
//        	System.out.println("right: " + this.rightAngleBound);
//        	for(int i = 0; i < 5; i++) {
//        		System.out.print(state[i] + ";");
//        	}
//        	System.out.print("\n");
            return true;
        } /* to signal failure */
        return false;
    }

    public double getLeftCartBound() {
        return this.leftCartBound;
    }

    public double getRightCartBound() {
        return this.rightCartBound;
    }

    public double getRightAngleBound() {
        return this.rightAngleBound;
    }

    public double getLeftAngleBound() {
        return this.leftAngleBound;
    }

    private String makeTaskSpec() {

        double xMin = leftCartBound;
        double xMax = rightCartBound;

        //Dots are guesses
        double xDotMin = -6.0d;
        double xDotMax = 6.0d;
        double thetaMin1 = leftAngleBound;
        double thetaMax1 = rightAngleBound;
        double thetaDotMin1 = -6.0d;
        double thetaDotMax1 = 6.0d;
        double thetaMin2 = leftAngleBound;
        double thetaMax2 = rightAngleBound;
        double thetaDotMin2 = -6.0d;
        double thetaDotMax2 = 6.0d;

        TaskSpecVRLGLUE3 theTaskSpecObject = new TaskSpecVRLGLUE3();
        theTaskSpecObject.setEpisodic();
        theTaskSpecObject.setDiscountFactor(1.0d);
        theTaskSpecObject.addContinuousObservation(new DoubleRange(xMin, xMax));
        if(!nonMarkovian) {
        	theTaskSpecObject.addContinuousObservation(new DoubleRange(xDotMin, xDotMax));
        }
        theTaskSpecObject.addContinuousObservation(new DoubleRange(thetaMin1, thetaMax1));
        if(!nonMarkovian) {
        	theTaskSpecObject.addContinuousObservation(new DoubleRange(thetaDotMin1, thetaDotMax1));
        }
        theTaskSpecObject.addContinuousObservation(new DoubleRange(thetaMin2, thetaMax2));
        if(!nonMarkovian) {
        	theTaskSpecObject.addContinuousObservation(new DoubleRange(thetaDotMin2, thetaDotMax2));
        }
        theTaskSpecObject.addDiscreteAction(new IntRange(0, 1));
        theTaskSpecObject.setRewardRange(new DoubleRange(-1, 0));
        theTaskSpecObject.setExtra("EnvName:DoublePole");

        String newTaskSpecString = theTaskSpecObject.toTaskSpec();
        TaskSpec.checkTaskSpec(newTaskSpecString);

        return newTaskSpecString;
    }

    public static void main(String[] args){
        EnvironmentLoader L = new EnvironmentLoader(new DoublePole());
        L.run();
    }
}



