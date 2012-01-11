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

public class SinglePole implements EnvironmentInterface {

    /** Gravitational Acceleration */
	final static double GRAVITY = 9.81;
    
	/** Mass of cart */
    final static double MASSCART = 1.0;
    
    /** Mass of pole */
    final static double MASSPOLE = 0.1;
    
    /** Total mass */
    final static double TOTAL_MASS = (MASSPOLE + MASSCART);
    
    /** Pole length */
    final static double LENGTH = 0.5;	  /* actually half the pole's length */

    /** Pole mass-length */
    final static double POLEMASS_LENGTH = (MASSPOLE * LENGTH);
    
    /** Force magnitude */
    final static double FORCE_MAG = 10.0;
    
    /** Time step */
    final static double TAU = 0.02;	  /* seconds between state updates */

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
    
    boolean nonMarkovian = false;
    
    //State variables
    
    double x;			/* cart position, meters */

    double x_dot;			/* cart velocity */

    double theta;			/* pole angle, radians */

    double theta_dot;		/* pole angular velocity */


    public SinglePole() {
        leftCartBound = DEFAULTLEFTCARTBOUND;
        rightCartBound = DEFAULTRIGHTCARTBOUND;
        leftAngleBound = DEFAULTLEFTANGLEBOUND;
        rightAngleBound =  DEFAULTRIGHTANGLEBOUND;
        randomStarts = false;
    }

    public SinglePole(String fileName) {
		Properties props = Utils.loadProperties(fileName);
		leftCartBound = Double.parseDouble(props.getProperty("leftCartBound"));
		rightCartBound = Double.parseDouble(props.getProperty("rightCartBound"));
		leftAngleBound = -Math.toRadians(Double.parseDouble(props.getProperty("leftAngleBound")));
		rightAngleBound = Math.toRadians(Double.parseDouble(props.getProperty("rightAngleBound")));
		randomStarts = Boolean.parseBoolean(props.getProperty("randomStarts"));
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
        x = 0.0f;
        x_dot = 0.0f;
        theta = 0.0f;
        theta_dot = 0.0f;
        return makeTaskSpec();
    }

    public Observation env_start() {
    	if(randomStarts) {
    		x = 4.8 * randomGenerator.nextDouble() - 2.4;
    		x_dot = 2 * randomGenerator.nextDouble() - 1;
    		theta = 0.4 * randomGenerator.nextDouble() - 0.2;
    		theta_dot = 3 * randomGenerator.nextDouble() - 1.5;
    	} else {
	        x = 0.0f;
	        x_dot = 0.0f;
	        theta = 0.0f;
	        theta_dot = 0.0f;
    	}
        return makeObservation();
    }

    public Reward_observation_terminal env_step(Action action) {
    	
        double xacc;
        double thetaacc;
        double force;
        double costheta;
        double sintheta;
        double temp;

        if (action.intArray[0] > 0) {
            force = FORCE_MAG;
        } else {
            force = -FORCE_MAG;
        }

        costheta = Math.cos(theta);
        sintheta = Math.sin(theta);

        temp = (force + POLEMASS_LENGTH * theta_dot * theta_dot * sintheta) / TOTAL_MASS;

        thetaacc = (GRAVITY * sintheta - costheta * temp) / (LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta / TOTAL_MASS));

        xacc = (temp - POLEMASS_LENGTH * thetaacc * costheta) / TOTAL_MASS;
        
        /*** Update the four state variables, using Euler's method. ***/
        x += TAU * x_dot;
        x_dot += TAU * xacc;
        theta += TAU * theta_dot;
        theta_dot += TAU * thetaacc;
        
        while (theta >= Math.PI) {
            theta -= 2.0d * Math.PI;
        }
        while (theta < -Math.PI) {
            theta += 2.0d * Math.PI;
        }

        if (inFailure()) {
            return new Reward_observation_terminal(-1.0d, makeObservation(), 1);
        } else {
            return new Reward_observation_terminal(1.0d, makeObservation(), 0);
            
        }
    }

    public void env_cleanup() {
    }

    public String env_message(String message) {
        if(message.equals("what is your name?"))
            return "my name is skeleton_environment, Java edition!";
        return "I don't know how to respond to your message";
    }

    protected Observation makeObservation() {
    	if(!nonMarkovian) {
	        Observation returnObs = new Observation(0, 4);
	        returnObs.doubleArray[0] = x;
	        returnObs.doubleArray[1] = x_dot;
	        returnObs.doubleArray[2] = theta;
	        returnObs.doubleArray[3] = theta_dot;
	        return returnObs;
    	} else {
	        Observation returnObs = new Observation(0, 2);
	        returnObs.doubleArray[0] = x;
	        returnObs.doubleArray[1] = theta;
	        return returnObs;
    	}
    }

    private boolean inFailure() {
        if (x < leftCartBound || x > rightCartBound || theta < leftAngleBound || theta > rightAngleBound) {
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
        double thetaMin = leftAngleBound;
        double thetaMax = rightAngleBound;
        double thetaDotMin = -6.0d;
        double thetaDotMax = 6.0d;

        TaskSpecVRLGLUE3 theTaskSpecObject = new TaskSpecVRLGLUE3();
        theTaskSpecObject.setEpisodic();
        theTaskSpecObject.setDiscountFactor(1.0d);
        theTaskSpecObject.addContinuousObservation(new DoubleRange(xMin, xMax));
    	if(!nonMarkovian) {
    		theTaskSpecObject.addContinuousObservation(new DoubleRange(xDotMin, xDotMax));
    	}
        theTaskSpecObject.addContinuousObservation(new DoubleRange(thetaMin, thetaMax));
    	if(!nonMarkovian) {
    		theTaskSpecObject.addContinuousObservation(new DoubleRange(thetaDotMin, thetaDotMax));
    	}
        theTaskSpecObject.addDiscreteAction(new IntRange(0, 1));
        theTaskSpecObject.setRewardRange(new DoubleRange(-1, 0));
        theTaskSpecObject.setExtra("EnvName:SinglePole");

        String newTaskSpecString = theTaskSpecObject.toTaskSpec();
        TaskSpec.checkTaskSpec(newTaskSpecString);

        return newTaskSpecString;
    }

    public static void main(String[] args){
        EnvironmentLoader L = new EnvironmentLoader(new SinglePole());
        L.run();
    }
}



