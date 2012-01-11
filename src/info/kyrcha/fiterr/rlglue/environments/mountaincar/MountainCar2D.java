package info.kyrcha.fiterr.rlglue.environments.mountaincar;

import info.kyrcha.fiterr.Utils;

import java.util.Properties;
import java.util.Random;

import org.apache.log4j.Logger;

import org.rlcommunity.rlglue.codec.EnvironmentInterface;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpec;
import org.rlcommunity.rlglue.codec.taskspec.TaskSpecVRLGLUE3;
import org.rlcommunity.rlglue.codec.taskspec.ranges.DoubleRange;
import org.rlcommunity.rlglue.codec.taskspec.ranges.IntRange;
import org.rlcommunity.rlglue.codec.types.Action;
import org.rlcommunity.rlglue.codec.types.Observation;
import org.rlcommunity.rlglue.codec.types.Reward_observation_terminal;

/**
 * Unpolluted version of the mountain car problem (with no visualization).
 * 
 * @author Kyriakos Chatzidimitriou, email: kyrcha [at] issel (dot) ee (dot) auth (dot) gr
 *
 */
public class MountainCar2D implements EnvironmentInterface {
	
		/** Logger named after the class */
		private static final Logger logger = Logger.getLogger(MountainCar2D.class.getName());
	
		/** The number of actions */
		static final int numActions = 3;
	
		/** The state of the mountain car */
		protected final MountainCarState2D theState;
		
		/** A RNG */
		private Random randomGenerator = new Random();
		
		private boolean nonMarkovian = false;
	    
	    public MountainCar2D() {
			logger.trace("Mountain Car noargs constructor");
	        boolean randomStartStates = true;
	        double transitionNoise = 0.0d;
	        long randomSeed = 0l;
	        double accFactor = 0.001;
	        theState = new MountainCarState2D(randomStartStates, transitionNoise, randomSeed, nonMarkovian, accFactor);
	    }
		
	    public MountainCar2D(String fileName) {
			Properties props = Utils.loadProperties(fileName);
			logger.trace("Mountain Car parameter constructor");
	        boolean randomStartStates = Boolean.parseBoolean(props.getProperty("randomStartStates"));
	        double transitionNoise = Double.parseDouble(props.getProperty("transitionNoise"));
	        long randomSeed = Long.parseLong(props.getProperty("randomSeed"));
	        boolean nonMarkovian = Boolean.parseBoolean(props.getProperty("nonMarkovian"));
	        double accFactor = Double.parseDouble(props.getProperty("accFactor"));
	        theState = new MountainCarState2D(randomStartStates, transitionNoise, randomSeed, nonMarkovian, accFactor);
	    }
	    
	    public String env_init() {
	        TaskSpecVRLGLUE3 theTaskSpecObject = new TaskSpecVRLGLUE3();
	        theTaskSpecObject.setEpisodic();
	        theTaskSpecObject.setDiscountFactor(1.0d);
	        theTaskSpecObject.addContinuousObservation(new DoubleRange(MountainCarState2D.MIN_POSITION, MountainCarState2D.MAX_POSITION));
	        if(!nonMarkovian) {
	        	theTaskSpecObject.addContinuousObservation(new DoubleRange(MountainCarState2D.MIN_VELOCITY, MountainCarState2D.MAX_VELOCITY));
	        }
	        theTaskSpecObject.addDiscreteAction(new IntRange(0, 2));
	        theTaskSpecObject.setRewardRange(new DoubleRange(-1, 0));
	        theTaskSpecObject.setExtra("EnvName:Mountain-Car Revision:" + this.getClass().getPackage().getImplementationVersion());
	        String taskSpecString = theTaskSpecObject.toTaskSpec();
	        TaskSpec.checkTaskSpec(taskSpecString);
	        return taskSpecString;
	    }
	    
	    /**
	     * Restart the car on the mountain.  Pick a random position and velocity if
	     * randomStarts is set.
	     * 
	     * @return
	     */
	    public Observation env_start() {
	        theState.reset();
	        return makeObservation();
	    }

	    public Reward_observation_terminal env_step(Action theAction) {
	    	
	        int a = theAction.intArray[0];

	        if (a > 2 || a < 0) {
	            logger.error("Invalid action selected in mountainCar: " + a);
	            a = randomGenerator.nextInt(3);
	        }

	        theState.update(a);
	        
	        Observation obs = makeObservation();
	        
	        Reward_observation_terminal rewardObs = new Reward_observation_terminal();
	        rewardObs.setObservation(obs);
	        rewardObs.setTerminal(theState.inGoalRegion());
	        rewardObs.setReward(theState.getReward());

	        return rewardObs;
	    }

	    public void env_cleanup() {
	    	theState.reset();
	    }

	    public String env_message(String message) {
	        if(message.equals("what is your name?"))
	            return "my name is skeleton_environment, Java edition!";

	        return "I don't know how to respond to your message";
	    }
	    
	    /**
	     * Turns theState object into an observation.
	     * @return
	     */
	    protected Observation makeObservation() {
	        return theState.makeObservation();
	    }


	}

