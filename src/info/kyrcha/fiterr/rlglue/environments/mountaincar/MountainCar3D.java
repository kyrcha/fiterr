package info.kyrcha.fiterr.rlglue.environments.mountaincar;

import java.util.Properties;

import org.rlcommunity.rlglue.codec.types.Action;
import org.rlcommunity.rlglue.codec.types.Observation;
import org.rlcommunity.rlglue.codec.taskspec.*;
import org.rlcommunity.rlglue.codec.types.Reward_observation_terminal;
import org.rlcommunity.rlglue.codec.taskspec.ranges.DoubleRange;
import org.rlcommunity.rlglue.codec.taskspec.ranges.IntRange;
import org.rlcommunity.rlglue.codec.EnvironmentInterface;

import info.kyrcha.fiterr.Utils;

/**
*
* @author Kyriakos C. Chatzidimitriou
* 
*/
public class MountainCar3D implements EnvironmentInterface {

    protected MountainCarState3D theState = null;
    
    private boolean nonMarkovian = false;

      public MountainCar3D() { 
    	  theState = new MountainCarState3D();
      }
      
      public MountainCar3D(String filename) { 
    	    Properties props = Utils.loadProperties(filename);
    	    nonMarkovian = Boolean.parseBoolean(props.getProperty("nonMarkovian"));
    	    boolean randomStarts = Boolean.parseBoolean(props.getProperty("randomStartStates"));
    	    double accFactor = Double.parseDouble(props.getProperty("accFactor"));
    	    theState = new MountainCarState3D(nonMarkovian, randomStarts, accFactor);
      }

        private String makeTaskSpec() {
        TaskSpecVRLGLUE3 theTaskSpecObject = new TaskSpecVRLGLUE3();
        theTaskSpecObject.setEpisodic();
        theTaskSpecObject.setDiscountFactor(1.0d);
        
        theTaskSpecObject.addContinuousObservation(new DoubleRange(MountainCarState3D.minPosition, MountainCarState3D.maxPosition));
        theTaskSpecObject.addContinuousObservation(new DoubleRange(MountainCarState3D.minPosition, MountainCarState3D.maxPosition));
        if(!nonMarkovian) {
        	theTaskSpecObject.addContinuousObservation(new DoubleRange(MountainCarState3D.minVelocity, MountainCarState3D.maxVelocity));
        	theTaskSpecObject.addContinuousObservation(new DoubleRange(MountainCarState3D.minVelocity, MountainCarState3D.maxVelocity));
        }

        theTaskSpecObject.addDiscreteAction(new IntRange(0, 4));
        theTaskSpecObject.setRewardRange(new DoubleRange(-1, 0));
        theTaskSpecObject.setExtra("EnvName:Mountain-Car 3D:" + this.getClass().getPackage().getImplementationVersion());

        String taskSpecString = theTaskSpecObject.toTaskSpec();
        TaskSpec.checkTaskSpec(taskSpecString);

        return taskSpecString;

    }


     public String env_init() {
       
        return makeTaskSpec();

    }


     public Observation env_start() {
        if (theState.randomStarts) {
            double randStartPosition = (Utils.rand.nextDouble() * (MountainCarState3D.maxPosition + Math.abs((MountainCarState3D.minPosition))) - Math.abs(MountainCarState3D.minPosition));
            theState.xposition = randStartPosition;
            randStartPosition = (Utils.rand.nextDouble() * (MountainCarState3D.maxPosition + Math.abs((MountainCarState3D.minPosition))) - Math.abs(MountainCarState3D.minPosition));
            theState.yposition = randStartPosition;
            randStartPosition = (Utils.rand.nextDouble() * (MountainCarState3D.maxVelocity + Math.abs((MountainCarState3D.minVelocity))) - Math.abs(MountainCarState3D.minVelocity));
            theState.xvelocity = randStartPosition;
            randStartPosition = (Utils.rand.nextDouble() * (MountainCarState3D.maxVelocity + Math.abs((MountainCarState3D.minVelocity))) - Math.abs(MountainCarState3D.minVelocity));
            theState.yvelocity = randStartPosition;
        } else {
            theState.xposition = theState.defaultInitPosition;
            theState.yposition = theState.defaultInitPosition;
            theState.xvelocity = theState.defaultInitVelocity;
            theState.yvelocity = theState.defaultInitVelocity;
        }
        return makeObservation();
    }

      public Reward_observation_terminal env_step(Action theAction) {

        int a = theAction.intArray[0];

        if (a > 4 || a < 0) {
            System.err.println("Invalid action selected in mountainCar: " + a);
            a = Utils.rand.nextInt(5);
        }

        theState.update(a);
        
        Observation obs = makeObservation();
        
        Reward_observation_terminal rewardObs = new Reward_observation_terminal();
        rewardObs.setObservation(obs);
        rewardObs.setTerminal(theState.inGoalRegion());
        rewardObs.setReward(theState.getReward());

        return rewardObs;

//        return makeRewardObservation(theState.getReward(), theState.inGoalRegion());
    }

    protected Observation makeObservation() {
    	if(!nonMarkovian) {
	        Observation currentObs = new Observation(0, 4);
	        currentObs.doubleArray[0] = theState.xposition;
	        currentObs.doubleArray[1] = theState.yposition;
	        currentObs.doubleArray[2] = theState.xvelocity;
	        currentObs.doubleArray[3] = theState.yvelocity;
	        return currentObs;
    	} else {
    		Observation currentObs = new Observation(0, 2);
	        currentObs.doubleArray[0] = theState.xposition;
	        currentObs.doubleArray[1] = theState.yposition;
	        return currentObs;
    	}
    }

      public String env_message(String theMessage) {
        return null;
       
    }


    public void env_cleanup() {
       
    }

}
