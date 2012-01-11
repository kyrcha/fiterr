package info.kyrcha.fiterr.rlglue.environments.sjs;

import info.kyrcha.fiterr.Utils;

import java.util.ArrayList;
import java.util.Iterator;
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

public class SJS implements EnvironmentInterface {
	
	private static final int JOB_TYPES = 4;
	
	private static final int SPACING = 4;
	
	private static final int INIT_JOBS = 100;
	
	private int remainingJobs = 100;
	
	private int[] jobTypeID;
	
	private int numberOfJobTypes = 0;
	
	private int numberOfStates = 0;
	
	private int numberOfActions = 0;
	
	private Random randomGenerator;
	
	ArrayList<Job> jobs;
	
	public SJS() {
		initRandom();
		numberOfJobTypes = 4;
		int[] jobTypeID = {0, 1, 2, 3};
		this.jobTypeID = jobTypeID;
		numberOfStates = numberOfJobTypes * SPACING;
		numberOfActions = numberOfJobTypes * SPACING;
	}

	public SJS(String fileName) {
		initRandom();
		Properties props = Utils.loadProperties(fileName);
		for(int i = 0; i < JOB_TYPES; i++) {
			if(Boolean.parseBoolean(props.getProperty("job.type."+(i+1)))) {
				numberOfJobTypes++;
			}
		}
		jobTypeID = new int[numberOfJobTypes];
		int counter = 0;
		for(int i = 0; i < JOB_TYPES; i++) {
			if(Boolean.parseBoolean(props.getProperty("job.type."+(i+1)))) {
				jobTypeID[counter] = i;
				counter++;
			}
		}
		numberOfStates = numberOfJobTypes * SPACING;
		numberOfActions = numberOfJobTypes * SPACING;
	}
	    
    private void initRandom() {
		randomGenerator = new Random();
		randomGenerator.setSeed(System.currentTimeMillis());
		randomGenerator.nextDouble();
		randomGenerator.nextDouble();
		randomGenerator.nextDouble();
		randomGenerator.nextDouble();
		randomGenerator.nextDouble();
    }

    /*RL GLUE METHODS*/
    public String env_init() {
    	remainingJobs = 100;
    	jobs = new ArrayList<Job>();
		for(int i = 0; i < INIT_JOBS; i++) {
			Job job = new Job(jobTypeID[randomGenerator.nextInt(numberOfJobTypes)]);
			jobs.add(job);
		}
        return makeTaskSpec();
    }

    public Observation env_start() {
        return makeObservation();
    }

    public Reward_observation_terminal env_step(Action action) {
    	int toServe = action.getInt(0);
    	int indexType = toServe / SPACING;
    	int period = toServe % SPACING;
    	int type = jobTypeID[indexType];
//    	System.out.println("Serving " + toServe + " Type: " +  type + " Period: " + period);
    	Iterator<Job> iter = jobs.iterator();
    	Job j = null;
    	boolean flag = false;
    	while(iter.hasNext() && !flag) {
    		j = iter.next();
    		if(type == j.getType() && period == j.getPeriod()) {
    			iter.remove();
    			flag = true;
    		}
    	}
    	if(!flag) {
    		j = jobs.remove(0);
    	}
    	double reward = j.getUtility();
//    	System.out.println(reward);
    	iter = jobs.iterator();
    	while(iter.hasNext()) {
    		iter.next().incrTime();
    	}
    	if(remainingJobs > 0) {
    		jobs.add(new Job(jobTypeID[randomGenerator.nextInt(numberOfJobTypes)]));
    		remainingJobs--;
    	}
        if (jobs.isEmpty()) {
            return new Reward_observation_terminal(reward, makeObservation(), 1);
        } else {
            return new Reward_observation_terminal(reward, makeObservation(), 0);
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
        Observation returnObs = new Observation(numberOfStates, 0);
        for(int i = 0; i < numberOfStates; i++) {
        	returnObs.intArray[i] = numberOfJobs(i);
//        	System.out.println(returnObs.intArray[i]);
        }
        return returnObs;
    }
    
    private int numberOfJobs(int state) {
    	int indexType = state / SPACING;
    	int type = jobTypeID[indexType];
    	int period = state % SPACING;
    	Iterator<Job> iter = jobs.iterator();
    	int counter = 0;
    	while(iter.hasNext()) {
    		Job j = iter.next();
    		if(type == j.getType() && period == j.getPeriod()) counter++; 
    	}
    	return counter;
    }

    private String makeTaskSpec() {

        TaskSpecVRLGLUE3 theTaskSpecObject = new TaskSpecVRLGLUE3();
        theTaskSpecObject.setEpisodic();
        theTaskSpecObject.setDiscountFactor(1.0d);
        for(int i = 0; i < numberOfStates; i++) {
        	theTaskSpecObject.addDiscreteObservation(new IntRange(0,100));
        }
        theTaskSpecObject.addDiscreteAction(new IntRange(0,numberOfActions-1));
        theTaskSpecObject.setRewardRange(new DoubleRange(-30000, 0));
        theTaskSpecObject.setExtra("EnvName:ServerJobScheduling");

        String newTaskSpecString = theTaskSpecObject.toTaskSpec();
        TaskSpec.checkTaskSpec(newTaskSpecString);

        return newTaskSpecString;
    }

    public static void main(String[] args){
        EnvironmentLoader L = new EnvironmentLoader(new SJS());
        L.run();
    }

}
