/*
Copyright 2007 Brian Tanner
http://rl-library.googlecode.com/
brian@tannerpages.com
http://brian.tannerpages.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 */
package info.kyrcha.fiterr.rlglue.environments.mountaincar;

import java.util.Random;
import org.rlcommunity.rlglue.codec.types.Observation;

/**
 * This class manages all of the problem parameters, current state variables, 
 * and state transition and reward dynamics.
 *
 * @author btanner
 */
public class MountainCarState2D {

	//	Current State Information

    private double position;
    private double velocity;
    
    //	Some of these are fixed.  This environment would be easy to parameterize further by changing these.
    
    final public static double MIN_POSITION = -1.2;
    final public static double MAX_POSITION = 0.5;
    final public static double MIN_VELOCITY = -0.07; // Original
    final public static double MAX_VELOCITY = 0.07; // Original
    
    final public static double GOAL_POSITION = 0.5;
//    final public static double ACCELERATION_FACTOR = 0.001; //Original
    final public static double GRAVITY_FACTOR = -0.0025; // Original
    final public static double HILL_PEAK_FREQUENCY = 3.0;
    
    //This is the middle of the valley (no slope)
    
    final public static double DEFAULT_INIT_POSITION = -0.5d;
    final public static double DEFAULT_INIT_VELOCITY= 0.0d;
    final public static double REWARD_PER_STEP = -1.0d;
    final public static double REWARD_AT_GOAL = 0.0d;
    final private Random randomGenerator;
    
    //These are configurable
    
    private boolean randomStarts = true;
    private double transitionNoise = 0.0d;
    private int lastAction = 0;
    private boolean nonMarkovian = false;
    private double accelerationFactor = 0.001;

    public MountainCarState2D(boolean randomStartStates, double transitionNoise, long randomSeed, boolean anonMarkovian
    		, double accFactor) {
        this.randomStarts = randomStartStates;
        this.transitionNoise = transitionNoise;
        nonMarkovian = anonMarkovian;
        accelerationFactor = accFactor;

        if (randomSeed == 0) {
            this.randomGenerator = new Random(System.currentTimeMillis());
        } else {
            this.randomGenerator = new Random(randomSeed);
        }

        // Throw away the first few because the first bits are not that random.
        randomGenerator.nextDouble();
        randomGenerator.nextDouble();
        reset();
    }

    public double getPosition() {
        return position;
    }

    public double getVelocity() {
        return velocity;
    }

    /**
     * Calculate the reward for the 
     * @return
     */
    public double getReward() {
        if (inGoalRegion()) {
            return REWARD_AT_GOAL;
        } else {
            return REWARD_PER_STEP;
        }
    }

    /**
     * IS the agent past the goal marker?
     * @return
     */
    public boolean inGoalRegion() {
        return position >= GOAL_POSITION;
    }

    protected void reset() {
        position = DEFAULT_INIT_POSITION;
        velocity = DEFAULT_INIT_VELOCITY;
        if (randomStarts) {
            //Dampened starting values
//            double randStartPosition = DEFAULT_INIT_POSITION + .25d*(randomGenerator.nextDouble() - .5d);
        	double randStartPosition = -(1.7d * randomGenerator.nextDouble()) + .5d;
            position = randStartPosition;
//            double randStartVelocity = DEFAULT_INIT_VELOCITY + .025d*(randomGenerator.nextDouble() - .5d);
            double randStartVelocity = 0.14 * (randomGenerator.nextDouble() - .5d);
            velocity = randStartVelocity;
        }
    }

    /**
     * Update the agent's velocity, threshold it, then
     * update position and threshold it.
     * @param a Should be in {0 (left), 1 (neutral), 2 (right)}
     */
    void update(int a) {
        lastAction = a;
        double acceleration = accelerationFactor;

        //Noise should be at most
        double thisNoise = 2.0d * accelerationFactor * transitionNoise * (randomGenerator.nextDouble() - .5d);

        velocity += (thisNoise+((a - 1)) * (acceleration)) + getSlope(position) * (GRAVITY_FACTOR);
        if (velocity > MAX_VELOCITY) {
            velocity = MAX_VELOCITY;
        }
        if (velocity < MIN_VELOCITY) {
            velocity = MIN_VELOCITY;
        }
        position += velocity;
        if (position > MAX_POSITION) {
            position = MAX_POSITION;
        }
        if (position < MIN_POSITION) {
            position = MIN_POSITION;
        }
        if (position == MIN_POSITION && velocity < 0) {
            velocity = 0;
        }

    }

    public int getLastAction() {
        return lastAction;
    }

    /**
     * Get the height of the hill at this position
     * @param queryPosition
     * @return
     */
    public double getHeightAtPosition(double queryPosition) {
        return -Math.sin(HILL_PEAK_FREQUENCY * (queryPosition));
    }

    /**
     * Get the slop of the hill at this position
     * @param queryPosition
     * @return
     */
    public double getSlope(double queryPosition) {
        /*The curve is generated by cos(hillPeakFrequency(x-pi/2)) so the 
         * pseudo-derivative is cos(hillPeakFrequency* x) 
         */
        return Math.cos(HILL_PEAK_FREQUENCY * queryPosition);
    }

    Observation makeObservation() {
    	if(!nonMarkovian) {
    		Observation currentObs = new Observation(0, 2);
    		currentObs.doubleArray[0] = getPosition();
    		currentObs.doubleArray[1] = getVelocity();
    		return currentObs;
    	} else {
    		Observation currentObs = new Observation(0, 1);
    		currentObs.doubleArray[0] = getPosition();
    		return currentObs;
    	}
    }
}
