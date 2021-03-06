package info.kyrcha.fiterr.rlglue.environments.mountaincar;

/**
 *
 * @author Kyriakos C. Chatzidimitriou
 */
public class MountainCarState3D {
	
	//	Current State Information
    double xposition;
    double xvelocity;
    double yposition;
    double yvelocity;


    // Some of these are fixed.  This environment would be easy to parameterize further by changing these.
    public static final double minPosition = -1.2;
    public static final double maxPosition = 0.6;
    public static final double minVelocity = -0.07; // Original
    public static final double maxVelocity = 0.07; // Original

    public double goalPosition = 0.5;
    public double accelerationFactor = 0.001; // Original
    public double gravityFactor = -0.0025; // Original
    public double hillPeakFrequency = 3.0;

    //This is the middle of the valley (no slope)
    public double defaultInitPosition = -0.5d;
    public double defaultInitVelocity = 0.0d;
    public double rewardPerStep = -1.0d;
    public double rewardAtGoal = 0.0d;

    //These are configurable
    public boolean randomStarts = true;
    
    private boolean nonMarkovian = false;

    public MountainCarState3D() { }
    
    public MountainCarState3D(boolean anonMarkovian, boolean arandomStarts, double accFactor) {
    	nonMarkovian = anonMarkovian;
    	randomStarts = arandomStarts;
    	accelerationFactor = accFactor; 
    }

    /**
     * Calculate the reward for the
     * @return
     */
    public double getReward() {
        if (inGoalRegion()) {
            return rewardAtGoal;
        } else {
            return rewardPerStep;
        }
    }


    /**
     * IS the agent past the goal marker?
     * @return
     */
    public boolean inGoalRegion() {
        if(xposition >= goalPosition && yposition >= goalPosition)
            return true;
        return false;
    }

   /**
    * Update the agent's velocity, threshold it, then
    * update position and threshold it.
    * @param a Should be in {0 (coast), 1 (left/west), 2 (right/east), 3 (down/south), 4 (up/north)}
    */
    void update(int a) {
        switch (a) {
            case 0:
                xvelocity += getSlope(xposition) * (gravityFactor);
                yvelocity += getSlope(yposition) * (gravityFactor);
                break;
            case 1:
                xvelocity += -accelerationFactor + getSlope(xposition) * (gravityFactor);
                yvelocity += getSlope(yposition) * (gravityFactor);
                break;
            case 2:
                xvelocity += +accelerationFactor + getSlope(xposition) * (gravityFactor);
                yvelocity += getSlope(yposition) * (gravityFactor);
                break;
            case 3:
                xvelocity += getSlope(xposition) * (gravityFactor);
                yvelocity += -accelerationFactor + getSlope(yposition) * (gravityFactor);
                break;
            case 4:
                xvelocity += getSlope(xposition) * (gravityFactor);
                yvelocity += +accelerationFactor + getSlope(yposition) * (gravityFactor);
                break;
        }


        if (xvelocity > maxVelocity) {
            xvelocity = maxVelocity;
        } else if (xvelocity < minVelocity) {
            xvelocity = minVelocity;
        }
        if (yvelocity > maxVelocity) {
            yvelocity = maxVelocity;
        } else if (yvelocity < minVelocity) {
            yvelocity = minVelocity;
        }


        xposition += xvelocity;
        yposition += yvelocity;

        if (xposition > maxPosition) {
            xposition = maxPosition;
        }
        if (xposition < minPosition) {
            xposition = minPosition;
        }
        if (xposition == maxPosition && xvelocity > 0) {
            xvelocity = 0;
        }
        if (xposition == minPosition && xvelocity < 0) {
            xvelocity = 0;
        }


        if (yposition > maxPosition) {
            yposition = maxPosition;
        }
        if (yposition < minPosition) {
            yposition = minPosition;
        }
        if (yposition == maxPosition && yvelocity > 0) {
            yvelocity = 0;
        }
        if (yposition == minPosition && yvelocity < 0) {
            yvelocity = 0;
        }

    }
/**
 * Get the height of the hill at this position
 * @param queryPosition
 * @return
 */
    public double getHeightAtPosition(double queryPosition) {
        return -Math.sin(hillPeakFrequency * (queryPosition));
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
        return Math.cos(hillPeakFrequency * queryPosition);
    }


//    /**
//     * This is basically a copy constructor and we use it when were doing
//     * env_save_state and env_save_state
//     * @param stateToCopy
//     */
//    public MountainCarState3D(MountainCarState3D stateToCopy) {
//        this.xposition = stateToCopy.xposition;
//        this.xvelocity = stateToCopy.xvelocity;
//        this.yposition = stateToCopy.yposition;
//        this.yvelocity = stateToCopy.yvelocity;
//        this.minPosition = stateToCopy.minPosition;
//        this.maxPosition = stateToCopy.maxPosition;
//        this.minVelocity = stateToCopy.minVelocity;
//        this.maxVelocity = stateToCopy.maxVelocity;
//        this.goalPosition = stateToCopy.goalPosition;
//        this.accelerationFactor = stateToCopy.accelerationFactor;
//        this.gravityFactor = stateToCopy.gravityFactor;
//        this.hillPeakFrequency = stateToCopy.hillPeakFrequency;
//        this.defaultInitPosition = stateToCopy.defaultInitPosition;
//        this.defaultInitVelocity = stateToCopy.defaultInitVelocity;
//        this.rewardPerStep = stateToCopy.rewardPerStep;
//        this.rewardAtGoal = stateToCopy.rewardAtGoal;
//
//        this.randomStarts = stateToCopy.randomStarts;
//
////These are pointers but that's ok
//        this.randomGenerator = stateToCopy.randomGenerator;
//    }
}
