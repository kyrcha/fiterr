package info.kyrcha.fiterr.rlglue.experiments;

import info.kyrcha.fiterr.Utils;
import info.kyrcha.fiterr.ne.MetaNEATGenome;
import info.kyrcha.fiterr.rlglue.Platform;

import java.util.Properties;

import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;
import org.rlcommunity.rlglue.codec.RLGlue;

public class NEARSJSExperiment extends NEARExperiment {
	
	/** Logger named after the class */
//	private static final Logger logger = Logger.getLogger(NEARSJSExperiment.class.getName());

	public NEARSJSExperiment(Properties props) {
		super(props);
	}
    
	@Override
    protected double rewardToFitness(double reward) {
    	return (20000.0 + reward) / 1.0;
    }
    
	@Override
    protected double fitnessToReward(double fitness) {
    	return (1.0 * fitness) - 20000.0;
    }

    public static void main(String[] args) {
    	try {
			// Must have a parameter file
			if (args.length < 1) {
				Utils.fatalError("You must specify a parameter file");
			}
			// Configure the logger
			String fileName = args[0];
			Properties props = Utils.loadProperties(fileName);
//			PropertyConfigurator.configure(props.getProperty(Platform.P_LOG));
			NEARSJSExperiment theExperiment = new NEARSJSExperiment(props);
	        theExperiment.runExperiment();
    	} catch (Exception e) {
//			logger.error(e);
			e.printStackTrace();
		}
    }

}
