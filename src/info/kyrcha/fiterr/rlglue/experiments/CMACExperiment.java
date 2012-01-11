package info.kyrcha.fiterr.rlglue.experiments;

import info.kyrcha.fiterr.Utils;
import info.kyrcha.fiterr.ne.MetaNEATEvolvable;
import info.kyrcha.fiterr.ne.MetaNEATGenome;
import info.kyrcha.fiterr.ne.esn.near.NEARGenome;
import info.kyrcha.fiterr.rlglue.Platform;

import java.lang.reflect.Constructor;
import java.util.Properties;

import org.apache.commons.math.stat.StatUtils;
import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

import org.rlcommunity.rlglue.codec.RLGlue;

/**
 *
 * @author Kyriakos Chatzidimitriou, email: kyrcha [at] issel (dot) ee (dot) auth (dot) gr
 * 
 */
public class CMACExperiment {
	
	// Class variables
	
	/** Logger named after the class */
//	private static final Logger logger = Logger.getLogger(NEARExperiment.class.getName());
	
	protected int episodes = 100;
	
	protected int steps = 2500;
    
    public CMACExperiment(Properties props) {
    	episodes = Integer.parseInt(props.getProperty("experiment.episodes"));
    	steps = Integer.parseInt(props.getProperty("experiment.steps"));
    }

    public void runExperiment() {
//       logger.info("Experiment starts!");
       System.out.println("Experiment starts!");
       for(int episode = 0; episode < episodes; episode++) {
    	  RLGlue.RL_init();
	      RLGlue.RL_episode(steps);
	      double totalReward = RLGlue.RL_return();
	      System.out.println((episode + 1) + ";" + totalReward);
	      RLGlue.RL_cleanup();
       }
       RLGlue.RL_agent_message("close-file");
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
			CMACExperiment theExperiment = new CMACExperiment(props);
	        theExperiment.runExperiment();
    	} catch (Exception e) {
//			logger.error(e);
			e.printStackTrace();
		}
    }
}
