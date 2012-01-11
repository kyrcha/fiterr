package info.kyrcha.fiterr.rlglue;

import info.kyrcha.fiterr.Utils;

import java.io.File;
import java.io.FileInputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Properties;

import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

import org.rlcommunity.rlglue.codec.AgentInterface;
import org.rlcommunity.rlglue.codec.EnvironmentInterface;
import org.rlcommunity.rlglue.codec.LocalGlue;
import org.rlcommunity.rlglue.codec.RLGlue;

/**
 * Platform class runs all RL-Glue components (agent, environment, experiment) without
 * using network sockets in order to speed up execution. It will also load the parameter
 * file of the experiment to be conducted.
 * 
 * @author Kyriakos Chatzidimitriou, email: kyrcha [at] issel (dot) ee (dot) auth (dot) gr
 *
 */
public class Platform {
	
	// PRIVATE IVARS
	
	/** Logger named after the class */
//	private static final Logger logger = Logger.getLogger(Platform.class.getName());
	
	/** Single point of control for the property agent */
	private static final String P_AGENT = "agent";
	
	/** Single point of control for the property environment */
	private static final String P_ENVIRONMENT = "environment";
	
	/** Single point of control for the property experiment */
	private static final String P_EXPERIMENT = "experiment";
	
	/** Single point of control for the logger configuration file */
	private static String logConfigFile = "params/log.conf";
	
	/** Single point of control for the property log */
	public static final String P_LOG = "log";

	/**
	 * Main entry of the application.
	 * 
	 * @param args Should contain the specification of the run (Experiment, Agent, Environment)
	 */
	public static void main(String[] args) {
		try {
			// Must have a parameter file
			if (args.length < 1) {
				Utils.fatalError("You must specify a parameter file");
			}
			String fileName = args[0];
			File file = new File(fileName);
			// Test if I can read the file
			if (!file.canRead()) {
				Utils.fatalError("Cannot read parameter file: " + fileName);
			}
			// Create and load default properties
			Properties props = new Properties();
			FileInputStream in = new FileInputStream(fileName);
			props.load(in);
			
			in.close();
			// Configure the logger
//			logConfigFile = props.getProperty(P_LOG);
//			PropertyConfigurator.configure(logConfigFile);

			// Start Logging
//			logger.info((new java.util.Date(System.currentTimeMillis())));
//			logger.info("Experiment starts!!!");
			// Create the Agent
			Class<?> agentClass = Class.forName(props.getProperty(P_AGENT));
			Constructor<?> ctor = agentClass.getDeclaredConstructor(String.class);
			String agentParams = props.getProperty("agent.params");
			AgentInterface theAgent = (AgentInterface)ctor.newInstance(agentParams);
			// Create the Environment
			Class<?> environmentClass = Class.forName(props.getProperty(P_ENVIRONMENT));
			ctor = environmentClass.getDeclaredConstructor(String.class);
			EnvironmentInterface theEnvironment = (EnvironmentInterface)ctor.newInstance(props.getProperty("environment.params"));			
			LocalGlue localGlueImplementation = new LocalGlue(theEnvironment, theAgent);
			RLGlue.setGlue(localGlueImplementation);			
			// Run the main method of the Experiment. This will run the experiment in the main thread.  
			// The Agent and Environment will run locally, without sockets.
			try {
			    Class<?> experimentClass = Class.forName(props.getProperty(P_EXPERIMENT));
			    Class<?>[] argTypes = new Class<?>[] { String[].class };
			    Method main = experimentClass.getDeclaredMethod("main", argTypes);
			    // One way
//		  	    String[] mainArgs = Arrays.copyOfRange(args, 0, args.length);
//			    main.invoke(null, (Object)mainArgs);
			    // Another way
			    main.invoke(null, (Object)args);
			} catch (ClassNotFoundException x) {
				x.printStackTrace();
			} catch (NoSuchMethodException x) {
				x.printStackTrace();
			} catch (IllegalAccessException x) {
				x.printStackTrace();
			} catch (InvocationTargetException x) {
				x.printStackTrace();
			}
		} catch (Exception e) {
//			logger.error(e);
			e.printStackTrace();
		}
		
//		logger.info("Experiment Complete!!!\n");
	}

}
