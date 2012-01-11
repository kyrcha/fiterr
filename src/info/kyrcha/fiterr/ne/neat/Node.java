package info.kyrcha.fiterr.ne.neat;

/**
 * Node class represents a neuron in the NEAT algorithm.
 * 
 * @author Kyriakos C. Chatzidimitriou (EMAIL - kyrcha [at] gmail (dot) com, WEB - http://kyrcha.info)
 *
 */
public class Node {
	
	// Class variables
	
	// Scope: private
	
	/** ID of the node */
	private int id = 0;
	
	/** An integer denoting the type of node: SENSOR, OUTPUT, HIDDEN */
	private int type;
	
	/** The depth of the node starting from source (an imaginary */
	private long depth = 0;
	
	/** Check whether we have visited the node or not during the activation procedure */
	private boolean visited = false;
	
	/** The value of the last activation */
	private double lastActivation = 0;
	
	/** ???TODO document color */
	private int color = 0;
	
	// Scope: package
	
	/** Type of node: source */
	static final int SOURCE = -1;
	
	/** Type of node: sensor */
	static final int SENSOR = 0;
	
	/** Type of node: hidden */
	static final int HIDDEN = 1;
	
	/** Type of node: output */
	static final int OUTPUT = 2;
	
	/** Type of node: bias */
	static final int BIAS = 3;
	
	/** Type of node: not found */
	static final int NOT_FOUND = -1;
	
	// End of class variables
	
	// Constructors
	
	Node(){}
	
	Node(int atype, int aid, long adepth) {
		type = atype;
		id = aid;
		depth = adepth;
	}
	
	// End of constructors
	
	// Class methods
	
	// Scope: public
	
	public long getDepth() {
		return depth;
	}
	
	public void setDepth(long adepth) {
		depth = adepth;
	}
	
	public int getColor() {
		return color;
	}
	
	public void setColor(int acolor) {
		color = acolor;
	}
	
	public int getType() {
		return type;
	}
	
	public void setType(int atype) {
		type = atype;
	}
	
	public int getId() {
		return id;
	}
	
	public void setId(int aid) {
		id = aid;
	}
	
	public boolean getVisited() {
		return visited;
	}
	
	public void setVisited(boolean avisited) {
		visited = avisited;
	}
	
	public double getLastActivation() {
		return lastActivation;
	}
	
	public void setLastActivation(double alastActivation) {
		lastActivation = alastActivation;
	}
	
	public static Node newInstance(Node aNode) {
		return new Node(aNode.getType(), aNode.getId(), aNode.getDepth());
	}
	
	// End of class methods

}
