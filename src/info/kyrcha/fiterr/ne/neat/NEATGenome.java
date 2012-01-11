package info.kyrcha.fiterr.ne.neat;

import info.kyrcha.fiterr.Function;
import info.kyrcha.fiterr.Utils;

import info.kyrcha.fiterr.ne.MetaNEATGenome;
import info.kyrcha.fiterr.ne.Network;

import java.util.ArrayList;
import java.util.Iterator;

public class NEATGenome extends MetaNEATGenome {
	
	// Class Variables
	
	// Scope: private
	
	private NEAT neat;
	
	private int activeConnections;
		
	private ArrayList<Node> nodes = new ArrayList<Node>();
	
	private ArrayList<Connection> connections = new ArrayList<Connection>();
	
	// End of class variables
	
	// Constructors
	
	NEATGenome(){}
	
	NEATGenome(NEAT aneat) {
		neat = aneat;
		// Initialize fitness
		fitness = 0;
		// Create initial nodes and connections based on the number of inputs and outputs
		int id = 1;
		// Create the input nodes
		for(int i = 0; i < neat.getNInputUnits(); i++) {
			Node node = new Node(Node.SENSOR, id, 0);
			id++;
			nodes.add(node);
			nInputUnits++;
		}
		// Create the output nodes
		for(int i = 0; i < neat.getNOutputUnits(); i++) {
			Node node = new Node(Node.OUTPUT, id, Integer.MAX_VALUE);
			id++;
			nodes.add(node);
			nOutputUnits++;
		}
		// Create the connections (fully connect sensors with outputs)
		int pseudoInnovationNumber = 0;
		for(int i = 0; i < neat.getNInputUnits(); i++) {
			for(int j = 0; j < neat.getNOutputUnits(); j++) {
				pseudoInnovationNumber++;
				Connection conx = new Connection(
						nodes.get(i).getId(), 
						nodes.get(neat.getNInputUnits() + j).getId(), 
						Utils.pertubation(NEAT.weightRange), 
						pseudoInnovationNumber);
				activeConnections++;
				insertConnection(conx);
			}
		}
	}
	
	// End of constructors
	
	// Class methods
	
	// Scope: private
	
	private boolean insertNode(Node anode) {
		for(int i = 0; i < nodes.size(); i++) {
			Node current = nodes.get(i);
			if(current.getId() == anode.getId()) {
				return false;
			} else if(current.getId() > anode.getId()) {
				nodes.add(i, anode);
				return true;
			}
		}
		nodes.add(anode);
		return true;
	}
	
	private Node findNode(int id) {
		Iterator<Node> nodeIter = nodes.iterator();
		while(nodeIter.hasNext()) {
			Node n = nodeIter.next();
			if(n.getId() == id) {
				return n;
			}
		}
		return null;
	}
	
	/** Mutate Weights */
	private void mutateWeights() {
		Iterator<Connection> iter = this.connections.iterator();
		double endpart = 0.8 * connections.size();
		double num = 0.0;
		// Mutate the weights
		while(iter.hasNext()) {
			if(Utils.rand.nextDouble() < NEAT.weightMutationProbability || ((connections.size() >= 10) && (num > endpart))) {
				Connection c = iter.next();
				double oldval = c.getWeight();
				double pertube = Utils.rand.nextDouble() * NEAT.weightRange;
				// Once in a while leave the end part alone
				if(num > endpart)
					if(Utils.rand.nextDouble() < 0.2)
						pertube = 0;
				// Decid positive or negative
				if(Utils.rand.nextBoolean()) {
				// Positive
					// If it goes over max, find something smaller
					if((oldval + pertube) > 100.0) {
						pertube = (100 - oldval) * Utils.rand.nextDouble();
					}
					c.setWeight(oldval + pertube);
				} else {
				// Negative
					if((oldval - pertube) < -100.0) {
						pertube = (oldval + 100.0) * Utils.rand.nextDouble();
					}
					// If it goes below min, find something smaller
					c.setWeight(oldval - pertube);
				}
			} else {
				iter.next();
			}
			num += 1.0;
		}
		
		// Once in a while shake things up
		boolean severe;
		if(Utils.rand.nextDouble() > 0.5) {
			severe = true;
		} else {
			severe = false;
		}
		
		iter = this.connections.iterator();
		endpart = 0.8 * connections.size();
		num = 0.0;
		double gausspoint;
		double coldgausspoint;
		double powermod = 1.0;
		while(iter.hasNext()) {
			Connection c = iter.next();
			if(severe) {
				gausspoint = 0.3;
				coldgausspoint = 0.1;
			} else if((connections.size() >= 10) && (num > endpart)) {
				gausspoint = 0.5;
				coldgausspoint = 0.3;
			} else {
				if(Utils.rand.nextFloat() > 0.5) {
					gausspoint = 1.0 - NEAT.weightMutationProbability;
					coldgausspoint = 1.0 - NEAT.weightMutationProbability - 0.1;
				} else {
					gausspoint = 1.0 - NEAT.weightMutationProbability;
					coldgausspoint = 1.0 - NEAT.weightMutationProbability;
				}
			}
			
			double posneg = 1.0;
			
			if(Utils.rand.nextBoolean()) {
				posneg = -1.0;
			}
			
			double randnum = posneg * Utils.rand.nextDouble() * NEAT.weightRange * powermod;
			
			if(NEAT.MUT_TYPE.equalsIgnoreCase("Gaussian")) {
				double randchoice = Utils.rand.nextDouble();
				if(randchoice > gausspoint) {
					c.setWeight(randnum);
				} else if(randchoice > coldgausspoint) {
					c.setWeight(randnum);
				}
			} else if(NEAT.MUT_TYPE.equalsIgnoreCase("ColdGaussian")) {
				c.setWeight(randnum);
			}
			num += 1.0;
		}
	}
	
	private void splitConnection(Connection conn) {
		// Add the node (with the proper id - the next after the last)
		int lastId = nodes.get(nodes.size()-1).getId();
		int newId = lastId + 1;
		Node newNode = new Node(Node.HIDDEN, newId, (findNode(conn.getFrom()).getDepth() + findNode(conn.getTo()).getDepth())/2);
		nodes.add(newNode);
		nInternalUnits++;
		// Add the two new connections
		insertConnection(createConnection(conn.getFrom(), newNode.getId(), 1.0));
		activeConnections++;
		insertConnection(createConnection(newNode.getId(), conn.getTo(), conn.getWeight()));
		activeConnections++;
		// Disable the old one
		conn.disable();
		activeConnections--;
	}
	
	private Connection createConnection(int from, int to, double weight) {
		int gin = neat.exists(from, to);
		if(gin == NEAT.NOT_FOUND) {
			gin = neat.incrementInnovationNumber();
			neat.addInnovation(from, to, gin);
		}
		Connection connection = new Connection(from, to, weight, gin);
		return connection;
	}
	
	/** The method adds a node to the network by splitting a link */
	private void addNode() {
		if(connections.size() < 15) {
			// For a very small genome we need to bias splitting towards older links
			// TODO add bias
			for(int i = 0; i < connections.size(); i++) {
				if(connections.get(i).getEnabled() && Utils.rand.nextDouble() > 0.3) {
					Connection conn = connections.get(i);
					splitConnection(conn);
					break;
				}
			}
		} else {
			// Pick a random connection to add a node and disable it from the list of active connections
			ArrayList<Integer> indexListOfActiveConnections = new ArrayList<Integer>();
			for(int i = 0; i < connections.size(); i++) {
				if(connections.get(i).getEnabled()) {
					indexListOfActiveConnections.add(new Integer(i));
				}
			}
			// If there exists such a connection
			if(indexListOfActiveConnections.size() > 0) {
				int connIdx = Utils.rand.nextInt(indexListOfActiveConnections.size());
				Connection conn = connections.get(indexListOfActiveConnections.get(connIdx).intValue());
				// Disable it
				splitConnection(conn);
			}
		}
	}
	
	/** Add a link when structural mutation takes place */
	private void addLink() {
		ArrayList<Connection> possibleConnections = new ArrayList<Connection>();
		ArrayList<Connection> possibleRecurrentConnections = new ArrayList<Connection>();
		for(int i = 0; i < nodes.size(); i++) {
			Node from = nodes.get(i);
			for(int j = 0; j < nodes.size(); j++) {
				Node to = nodes.get(j);
				// No connections should go back to the input
				if(to.getType() == Node.SENSOR) {
					continue;
				} else {
					Iterator<Connection> currentConnections = connections.iterator();
					boolean exists = false;
					while(currentConnections.hasNext()) {
						Connection check = currentConnections.next();
						if(check.getFrom() == from.getId() && check.getTo() == to.getId()) {
							exists = true;
							break;
						}
					}
					if(!exists) {
						// Recurrent is a connection from a node of greater depth to an node of equal or smaller depth
						if(from.getDepth() > to.getDepth() || from.getId() == to.getId()) {
							possibleRecurrentConnections.add(new Connection(from.getId(), to.getId(), Utils.pertubation(NEAT.weightRange), NEAT.NOT_FOUND));
						} else {
							possibleConnections.add(new Connection(from.getId(), to.getId(), Utils.pertubation(NEAT.weightRange), NEAT.NOT_FOUND));
						}
					}
				}
			}
		}
		if(Utils.rand.nextDouble() < NEAT.recurrentProb && possibleRecurrentConnections.size() > 0) {
			int randIdx = Utils.rand.nextInt(possibleRecurrentConnections.size());
			Connection newEntrant = possibleRecurrentConnections.get(randIdx);
			// There is certainly a link to be added
			addNewConnection(newEntrant);
		} else if(possibleConnections.size() > 0) {
			int randIdx = Utils.rand.nextInt(possibleConnections.size());
			Connection newEntrant = possibleConnections.get(randIdx);
			// There is certainly a link to be added
			addNewConnection(newEntrant);
		}
	}
	
	private void addNewConnection(Connection newEntrant) {
		// There is certainly a link to be added
		int in = neat.exists(newEntrant.getFrom(), newEntrant.getTo());
		if(in == NEAT.NOT_FOUND) {
			in = neat.incrementInnovationNumber();
			neat.addInnovation(newEntrant.getFrom(), newEntrant.getTo(), in);
		}
		newEntrant.setInnovationNumber(in);
		insertConnection(newEntrant);
		activeConnections++;
	}
	
	// Scope: public
	
	/** 
	 * Convert the genome to its phenotype, a neural network  
	 */
	public Network toPhenotype() {
		Network net = new Network(nInputUnits, nInternalUnits, nOutputUnits, Function.SIGMOID, Function.SIGMOID);
		Iterator<Connection> iter = connections.iterator();
		int hiddenNodesStart = nInputUnits + nOutputUnits;
		while(iter.hasNext()) {
			Connection c = iter.next();
			if(c.getEnabled()) {
				// Take all cases
				int in = findNode(c.getFrom()).getType(); 
				int out = findNode(c.getTo()).getType();
				double w = c.getWeight();
				if(in == Node.SENSOR && out == Node.HIDDEN) {
					int m = c.getTo() - hiddenNodesStart- 1;
					int n = c.getFrom() - 1;
					net.setWin(m, n, w);
				} else if(in == Node.SENSOR && out == Node.OUTPUT) {
					int m = c.getTo() - nInputUnits - 1;
					int n = c.getFrom() - 1;
					net.setWout(m, n, w);
				} else if(in == Node.HIDDEN && out == Node.HIDDEN) {
					int m = c.getTo() - hiddenNodesStart- 1;
					int n = c.getFrom() - hiddenNodesStart- 1;
					net.setW(m, n, w);
				} else if(in == Node.HIDDEN && out == Node.OUTPUT) {
					int m = c.getTo() - nInputUnits - 1;
					int n = c.getFrom() - nOutputUnits - 1;
					net.setWout(m, n, w);
				} else if(in == Node.OUTPUT && out == Node.HIDDEN) {
					int m = c.getTo() - hiddenNodesStart- 1;
					int n = c.getFrom() - nInputUnits - 1;
					net.setWback(m, n, w);
				} else if(in == Node.OUTPUT && out == Node.OUTPUT) {
					int m = c.getTo() - nInputUnits - 1;
					int n = c.getFrom() + nInternalUnits - 1;
					net.setWout(m, n, w);
				}
			}
		}
		return net;
	}
	
	/** Get the number of nodes (input, output, hidden) */
	public int getNumberOfNodes() {
		return nodes.size();
	}
	
	/** Get the total number of connections (enabled and disabled) */
	public int getNumberOfConnections() {
		return connections.size();
	}
	
	/** Get the number of active/enabled connections */
	public int getActiveConnections() {
		return activeConnections;
	}
	
	/** 
	 * Transform the genome to be represented by a String
	 * 
	 * @return A String representing the genome 
	 */
	public String toString() {
		// TODO Use String Builder
		String temp = new String("Nodes-");
		Iterator<Node> iter = nodes.iterator();
		while(iter.hasNext()) {
			Node n = iter.next();
			temp += n.getId() + "/" + n.getType() + "/" + n.getDepth() + "-";
		}
		Iterator<Connection> iterConn = connections.iterator();
		temp += "|Connections/";
		while(iterConn.hasNext()) {
			Connection c = iterConn.next();
			temp += c.getFrom() +"->" +c.getTo() + ";" + c.getWeight() + ";" + c.getInnovationNumber() + ";"  + c.getEnabled() + "/";
		}
		return temp;
	}

	/** 
	 * Method that verifies the integrity of the genome. Checks for duplicate connections
	 * and removes them.
	 */
	public void verify() {
		Iterator<Connection> connIter = connections.iterator();
		ArrayList<Connection> verifiedConns = new ArrayList<Connection>();
		int numOfNodes = nodes.size();
		boolean[][] graph = new boolean[numOfNodes][numOfNodes];
		while(connIter.hasNext()) {
			Connection conn = connIter.next();
			if(!graph[conn.getFrom() - 1][conn.getTo() - 1]) {
				graph[conn.getFrom() - 1][conn.getTo() - 1] = true;
				verifiedConns.add(conn);
			}
		}
		connections = verifiedConns;
	}
	
	public MetaNEATGenome xover(MetaNEATGenome aparent2) {
		NEATGenome parent2 = (NEATGenome)aparent2;
		// The offspring
		NEATGenome offspring = new NEATGenome();
		// Initialize
		offspring.neat = neat; // only one NEAT exists
		offspring.nInputUnits = nInputUnits; // and number of input
		offspring.nOutputUnits = nOutputUnits; // and number of outputs
		offspring.activeConnections = 0; // reset active connections
		offspring.connections = new ArrayList<Connection>(); // reset connections
		offspring.nodes = new ArrayList<Node>();
		int better = 0;
		// Check to see the most fit parent (Get the excess and disjoint genes from the most fit parent)
		if(getAdjFitness() > parent2.getAdjFitness()) {
			better = 1;
		} else if(getAdjFitness() < parent2.getAdjFitness()) {
			better = 2;
		}
		boolean cont = true; //continue?
		// Maximum length
		int N = Math.max(connections.size(), parent2.connections.size());
		// The two parent indexes used for scanning
		int parIdx1 = 0;
		int parIdx2 = 0; 
		while(cont) {
			Connection parCon1 = connections.get(Math.min(parIdx1, connections.size() - 1));
			Connection parCon2 = parent2.connections.get(Math.min(parIdx2, parent2.connections.size() - 1));
			if(parCon1.getInnovationNumber() == parCon2.getInnovationNumber()) {// Matching
				if(better != 0) {
					if(Utils.rand.nextDouble() < NEAT.mateAverage) {
						double w1 = parCon1.getWeight();
						double w2 = parCon2.getWeight();
						Connection newConn;
						if(better == 1) {
							newConn = Connection.newInstance(parCon1);
							if(parCon1.getEnabled()) {
								offspring.activeConnections++;
							}
							newConn.setWeight((w1 + w2) / 2);
							// insert the connection and the corresponding nodes
							offspring.insertConnection(newConn);
							offspring.insertNode(Node.newInstance(findNode(newConn.getFrom())));
							offspring.insertNode(Node.newInstance(findNode(newConn.getTo())));
						} else {
							newConn = Connection.newInstance(parCon2);
							if(parCon2.getEnabled()) {
								offspring.activeConnections++;
							}
							newConn.setWeight((w1 + w2) / 2);
							// insert the connection and the corresponding nodes
							offspring.insertConnection(newConn);
							offspring.insertNode(Node.newInstance(parent2.findNode(newConn.getFrom())));
							offspring.insertNode(Node.newInstance(parent2.findNode(newConn.getTo())));
						}
					} else {
						if(Utils.rand.nextDouble() < 0.5) {
							Connection newConn = Connection.newInstance(parCon1);
							offspring.insertConnection(newConn);
							offspring.insertNode(Node.newInstance(findNode(newConn.getFrom())));
							offspring.insertNode(Node.newInstance(findNode(newConn.getTo())));
							if(parCon1.getEnabled()) {
								offspring.activeConnections++;
							}
						} else {
							Connection newConn = Connection.newInstance(parCon2);
							offspring.insertConnection(newConn);
							offspring.insertNode(Node.newInstance(parent2.findNode(newConn.getFrom())));
							offspring.insertNode(Node.newInstance(parent2.findNode(newConn.getTo())));
							if(parCon2.getEnabled()) {
								offspring.activeConnections++;
							}
						}
					}
				} else {
					if(Utils.rand.nextDouble() < 0.5) {
						Connection newConn = Connection.newInstance(parCon1);
						offspring.insertConnection(newConn);
						offspring.insertNode(Node.newInstance(findNode(newConn.getFrom())));
						offspring.insertNode(Node.newInstance(findNode(newConn.getTo())));
						if(parCon1.getEnabled()) {
							offspring.activeConnections++;
						}
					} else {
						Connection newConn = Connection.newInstance(parCon2);
						offspring.insertConnection(newConn);
						offspring.insertNode(Node.newInstance(parent2.findNode(newConn.getFrom())));
						offspring.insertNode(Node.newInstance(parent2.findNode(newConn.getTo())));
						if(parCon2.getEnabled()) {
							offspring.activeConnections++;
						}
					}
				}
				parIdx1++;
				parIdx2++;
			} else if(parCon1.getInnovationNumber() > parCon2.getInnovationNumber()) {
				// parent 2 has reached its limit
				if(parIdx2 == parent2.connections.size()) { // excess
					parIdx1++;
					if(better == 0 || better == 1) {
						Connection newConn = Connection.newInstance(parCon1);
						offspring.insertConnection(newConn);
						offspring.insertNode(Node.newInstance(findNode(newConn.getFrom())));
						offspring.insertNode(Node.newInstance(findNode(newConn.getTo())));
						if(parCon1.getEnabled()) {
							offspring.activeConnections++;
						}
					}
				} else { // disjoint
					parIdx2++;
					if(better == 0 || better == 2) {
						Connection newConn = Connection.newInstance(parCon2);
						offspring.insertConnection(newConn);
						offspring.insertNode(Node.newInstance(parent2.findNode(newConn.getFrom())));
						offspring.insertNode(Node.newInstance(parent2.findNode(newConn.getTo())));
						if(parCon2.getEnabled()) {
							offspring.activeConnections++;
						}
					}
				}
			} else if(parCon1.getInnovationNumber() < parCon2.getInnovationNumber()) {
				// parent 1 has reached its limit
				if(parIdx1 == connections.size()) {
					parIdx2++;
					if(better == 0 || better == 2) {
						Connection newConn = Connection.newInstance(parCon2);
						offspring.insertConnection(newConn);
						offspring.insertNode(Node.newInstance(parent2.findNode(newConn.getFrom())));
						offspring.insertNode(Node.newInstance(parent2.findNode(newConn.getTo())));
						if(parCon2.getEnabled()) {
							offspring.activeConnections++;
						}
					}
				} else {
					parIdx1++;
					if(better == 0 || better == 1) {
						Connection newConn = Connection.newInstance(parCon1);
						offspring.insertConnection(newConn);
						offspring.insertNode(Node.newInstance(findNode(newConn.getFrom())));
						offspring.insertNode(Node.newInstance(findNode(newConn.getTo())));
						if(parCon1.getEnabled()) {
							offspring.activeConnections++;
						}
					}
				}
			}
			if(parIdx1 == N || parIdx2 == N) {
				cont = false;
			}
		}
		int hiddenCounter = 0;
		int inputCounter = 0;
		int outputCounter = 0;
		Iterator<Node> iter = offspring.getNodeIterator();
		while(iter.hasNext()) {
			Node n = iter.next();
			if(n.getType() == Node.HIDDEN)
				hiddenCounter++;
			if(n.getType() == Node.SENSOR)
				inputCounter++;
			if(n.getType() == Node.OUTPUT)
				outputCounter++;
		}
		offspring.nInternalUnits = hiddenCounter;
		if(nInputUnits != inputCounter) {
			System.out.println("ERROR xover");
			System.exit(1);
		}
		if(nOutputUnits != outputCounter) {
			System.out.println("ERROR xover");
			System.exit(1);
		}
		return offspring;
	}
	
	public void mutate() {
		// Mutate weights
		mutateWeights();
		// Add node
		if(Utils.rand.nextDouble() < NEAT.addNodeProb) {
			addNode();
		} else if(Utils.rand.nextDouble() < NEAT.addLinkProb) { // Add link
			addLink();
		}
	}
	
	boolean insertConnection(Connection aconn) {
		for(int i = 0; i < connections.size(); i++) {
			Connection current = connections.get(i);
			if(current.getInnovationNumber() == aconn.getInnovationNumber()) {
				return false;
			} else if(current.getInnovationNumber() > aconn.getInnovationNumber()) {
				connections.add(i, aconn);
				return true;
			}
		}
		connections.add(aconn);
		return true;
	}
	
	public double calculateDistance(MetaNEATGenome arepresentative) {
		NEATGenome representative = (NEATGenome)arepresentative;
		NEATGenome applicant = this;
		int N = Math.max(applicant.connections.size(), representative.connections.size());
		int excess = 0;
		int disjoint = 0;
		int matching = 0;
		double weightDiff = 0.0;
		int appIdx = 0;
		int repIdx = 0;
		boolean cont = true;
		while(cont) {
			Connection applicantCon = applicant.connections.get(Math.min(appIdx, applicant.connections.size()-1));
			Connection representativeCon = representative.connections.get(Math.min(repIdx, representative.connections.size()-1));
			if(applicantCon.getInnovationNumber() == representativeCon.getInnovationNumber()) {
				matching++;
				weightDiff += Math.abs(applicantCon.getWeight() - representativeCon.getWeight());
				appIdx++;
				repIdx++;
			} else if(applicantCon.getInnovationNumber() > representativeCon.getInnovationNumber()) {
				// representative has reached its limit
				if(repIdx == representative.connections.size()) {
					excess++;
					appIdx++;
				} else {
					disjoint++;
					repIdx++;
				}
			} else if(applicantCon.getInnovationNumber() < representativeCon.getInnovationNumber()) {
				// gene has reached its limit
				if(appIdx == applicant.connections.size()) {
					excess++;
					repIdx++;
				} else {
					disjoint++;
					appIdx++;
				}
			}
			if(appIdx == N || repIdx == N) {
				cont = false;
			}
		}
		return (NEAT.C1 * excess / N) + (NEAT.C2 * disjoint / N) + (NEAT.C3 * weightDiff / matching);
	}
	
	void setActiveConnections(int aactiveConnections) {
		activeConnections = aactiveConnections;
	}
	
	NEAT getNEAT() {
		return neat;
	}
	
	void setNEAT(NEAT aneat) {
		neat = aneat;
	}
	
	Iterator<Node> getNodeIterator() {
		return nodes.iterator();
	}
	
	Iterator<Connection> getConnectionIterator() {
		return connections.iterator();
	}
	
	public NEATGenome clone() {
		NEATGenome clone = new NEATGenome();
		clone.nInputUnits = nInputUnits ;
		clone.nOutputUnits = nOutputUnits;
		clone.nInternalUnits = nInternalUnits;
		clone.fitness = 0d;
		clone.adjFitness = 0;
		clone.setSpecies(getSpecies());
		clone.setNEAT(getNEAT());
		clone.setActiveConnections(getActiveConnections());
		Iterator<Node> nodeIter = getNodeIterator();
		while(nodeIter.hasNext()) {
			clone.insertNode(Node.newInstance(nodeIter.next()));
		}
		Iterator<Connection> connIter = getConnectionIterator(); 
		while(connIter.hasNext()) {
			clone.insertConnection(Connection.newInstance(connIter.next()));
		}
		return clone;
	}

	@Override
	public String message(String astring) {
		// TODO Auto-generated method stub
		return null;
	}
	
}
