package info.kyrcha.fiterr.ne.neat;

import info.kyrcha.fiterr.Utils;

public class Connection {
	
	private int from;
	
	private int to;
	
	private double weight;
	
	private boolean enabled = true;
	
	private int innovationNumber;
	
	
	Connection(int afrom, int ato, double aweight, int ainnovationNumber) {
		from = afrom;
		to = ato;
		weight = aweight;
		innovationNumber = ainnovationNumber;
	}
	
	public int getInnovationNumber() {
		return innovationNumber;
	}
	
	public void setInnovationNumber(int in) {
		innovationNumber = in;
	}
	
	public double getWeight() {
		return weight;
	}
	
	public void setWeight(double aweight) {
		weight = aweight;
	}
	
	public int getTo() {
		return to;
	}
	
	public int getFrom() {
		return from;
	}
	
	public void mutateWeight() {
		weight += Utils.pertubation(NEAT.weightRange);
	}
	
	public boolean getEnabled() {
		return enabled;
	}
	
	public void setEnabled(boolean aenabled) {
		enabled = aenabled;
	}
	
	public void disable() {
		enabled = false;
	}
	
	public void enable() {
		enabled = true;
	}
	
	public static Connection newInstance(Connection aConnection) {
		Connection clone =  new Connection(
				aConnection.getFrom(),
				aConnection.getTo(),
				aConnection.getWeight(),
				aConnection.getInnovationNumber());
		clone.setEnabled(aConnection.getEnabled());
		return clone;
	}

}
