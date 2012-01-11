package info.kyrcha.fiterr.ne.neat;

public class Innovation {
	
	private int from;
	
	private int to;
	
	private int innovationNumber;
	
	public Innovation(int afrom, int ato, int ainnovationNumber) {
		from = afrom;
		to = ato;
		innovationNumber = ainnovationNumber;
	}
	
	public int getFrom() {
		return from;
	}

	public int getTo() {
		return to;
	}
	
	public int getInnovationNumber() {
		return innovationNumber;
	}
	
}
