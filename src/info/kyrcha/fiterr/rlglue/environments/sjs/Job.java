package info.kyrcha.fiterr.rlglue.environments.sjs;

public class Job {
	
	private int type;
	
	private int time;
	
	public Job(int atype) {
		type = atype;
		time = 0;
	}
	
	public void incrTime() {
		time++;
	}
	
	public int getType() {
		return type;
	}
	
	public int getPeriod() {
		if(time < 50) {
			return 0;
		} else if(time < 100) {
			return 1;
		} else if(time < 150) {
			return 2;
		} else {
			return 3;
		}
	}
	
	public double getUtility() {
		if(type == 0) {
			return utilityType1();
		} else if(type == 1) {
			return utilityType2();
		} else if(type == 2) {
			return utilityType3();
		} else {
			return utilityType4();
		}
	}
	
	public double utilityType1() {
		if(time <= 100) {
			return 0;
		} else if (time <= 150){
			return -time + 100;
		} else {
			return (- 2 * time) + 250; 
		}
	}
	
	public double utilityType2() {
		if(time <= 100) {
			return 0;
		} else if (time <= 150){
			return (- 2 * time) + 200;
		} else {
			return -time + 50; 
		}
	}
	
	public double utilityType3() {
		if(time <= 50) {
			return -time;
		} else if (time <= 100){
			return (- 2 * time) + 50;
		} else {
			return -150; 
		}
	}
	
	public double utilityType4() {
		if(time <= 50) {
			return (- 2 * time);
		} else if (time <= 100){
			return -time - 50;
		} else {
			return -150; 
		}
	}

}
