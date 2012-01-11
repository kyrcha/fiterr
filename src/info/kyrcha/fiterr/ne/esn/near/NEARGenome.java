package info.kyrcha.fiterr.ne.esn.near;

import Jama.EigenvalueDecomposition;

import Jama.Matrix;
import info.kyrcha.fiterr.Function;
import info.kyrcha.fiterr.LearningMode;
import info.kyrcha.fiterr.Utils;
import info.kyrcha.fiterr.esn.ESN;
import info.kyrcha.fiterr.esn.ESNType;
import info.kyrcha.fiterr.ne.MetaNEATGenome;
import info.kyrcha.fiterr.ne.Network;

public class NEARGenome extends MetaNEATGenome {
	
	private static final double RAND_INP = 1d;
	
	private static final double RAND_INT = 1d;
	
	private static final double RAND_OUT = 1d;
	
	private NEAR near;
	
	private double[][] wIn;
	
	private double[][] w;
	
	private boolean[][] wBool;
	
	private double[][] wOut;

	private double[][] wBack;
	
	private double rho = 0;
	
	private double D = 0;
	
	private double runningTally = 0;
	
	private int nActiveReservoirConns;
	
	private boolean outOfCrossoverLargest = false;
	
	private boolean outOfCrossoverFittest = false;
	
	private double parentFitness1 = 0;
	
	private double parentFitness2 = 0;
	
	public ESN toESN() {
		ESN aesn = new ESN();
		aesn.setNumberOfInputUnits(nInputUnits);
		aesn.setNumberOfOutputUnits(nOutputUnits);
		aesn.setNumberOfInternalUnits(nInternalUnits);
		aesn.setNumberOfTotalUnits(nInputUnits+nOutputUnits+nInternalUnits);
		aesn.setLearningMode(near.learningMode);
		aesn.setWeightComputeMethod(near.weightComputeMethod);
		aesn.setESNType(ESNType.PLAIN_ESN);
		aesn.setReservoirActivationFunction(near.reservoirActivationFunction);
		aesn.setOutputActivationFunction(near.outputActivationFunction);
		aesn.setDensity(D);
		aesn.setSpectralRadius(rho);
		aesn.setInputScaling(near.inputScaling);
		aesn.setOutputScaling(near.teacherScaling);
		aesn.setInputShift(near.inputShift);
		aesn.setOutputShift(near.teacherShift);
		aesn.setNoiseLevel(near.noiseLevel);
		aesn.setInputWeights(wIn);
		aesn.setFeedbackWeights();
		aesn.setFeedbackScaling(near.feedbackScaling);
		aesn.setOutputWeights(wOut);
		Matrix internal = new Matrix(Utils.cloneMatrix(w));
		try {
			EigenvalueDecomposition evd = internal.eig();
	    	double[] reEigVals = evd.getRealEigenvalues();
	    	double[] imEigVals = evd.getImagEigenvalues();
	    	double[] eigVals = new double[nInternalUnits];
	    	double maxVal = Double.MIN_VALUE;
	    	for(int i = 0; i < reEigVals.length; i++) {
	    		eigVals[i] = Math.sqrt(reEigVals[i] * reEigVals[i] + imEigVals[i] * imEigVals[i]);
	    		if(maxVal < eigVals[i]) {
	    			maxVal = eigVals[i];
	    		}
	    	}
	    	if(maxVal < 0.001) {
	    		maxVal = 1.0;
	    	}
	    	internal = internal.times(1/maxVal);
	    	internal.times(rho);
	    	aesn.setInternalWeights(internal.getArray());
		} catch (Exception e) {
			e.printStackTrace();
		}
		aesn.buildWeightMatrix();
		aesn.flush();
		return aesn;
	}

	@Override
	public Network toPhenotype() {
		Network net = new Network(nInputUnits, nInternalUnits, nOutputUnits, near.reservoirActivationFunction, near.outputActivationFunction);
		net.setWback(wBack);
		net.setWin(wIn);
		// No output to output connection for this version
		for(int i = 0; i < nOutputUnits; i++) {
			for(int j = 0; j < (nInputUnits + nInternalUnits + nOutputUnits); j++) {
				if(j < (nInputUnits + nInternalUnits)) {
					net.setWout(i, j, wOut[i][j]);
				} else {
					net.setWout(i, j, 0);
				}
			}
		}
		Matrix internal = new Matrix(Utils.cloneMatrix(w));
		try {
			EigenvalueDecomposition evd = internal.eig();
	    	double[] reEigVals = evd.getRealEigenvalues();
	    	double[] imEigVals = evd.getImagEigenvalues();
	    	double[] eigVals = new double[nInternalUnits];
	    	double maxVal = Double.MIN_VALUE;
	    	for(int i = 0; i < reEigVals.length; i++) {
	    		eigVals[i] = Math.sqrt(reEigVals[i] * reEigVals[i] + imEigVals[i] * imEigVals[i]);
	    		if(maxVal < eigVals[i]) {
	    			maxVal = eigVals[i];
	    		}
	    	}
	    	if(maxVal < 0.001) {
	    		maxVal = 1.0;
	    	}
	    	internal = internal.times(1/maxVal);
	    	internal.times(rho);
	    	net.setW(internal.getArray());
	    	net.setNoiseLevel(near.noiseLevel);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return net;
	}

	@Override
	public double calculateDistance(MetaNEATGenome arepresentative) {
		NEARGenome representative = (NEARGenome)arepresentative;
		NEARGenome applicant = this;
		
		// Metric 1: node difference
		int nodeDifference = Math.abs(applicant.nInternalUnits - representative.nInternalUnits);
		double metric1 = (double)nodeDifference / (double)representative.nInternalUnits;
		
		// Metric 2: sparseness
		double applicantSparseness = (double)applicant.nActiveReservoirConns / (double)(applicant.nInternalUnits * applicant.nInternalUnits);
		double representativeSparseness = (double)representative.nActiveReservoirConns / (double)(representative.nInternalUnits * representative.nInternalUnits);
		double sparsenessDifference = Math.abs(applicantSparseness - representativeSparseness);
		double metric2 = sparsenessDifference / representativeSparseness;
		if(representative.nActiveReservoirConns == 0) {
			metric2 = 0;
		}
		
		// Metric 3: Spectral radius difference
		double spectralDifference = Math.abs(applicant.rho - representative.rho);
		double metric3 = spectralDifference / representative.rho;
		
		// Distance metric
		double dist = near.C1 * metric1 + near.C2 * metric2 + near.C3 * metric3;
		return dist; 
	}

	@Override
	public MetaNEATGenome clone() {
		NEARGenome clone = new NEARGenome();
		// Set the matrices
		clone.w = Utils.cloneMatrix(w);
		clone.wBack = Utils.cloneMatrix(wBack);
		clone.wIn = Utils.cloneMatrix(wIn);
		clone.wBool = Utils.cloneMatrix(wBool);
		clone.wOut = Utils.cloneMatrix(wOut);
		// Set NEAR
		clone.near = near;
		// Set species
		clone.setSpecies(getSpecies());
		// Fitness
		clone.fitness = 0d;
		clone.adjFitness = 0;
		// Set units
		clone.nInputUnits = nInputUnits;
		clone.nOutputUnits = nOutputUnits;
		clone.nInternalUnits = nInternalUnits;
		// Some more properties
		clone.rho = rho;
		clone.D = D;
		clone.nActiveReservoirConns = this.nActiveReservoirConns;
		clone.runningTally = runningTally;
		return clone;
	}

	@Override
	public void mutate() {
		// Mutate weights
		mutateWeights();
		// Mutate spectral radius
		if(Utils.rand.nextDouble() < near.mutateRHO) {
			if(Utils.rand.nextDouble() < 0.5) {
				rho += Utils.pertubation(0.1);
				rho = Math.min(NEAR.RHO_UB, Math.max(rho, NEAR.RHO_LB));
			} else {
				rho = NEAR.RHO_LB + Utils.rand.nextDouble() * (NEAR.RHO_UB - NEAR.RHO_LB);
			}
		}
		// Mutate sparseness
		if(Utils.rand.nextDouble() < near.mutateD) {
			if(Utils.rand.nextDouble() < 0.5) {
				D += Utils.pertubation(0.15);
				D = Math.min(NEAR.D_UB, Math.max(D, NEAR.D_LB));
			} else {
				D = NEAR.D_LB + Utils.rand.nextDouble() * (NEAR.D_UB - NEAR.D_LB);
			}
		}
		// Add node
		// In order to keep a balance mn must be analog to ml
		if(Utils.rand.nextDouble() < near.mn) {
			addNode();
		}
		// Add Connection
		if(Utils.rand.nextDouble() < near.ml) {
			mutateLinks();
		}
	}

	@Override
	public MetaNEATGenome xover(MetaNEATGenome aparent2) {
		NEARGenome offspring;
		// Choose
		double threshProb = 0.5;
		if(near.crossoverAdaptation) {
			threshProb = near.getFittestProb();
		}
		if(Utils.rand.nextDouble() >= threshProb) {
			offspring = xoverComplexify(this, (NEARGenome)aparent2);
			offspring.setOutOfCrossoverLargest(true);
		} else {
			offspring = xoverFittest(this, (NEARGenome)aparent2);
			offspring.setOutOfCrossoverFittest(true);
		}
		offspring.setParentFitness1(this.getFitness());
		offspring.setParentFitness2(aparent2.getFitness());
		return offspring;
	}

	@Override
	public void verify() {
		if(nInternalUnits > 1) {
			double realDensity = (double)nActiveReservoirConns / (double)(nInternalUnits * nInternalUnits);
			if(realDensity > D) {
				int correctConns = (int)Math.round((D * (double) (nInternalUnits * nInternalUnits)));
				if(correctConns > 0) {
					int difference = Math.abs(nActiveReservoirConns - correctConns);
					for(int i = 0; i < difference; i++) {
						removeConnection();
					}
				}
			} else {
				int correctConns = (int) (D * (double) (nInternalUnits * nInternalUnits));
				if(correctConns > 0) {
					int difference = Math.abs(correctConns - nActiveReservoirConns);
					for(int i = 0; i < difference; i++) {
						addConnection();
					}
				}
			}
		}
		if(near.darwinian) {
//			wOut = new double[nOutputUnits][nInputUnits + nInternalUnits];
			wOut = Utils.randomMatrixPlusMinus(nOutputUnits, nInputUnits + nInternalUnits, RAND_OUT);
//			wIn = Utils.randomMatrixPlusMinus(nInternalUnits, nInputUnits, RAND_INP);
		}
	}
	
	public NEARGenome(){}
	
//	public NEARGenome(NEAR anear) {
////		// 2D-MC-M
////		int nInputUnitsPre = 3;
////		int nOutputUnitsPre = 3;
////		int nInternalUnitsPre = 1;
////		D = 0.1312584347723066;
////		rho = 0.8842303966550449;
////		String reservoir = "-0.25129231458271695";
////		String input = "0.12037943148251107;-0.321475250859778;0.8060203240481948";
////		String output = "0.2599759099377215;0.28414053553898294;-0.69941154298671;-0.5199071508413967;-0.7245108705721948;0.5061466749554556;0.1593088563900827;-0.7593664348690393;0.10956776634233831;0.38002733668679517;2.661062788598457;0.09594427806365928";
////		int[] actionMap = {1,0,0,2,2};
////		int[] stateMap = {0,1,2,1,2};
//		
////		// 2D-MC-NM
////		int nInputUnitsPre = 2;
////		int nOutputUnitsPre = 3;
////		int nInternalUnitsPre = 4;
////		D = 0.5822778485454729;
////		rho = 0.6365033781314609;
////		String reservoir = "0.0;0.0;0.12003619302236157;0.6376508857323492;0.0;0.0;0.0;-0.6893132603698634;1.5541546366502623;-0.12354347450708913;1.3726849773770975;0.2278601876580566;0.6391216357911345;0.0;0.0;0.41061357524087";
////		String input = "-0.6476669683804106;0.646544325969509;-0.15227044792930444;0.8552213194969829;-0.22709590557562698;0.27035505152878225;0.27250717435941363;0.2981103572914343";
////		String output = "-1.771742770870008;0.7461014524091405;1.997821381308037;-0.03016885001211731;1.4261976843061903;0.6068276486632773;-2.259418082832017;0.585509895511679;0.24190617870556824;-0.21865547085337977;2.967860955879286;-0.03876774279606424;-1.501436159219397;0.6305809645379186;2.4076889274241493;0.48411866380823193;1.3639029189986325;0.004473630004649135";
////		int[] actionMap = {1,0,0,2,2};
////		int[] stateMap = {0,1,1};
//		
////		// SJS-2
//		int nInputUnitsPre = 9;
//		int nOutputUnitsPre = 8;
//		int nInternalUnitsPre = 2;
//		D = 0.9225161280527167;
//		rho = 0.9557711140079854;
//		String reservoir = "-0.28268293861248106;0.9443728182239803;-0.13023417316905028;-0.5117674826745188";
//		String input = "0.42991831775608147;0.9202898654764915;-0.7093995189492011;0.15032297635566416;0.9582046941115143;0.4358532650223222;0.7275943522443271;0.09725159465244015;0.4575353875952606;0.5155882116889681;-0.8489086160242703;0.7699563552531019;0.35579270275222097;0.7216967404106904;0.05240892676642561;-0.07982434331538757;-0.048464752266182964;-0.09692394753003142";
//		String output = "-3.8026520310404983;-4.854900731494238;2.0398542328138793;2.506018693299855;0.8307668185032381;-0.8184437775938086;3.8139848006573533;2.3544014060670397;2.6694852660270576;-0.467291635047545;-1.9169513337274537;-0.9543303788927899;-4.804480773688671;4.411162875510209;2.733463862279238;4.756033821054796;0.6634392663420877;0.8616246317016786;1.2944374414985318;2.1800446153099196;-0.8653652610899456;0.05003951655668469;-0.05949637355390599;4.938193293501659;1.380956892726515;-3.045294326830931;1.9108458290357675;3.5236515346877835;-0.9827189068998062;0.9911412640284318;-1.9408694889424132;-1.646023125127355;-2.173127473143053;-3.155401071126908;5.093838905911122;3.843013538402323;-0.2029460381107467;3.171782597972686;0.2725013402492611;1.181066491557915;4.364837182822271;0.4895536809598715;2.532855437056512;-2.050597601562904;-0.6904574214431567;2.1580760038926727;3.8267232346847933;7.631526937643799;0.462383560824733;-0.5890387544763852;4.66975609651392;0.8893470814308062;0.5806368615761257;2.058151738383223;1.1856460484669515;-1.8720583357435125;-2.9071339173960915;-0.6846355767485185;1.5430117821056806;2.199401481056779;1.3905264734122904;3.752216633911151;-1.5081758750875873;3.037226825900979;1.3342813685246075;-1.0382988910344244;-3.385411624217747;-1.740061724281003;5.271952300370771;-2.7781094657461463;0.8949217534735999;10.292128984037396;-0.17281255581287056;-0.11252232254626686;3.3460223868534387;4.327965395164602;0.48280734708573675;-0.40331885192206324;-0.2529039945061398;9.204780559306007;3.7831960339939648;6.247803963221278;11.959551576980942;13.38448028760419;-3.0223419350705814;0.5796287691571618;2.5277090536214497;-2.5144220796750507";
//		int[] actionMap = {0,1,2,3,0,1,2,3,4,5,6,7,4,5,6,7};
//		int[] stateMap = {0,1,2,3,4,1,2,3,4,5,6,7,8,5,6,7,8};
//		
//		boolean doubled = false;
//		boolean taylor = false;
//		
//		// Standard
//		near = anear;
//		nActiveReservoirConns = 0;
//		runningTally = 0;
//		nInputUnits = near.getNInputUnits();
//		nOutputUnits = near.getNOutputUnits();
//		if(doubled) {
//			nInternalUnits = nInternalUnitsPre * 2;
//		} else {
//			nInternalUnits = nInternalUnitsPre;
//		}
//		wBack = new double[nInternalUnits][nOutputUnits];
//		wOut = Utils.randomMatrixPlusMinus(nOutputUnits, nInputUnits + nInternalUnits, RAND_OUT);
//		// Initialize input matrix between -RAND_INP and RAND_INP
//		wIn = Utils.randomMatrixPlusMinus(nInternalUnits, nInputUnits, RAND_INP);
//		
//		// Parse old weights
//		String[]  reservoirTokens = reservoir.split("\\;");
//		String[]  inputTokens = input.split("\\;");
//		String[]  outputTokens = output.split("\\;");
//		double[][] winsmall = new double[nInternalUnitsPre][nInputUnitsPre];
//		double[][] wsmall = new double[nInternalUnitsPre][nInternalUnitsPre];
//		double[][] woutsmall = new double[nOutputUnitsPre][nInputUnitsPre + nInternalUnitsPre];
//		int counter = 0;
//		for(int i = 0; i < nInternalUnitsPre; i++) {
//			for(int j = 0; j < nInputUnitsPre; j++) {
//				winsmall[i][j] = Double.parseDouble(inputTokens[counter]);
//				counter++;
//			}
//		}
//		counter = 0;
//		for(int i = 0; i < nInternalUnitsPre; i++) {
//			for(int j = 0; j < nInternalUnitsPre; j++) {
//				wsmall[i][j] = Double.parseDouble(reservoirTokens[counter]);
//				counter++;
//			}
//		}
//		counter = 0;
//		for(int i = 0; i < nOutputUnitsPre; i++) {
//			for(int j = 0; j < (nInputUnitsPre + nInternalUnitsPre); j++) {
//				woutsmall[i][j] = Double.parseDouble(outputTokens[counter]);
//				counter++;
//			}
//		}
//		if(taylor) {
//			for(int i = 0; i < nInternalUnits; i++) {
//				for(int j = 0; j < nInputUnits; j++) {
//					wIn[i][j] = winsmall[i%nInternalUnitsPre][stateMap[j]];
//				}
//			}
//			for(int i = 0; i < nOutputUnits; i++) {
//				for(int j = 0; j < (nInputUnits + nInternalUnits); j++) {
//					if(j < nInputUnits) {
//						wOut[i][j] = woutsmall[actionMap[i]][stateMap[j]];
//					} else {
//						wOut[i][j] = woutsmall[actionMap[i]][nInputUnitsPre + (j%nInternalUnitsPre)];
//					}
//				}
//			}
//		}
//				
//		w = new double[nInternalUnits][nInternalUnits];
//		wBool = new boolean[nInternalUnits][nInternalUnits];
//		for(int i = 0; i < nInternalUnits; i++) {
//			for(int j = 0; j < nInternalUnits; j++) {
//				if(i < nInternalUnitsPre && j < nInternalUnitsPre) {
//					w[i][j] = wsmall[i%nInternalUnitsPre][j%nInternalUnitsPre];
//					if(w[i][j] < 0.0 || w[i][j] > 0.0) {
//						wBool[i][j] = true;
//						runningTally += w[i][j];
//						nActiveReservoirConns++;
//					}
//				}
//				if(doubled) {
//					if(i >= nInternalUnitsPre && j >= nInternalUnitsPre) {
//						w[i][j] = wsmall[i%nInternalUnitsPre][j%nInternalUnitsPre];
//						if(w[i][j] < 0.0 || w[i][j] > 0.0) {
//							wBool[i][j] = true;
//							runningTally += w[i][j];
//							nActiveReservoirConns++;
//						}
//					}
//				}
//			}
//		}
//		
//	}
	
	public NEARGenome(NEAR anear) {
		near = anear;
		nActiveReservoirConns = 0;
		runningTally = 0;
		nInputUnits = near.getNInputUnits();
		nOutputUnits = near.getNOutputUnits();
		nInternalUnits = near.initIntNodes;
		rho = NEAR.RHO_LB + Utils.rand.nextDouble() * (NEAR.RHO_UB - NEAR.RHO_LB);
		D = NEAR.D_LB + Utils.rand.nextDouble() * (NEAR.D_UB - NEAR.D_LB);
		wBack = new double[nInternalUnits][nOutputUnits];
//		wOut = new double[nOutputUnits][nInputUnits + nInternalUnits];
		wOut = Utils.randomMatrixPlusMinus(nOutputUnits, nInputUnits + nInternalUnits, RAND_OUT);
		// Initialize input matrix between -RAND_INP and RAND_INP
		wIn = Utils.randomMatrixPlusMinus(nInternalUnits, nInputUnits, RAND_INP);
		// Initialize the reservoir
		w = new double[nInternalUnits][nInternalUnits];
		wBool = new boolean[nInternalUnits][nInternalUnits];
		for(int i = 0; i < nInternalUnits; i++) {
			for(int j = 0; j < nInternalUnits; j++) {
				if(Utils.rand.nextDouble() < D) {
					w[i][j] = (2 * Utils.rand.nextDouble() - 1) * RAND_INT;
					wBool[i][j] = true;
					runningTally += w[i][j];
					nActiveReservoirConns++;
				} else {
					w[i][j] = 0.0;
					wBool[i][j] = false;
				}
			}
		}
	}
	
	public double getSparseness() {
		return D;
	}
	
	public void setSparseness(double aD) {
		D = aD;
	}

	public int getActiveReservoirConns() {
		return nActiveReservoirConns;
	}
	
	public void setActiveReservoirConns(int arc) {
		nActiveReservoirConns = arc;
	}
	
	public double getSpectralRadius() {
		return rho;
	}
	
	public void setSpectralRadius(double arho) {
		rho= arho;
	}

	public double[][] getWIn() {
		return wIn;
	} 
	
	public void setWIn(double[][] aWIn) {
		wIn = aWIn;
	}
	
	public double[][] getW() {
		return w;
	}
	
	public void setW(double[][] aW) {
		w = aW;
	}
	
	public double[][] getWBack() {
		return wBack;
	}
	
	public void setWBack(double[][] aWBack) {
		wBack = aWBack;
	}
	
	public double[][] getWOut() {
		return wOut;
	}
	
	public void setWOut(double[][] aWOut) {
		wOut = aWOut;
	}
	
	public boolean[][] getWBool() {
		return wBool;
	}
	
	public void setWBool(boolean[][] aWBool) {
		wBool = aWBool;
	}
	
	public double getRunningTally() {
		return runningTally;
	}
	
	public void setRunningTally(double atally) {
		runningTally = atally;
	}
	
	public int getNInternalUnits() {
		return nInternalUnits;
	}

	private static NEARGenome xoverComplexify(NEARGenome parent1, NEARGenome parent2) {
		// Initialize a genome
		NEARGenome offspring = new NEARGenome();
		if(parent1.getFitness() > parent2.getFitness()) {
			offspring.setSpectralRadius(parent1.rho);
			offspring.setSparseness(parent1.D);
			offspring.near = parent1.near;
		} else {
			offspring.setSpectralRadius(parent2.rho);
			offspring.setSparseness(parent2.D);
			offspring.near = parent2.near;
		}
		// Initialize some temporary variables
		int activeConns = 0;
		int minInternalUnits = Math.min(parent1.nInternalUnits, parent2.nInternalUnits);
		int maxInternalUnits = Math.max(parent1.nInternalUnits, parent2.nInternalUnits);
		int nIntUnits = maxInternalUnits; // Complexify
		
		double[][] xoverW;
		double[][] xoverWIn;
		double[][] xoverWOut;
		boolean[][] xoverWBool;
		
		// Complexify the offspring => choose the largest reservoir
		if(parent1.nInternalUnits >= parent2.nInternalUnits) {
			xoverW = Utils.cloneMatrix(parent1.w);
			xoverWIn = Utils.cloneMatrix(parent1.wIn);
			xoverWOut = Utils.cloneMatrix(parent1.wOut);
			xoverWBool = Utils.cloneMatrix(parent1.wBool);
			offspring.nInputUnits = parent1.nInputUnits;
			offspring.nOutputUnits = parent1.nOutputUnits;
		} else {
			xoverW = Utils.cloneMatrix(parent2.w);
			xoverWIn = Utils.cloneMatrix(parent2.wIn);
			xoverWOut = Utils.cloneMatrix(parent2.wOut);
			xoverWBool = Utils.cloneMatrix(parent2.wBool);
			offspring.nInputUnits = parent2.nInputUnits;
			offspring.nOutputUnits = parent2.nOutputUnits;
		}
		offspring.nInternalUnits = nIntUnits;

		
		// Allign reservoirs based on historical/structural markings
		
		// Xover the reservoir based on historical markings
		double runningTally = 0;
		for(int i = 0; i < minInternalUnits; i++) {
			for(int j = 0; j < minInternalUnits; j++) {
				double value = 0.0;
				// (matching)
				if(parent1.wBool[i][j] && parent2.wBool[i][j]) {
					if(Utils.rand.nextDouble() < 0.5) {
						value = (parent1.w[i][j] + parent2.w[i][j]) / 2.0;
					} else {
						if(Utils.rand.nextDouble() < 0.5) {
							value = parent1.w[i][j];
						} else {
							value = parent2.w[i][j];
						}
					}
				// (disjoint)
				} else if(parent1.wBool[i][j]) {
					value = parent1.w[i][j];
				} else if(parent2.wBool[i][j]){
					value = parent2.w[i][j];
				}
				xoverW[i][j] = value;
				runningTally += value;
				if(parent1.wBool[i][j] || parent2.wBool[i][j]) {
					activeConns++;
					xoverWBool[i][j] = true;
				}
			}
		}
		// Calculate the rest of the reservoir (excess)
		for(int i = minInternalUnits; i < maxInternalUnits; i++) {
			for(int j = minInternalUnits; j < maxInternalUnits; j++) {
				if(xoverWBool[i][j]) {
					activeConns++;
					runningTally += xoverW[i][j];
				}
			}
		}
		offspring.setW(xoverW);
		offspring.setWBool(xoverWBool);
		offspring.setActiveReservoirConns(activeConns);
		offspring.setRunningTally(runningTally);
		
		// Xover the input
		for(int i = 0; i < minInternalUnits; i++) {
			for(int j = 0; j < offspring.nInputUnits; j++) {
				if(Utils.rand.nextDouble() < 0.5) {
					xoverWIn[i][j] = (parent1.wIn[i][j] + parent2.wIn[i][j]) / 2.0;
				} else {
					if(Utils.rand.nextDouble() < 0.5) {
						xoverWIn[i][j] = parent1.wIn[i][j];
					} else {
						xoverWIn[i][j] = parent2.wIn[i][j];
					}
				}
			}
		}
		offspring.setWIn(xoverWIn);
		// Xover the output
		for(int i = 0; i < offspring.nOutputUnits; i++) {
			for(int j = 0; j < offspring.nInputUnits + minInternalUnits; j++) {
				if(Utils.rand.nextDouble() < 0.5) {
					xoverWOut[i][j] = (parent1.wOut[i][j] + parent2.wOut[i][j]) / 2.0;
				} else {
					if(Utils.rand.nextDouble() < 0.5) {
						xoverWOut[i][j] = parent1.wOut[i][j];
					} else {
						xoverWOut[i][j] = parent2.wOut[i][j];
					}
				}
			}
		}
		offspring.setWOut(xoverWOut);
		offspring.setWBack(new double[nIntUnits][offspring.nOutputUnits]);
		return offspring;
	}
	
	private static NEARGenome xoverFittest(NEARGenome parent1, NEARGenome parent2) {
		NEARGenome offspring = new NEARGenome();
		int fitterUnits = 0;
		if(parent1.getFitness() > parent2.getFitness()) {
			offspring.setSpectralRadius(parent1.rho);
			offspring.setSparseness(parent1.D);
			offspring.near = parent1.near;
			fitterUnits = parent1.nInternalUnits;
			offspring.nInputUnits = parent1.nInputUnits;
			offspring.nOutputUnits = parent1.nOutputUnits;
		} else {
			offspring.setSpectralRadius(parent2.rho);
			offspring.setSparseness(parent2.D);
			offspring.near = parent2.near;
			fitterUnits = parent2.nInternalUnits;
			offspring.nInputUnits = parent2.nInputUnits;
			offspring.nOutputUnits = parent2.nOutputUnits;
		}
		
		// Initialize some temporary variables
		int activeConns = 0;
		int minInternalUnits = Math.min(parent1.nInternalUnits, parent2.nInternalUnits);
		int maxInternalUnits = Math.max(parent1.nInternalUnits, parent2.nInternalUnits);
		double runningTally = 0;
		
		double[][] xoverW;
		double[][] xoverWIn;
		double[][] xoverWOut;
		boolean[][] xoverWBool;
		
		// Choose the fittest reservoir
		if(parent1.getFitness() > parent2.getFitness()) {
			xoverW = Utils.cloneMatrix(parent1.w);
			xoverWIn = Utils.cloneMatrix(parent1.wIn);
			xoverWOut = Utils.cloneMatrix(parent1.wOut);
			xoverWBool = Utils.cloneMatrix(parent1.wBool);
		} else {
			xoverW = Utils.cloneMatrix(parent2.w);
			xoverWIn = Utils.cloneMatrix(parent2.wIn);
			xoverWOut = Utils.cloneMatrix(parent2.wOut);
			xoverWBool = Utils.cloneMatrix(parent2.wBool);
		}
		
		// Allign reservoirs based on historical/structural markings
		
		// Xover the reservoir based on historical markings
		for(int i = 0; i < minInternalUnits; i++) {
			for(int j = 0; j < minInternalUnits; j++) {
				double value = 0.0;
				boolean flag = false;
				if(parent1.wBool[i][j] || parent2.wBool[i][j]) {
					flag = true;
				}
				if(parent1.wBool[i][j] && parent2.wBool[i][j]) {
					if(Utils.rand.nextDouble() < 0.5) {
						value = (parent1.w[i][j] + parent2.w[i][j]) / 2.0;
					} else {
						if(Utils.rand.nextDouble() < 0.5) {
							value = parent1.w[i][j];
						} else {
							value = parent2.w[i][j];
						}
					}
				} else if(parent1.wBool[i][j]) {
					value = parent1.w[i][j];
				} else if(parent2.wBool[i][j]){
					value = parent2.w[i][j];
				}
				xoverW[i][j] = value;
				xoverWBool[i][j] = flag;
				runningTally += value;
				if(flag) {
					activeConns++;
				}
			}
		}
		// Calculate the rest of the reservoir if needed
		if(fitterUnits == maxInternalUnits) {
			for(int i = minInternalUnits; i < maxInternalUnits; i++) {
				for(int j = minInternalUnits; j < maxInternalUnits; j++) {
					if(xoverWBool[i][j]) {
						activeConns++;
						runningTally += xoverW[i][j];
					}
				}
			}
		}
		offspring.setW(xoverW);
		offspring.setWBool(xoverWBool);
		offspring.setActiveReservoirConns(activeConns);
		offspring.nInternalUnits = fitterUnits;
		offspring.setRunningTally(runningTally);
		
		// Xover the input
		for(int i = 0; i < minInternalUnits; i++) {
			for(int j = 0; j < offspring.nInputUnits; j++) {
				if(Utils.rand.nextDouble() < 0.5) {
					xoverWIn[i][j] = (parent1.wIn[i][j] + parent2.wIn[i][j]) / 2.0;
				} else {
					if(Utils.rand.nextDouble() < 0.5) {
						xoverWIn[i][j] = parent1.wIn[i][j];
					} else {
						xoverWIn[i][j] = parent2.wIn[i][j];
					}
				}
			}
		}
		offspring.setWIn(xoverWIn);

		// Xover the output
		for(int i = 0; i < offspring.nOutputUnits; i++) {
			for(int j = 0; j < offspring.nInputUnits + minInternalUnits; j++) {
				if(Utils.rand.nextDouble() < 0.5) {
					xoverWOut[i][j] = (parent1.wOut[i][j] + parent2.wOut[i][j]) / 2.0;
				} else {
					if(Utils.rand.nextDouble() < 0.5) {
						xoverWOut[i][j] = parent1.wOut[i][j];
					} else {
						xoverWOut[i][j] = parent2.wOut[i][j];
					}
				}
			}
		}
		offspring.setWOut(xoverWOut);
		offspring.setWBack(new double[fitterUnits][offspring.nOutputUnits]);
		return offspring;
	}
	
	
	private void addNode() {
		nInternalUnits++;
		// new back matrix
		wBack = new double[nInternalUnits][nOutputUnits];
		// Start with internal connections
		double[][] wNew = new double[nInternalUnits][nInternalUnits];
		boolean[][] wBoolNew = new boolean[nInternalUnits][nInternalUnits];
		for(int i = 0; i < w.length; i++) {
			for(int j = 0; j < w[i].length; j++) {
				if(wBool[i][j]) {
					wNew[i][j] = w[i][j];
					wBoolNew[i][j] = true;
				} else {
					wNew[i][j] = 0.0;
					wBoolNew[i][j] = false;
				}
			}
		}
		wBool = wBoolNew;
		w = wNew;
		// Continue with input units
		double[][] wInNew = Utils.randomMatrixPlusMinus(nInternalUnits, nInputUnits, RAND_INP);
		for(int i = 0; i < wIn.length; i++) {
			for(int j = 0; j < wIn[i].length; j++) {
					wInNew[i][j] = wIn[i][j];
			}
		}
		wIn = wInNew;
		// Continue with output units
		double[][] wOutNew = new double[nOutputUnits][nInputUnits + nInternalUnits];
		for(int i = 0; i < wOut.length; i++) {
			for(int j = 0; j < wOut[i].length; j++) {
					wOutNew[i][j] = wOut[i][j];
			}
		}
		wOut = wOutNew;
	}
	
	private void addConnection() {
		int tries = 50;
		boolean flag = false;
		while(tries > 0 && !flag) {
			int row = Utils.rand.nextInt(nInternalUnits);
			int col = Utils.rand.nextInt(nInternalUnits);
			if(!wBool[row][col]) {
				if(runningTally > 0) {
					w[row][col] = -Utils.rand.nextDouble();
				} else {
					w[row][col] = Utils.rand.nextDouble();
				}
				runningTally += w[row][col];
				wBool[row][col] = true;
				flag = true;
				nActiveReservoirConns++;
			} else {
				tries--;
			}
		}
	}
	
	private void removeConnection() {
		int tries = 50;
		boolean flag = false;
		while(tries > 0 && !flag) {
			int row = Utils.rand.nextInt(nInternalUnits);
			int col = Utils.rand.nextInt(nInternalUnits);
			if(wBool[row][col]) {
				runningTally -= w[row][col];
				w[row][col] = 0.0;
				wBool[row][col] = false;
				flag = true;
				nActiveReservoirConns--;
			} else {
				tries--;
			}
		}
	}
	
	private void mutateLinks() {
		addConnection();
	}
	
	private double runningTallyPerturbation(double value, double pertube, double maxWeight) {
		if(runningTally < 0) {
			// Positive
			// If it goes over max, find something smaller
			if((value + pertube) > maxWeight) {
				pertube = (maxWeight - value) * Utils.rand.nextDouble();
			}
			return value + pertube;
		} else {
			// Negative
			// If it goes below min, find something smaller
			if((value - pertube) < -maxWeight) {
				pertube = (value + maxWeight) * Utils.rand.nextDouble();
			}
			return value - pertube;
		}
	}
	
	private double perturbation(double value, double pertube, double maxWeight) {
		if(Utils.rand.nextBoolean()) {
			// Positive
			// If it goes over max, find something smaller
			if((value + pertube) > maxWeight) {
				pertube = (maxWeight - value) * Utils.rand.nextDouble();
			}
			return value + pertube;
		} else {
			// Negative
			// If it goes below min, find something smaller
			if((value - pertube) < -maxWeight) {
				pertube = (value + maxWeight) * Utils.rand.nextDouble();
			}
			return value - pertube;
		}
	}
	
	private void mutateWeights() {
		// Shake up thing on internal units and input units
		// Keep things between -1 and 1 by perturbing and once in a while change everything
		// Input Weights
		for(int i = 0; i < wIn.length;i ++) {
			for(int j = 0; j < wIn[i].length; j++) {
				// Mutate weight by some perturbation
				if(Utils.rand.nextDouble() < near.mutateLinkWeights) {
					double oldVal = wIn[i][j];
					double pertubation = Utils.pertubation(near.weightMutationPower);  
					wIn[i][j] = perturbation(oldVal, pertubation, MAX_WEIGHT);
				}
				// Some time restart it
				if(Utils.rand.nextDouble() > 0.5) {
					wIn[i][j] = (2.0 * Utils.rand.nextDouble() - 1.0) * RAND_INP;
				}
			}
		}
		
		// Find the rms value
		double rms = 0;
		for(int i = 0; i < wOut.length;i ++) {
			for(int j = 0; j < wOut[i].length; j++) {
				rms += wOut[i][j]*wOut[i][j];	
			}
		}
		rms = rms / (nOutputUnits * (nInputUnits + nInternalUnits));
		rms = Math.sqrt(rms);
		for(int i = 0; i < wOut.length;i ++) {
			for(int j = 0; j < wOut[i].length; j++) {
				// Mutate weight by some perturbation
				if(Utils.rand.nextDouble() < near.mutateLinkWeights) {
					double oldVal = wOut[i][j];
					double pertube = Utils.rand.nextDouble() * near.weightMutationPower;
					wOut[i][j] = perturbation(oldVal, pertube, MAX_OUTPUT);
				}
				// Some time restart it
				if(Utils.rand.nextDouble() > 0.5) {
					wOut[i][j] = (2.0 * Utils.rand.nextDouble() - 1.0) * rms;
				}
			}
		}
		
		// Internal Weights
		for(int i = 0; i < w.length;i ++) {
			for(int j = 0; j < w[i].length; j++) {
				if(wBool[i][j]) {
					// Mutate weight by some perturbation
					if(Utils.rand.nextDouble() < near.mutateLinkWeights) {
						double oldVal = w[i][j];
						double pertube = Utils.rand.nextDouble() * near.weightMutationPower;
						w[i][j] = runningTallyPerturbation(oldVal, pertube, MAX_WEIGHT);
						runningTally -= oldVal;
						runningTally += w[i][j];
					}
					if(Utils.rand.nextDouble() > 0.5) {
						// Some time restart it
						if(runningTally < 0) {
							runningTally -= w[i][j];
							w[i][j] = Utils.rand.nextDouble() * RAND_INT;
							runningTally += w[i][j];
						} else {
							runningTally -= w[i][j];
							w[i][j] = -Utils.rand.nextDouble() * RAND_INT;
							runningTally += w[i][j];
						}
					}
				}
			}
		}
		
	}
	
	public double getParentFitness1() {
		return parentFitness1;
	}
	
	public double getParentFitness2() {
		return parentFitness2;
	}
	
	public void setParentFitness1(double afitness) {
		parentFitness1 = afitness;
	}
	
	public void setParentFitness2(double afitness) {
		parentFitness2 = afitness;
	}
	
	public boolean getOutOfCrossoverLargest() {
		return outOfCrossoverLargest;
	}
	
	public void setOutOfCrossoverLargest(boolean aoutOfCrossoverLargest) {
		outOfCrossoverLargest = aoutOfCrossoverLargest;
	}
	
	public boolean getOutOfCrossoverFittest() {
		return outOfCrossoverFittest;
	}
	
	public void setOutOfCrossoverFittest(boolean aoutOfCrossoverFittest) {
		outOfCrossoverFittest = aoutOfCrossoverFittest;
	}

	@Override
	public int getNumberOfNodes() {
		return nInternalUnits + nInputUnits + nOutputUnits;
	}

	@Override
	public int getNumberOfConnections() {
		return nActiveReservoirConns + nInputUnits * nInternalUnits + nOutputUnits * (nInternalUnits + nInputUnits);
	}

	@Override
	public int getActiveConnections() {
		return this.nActiveReservoirConns;
	}

	@Override
	public String message(String astring) {
		String[] tokens = astring.split("\\;");
		int counter = 0;
		for(int i = 0; i < nOutputUnits; i++) {
			for(int j = 0; j < (nInputUnits + nInternalUnits); j++) {
				wOut[i][j] = Double.parseDouble(tokens[counter]);
				counter++;
			}
		}
		return null;
	}
	
	public String toString() {
		String astring = new String();
		astring += "D: " + D + " | ";
		astring += "rho: " + rho + " | ";
		astring += "N: " + nInternalUnits + ";";
		astring += "\n";
		astring += "R:";
		for(int i = 0; i < nInternalUnits; i++) {
			for(int j = 0; j < nInternalUnits; j++) {
				astring += w[i][j] + ";";
			}
		}
		astring += "\n";
		astring += "I:";
		for(int i = 0; i < nInternalUnits; i++) {
			for(int j = 0; j < nInputUnits; j++) {
				astring += wIn[i][j] + ";";
			}
		}
		astring += "\n";
		astring += "O:";
		for(int i = 0; i < nOutputUnits; i++) {
			for(int j = 0; j < (nInputUnits + nInternalUnits); j++) {
				astring += wOut[i][j] + ";";
			}
		}
		return astring;
	}
	
}
