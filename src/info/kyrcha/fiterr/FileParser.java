package info.kyrcha.fiterr;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;

/**
 * Helper class for parsing files, mainly for classification and time-series tasks
 * 
 * @author Kyriakos C. Chatzidimitriou (EMAIL - kyrcha [at] gmail (dot) com, WEB - http://kyrcha.info)
 *
 */
public class FileParser {
	
	public double[][] readFileOfDouble(String filename, String delim, int nattr) {
		List<Instance> list = new ArrayList<Instance>();
		int colDim = nattr;
		Scanner r = new Scanner(filename);
		while(r.hasNextLine()) {
			String line = r.nextLine();
			line.trim();
			String[] tokens = line.split(delim + "*");
			Instance inst = new Instance(nattr);
			inst.setInstance(tokens);
			list.add(inst);	
		}
		int rowDim = list.size();
		Iterator<Instance> iter =  list.iterator();
		double[][] data = new double[rowDim][colDim];
		int counter = 0;
		while(iter.hasNext()) {
			data[counter] = iter.next().getInstance();
			counter++;
		}
		return data;
	}

}
