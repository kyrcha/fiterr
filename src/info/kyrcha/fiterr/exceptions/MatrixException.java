package info.kyrcha.fiterr.exceptions;

/**
 * 
 * @author Kyriakos C. Chatzidimitriou (EMAIL - kyrcha [at] gmail (dot) com, WEB - http://kyrcha.info)
 *
 */
public class MatrixException extends Exception {
	
	private static final long serialVersionUID = 2L;

	private String msg;

    public MatrixException(String str) {
        super(str);
    }
    
    public String toString(){
        return "Matrix Exception - " + msg;
    }	

}
