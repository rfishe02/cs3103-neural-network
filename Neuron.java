
import java.util.ArrayList;
import java.util.Random;
import java.io.Serializable;

public class Neuron implements Serializable {

	private static final long serialVersionUID = 1L;
	private ArrayList<Double> w;
	private double in;
	private double a;

	// Setters & Getters

	public void setW(ArrayList<Double> weights) {
		this.w = weights;
	}

	public void setIn(double in) {
		this.in = in;
	}

	public void setA(double a) {
		this. a = a;
	}

	public ArrayList<Double> getW() {
		return w;
	}

	public double getIn() {
		return in;
	}

	public double getA() {
		return a;
	}

	/** Generate random input weights for the neuron in that layer. */

	public ArrayList<Double> genWeights(int size) {
		ArrayList<Double> weights = new ArrayList<>(size);
		double w;

		for(int i = 0; i < size; i++) {
			w = new Random().nextDouble() - new Random().nextDouble();
			weights.add(w);
		}

		return weights;
	}

}
