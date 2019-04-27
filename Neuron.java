
import java.util.ArrayList;
import java.util.Random;

public class Neuron {

	private ArrayList<Double> in;
	private ArrayList<Double> out;

	public Neuron() {

	}

	public ArrayList<Double> genWeights(int size) {

		ArrayList<Double> weights = new ArrayList<>(size);

		for(int i = 0; i < size; i++) {
			weights.add(new Random().nextDouble());
		}

		return weights;

	}

	/* Setters & Getters */

	public void setInput(ArrayList<Double> weights) {
		this.in = weights;
	}

	public void setOutput(ArrayList<Double> weights) {
		this.out = weights;
	}

	public ArrayList<Double> getInput() {
		return in;
	}

	public ArrayList<Double> getOutput() {
		return out;
	}

}
