
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Layer {

	ArrayList<Neuron> neurons;
	int neuronCount;

	// Setters & Getters

	public void setNeurons(ArrayList<Neuron> list) {
		neurons = list;
	}

	public ArrayList<Neuron> getNeurons() {
		return neurons;
	}

	public void setNeuronCount(int count) {
		neuronCount = count;
	}

	public int getNeuronCount() {
		return neuronCount;
	}

 	/** Create a layer with the given number of neurons, that will have input and output weights */

	public void makeLayers(int prev) {

		ArrayList<Neuron> neurons = new ArrayList<>();
		Neuron n;

		for(int a = 0; a < neuronCount; a++ ) {

			n = new Neuron();

			n.setW(n.genWeights(prev));
			n.setIn(0);
			n.setA(0);

			neurons.add(n);

		}

		this.neurons = neurons;

	}

}
