
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

	public void makeLayers(int prev, int next) {

		ArrayList<Neuron> neurons = new ArrayList<>();
		Neuron n;

		for(int a = 0; a < neuronCount-1; a++ ) {

			n = new Neuron();

			n.setInput(n.genWeights(prev));
			n.setOutput(new ArrayList<Double>(1));
			n.getOutput().add(0.0);

			neurons.add(n);

		}

		////////////////////////////////////
		// Create the bias neuron.

		n = new Neuron();
		n.setOutput(new ArrayList<Double>(1));
		n.getOutput().add(1.0);

		neurons.add(n);

		////////////////////////////////////

		this.neurons = neurons;

	}

}
