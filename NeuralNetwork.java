
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Stack;

public class NeuralNetwork {

	//private Layer outputLayer;
	private ArrayList<Layer> layers;
	private int inputLayerCount;
	private int hiddenLayerCount;
	private int ouputLayerCount;
	double eta = -0.125;

	/** */

	public void buildNetwork(int input, int hidden, int width, int output) {

		Layer h;

		// Set up the layers.
		// The input layer represents a separate input vector.
		// The ouput layer is the last layer of the ArrayList.

		// Make hidden layers.
		// Each node in the hidden layer has input weights, and an output.

		ArrayList<Layer> layers = new ArrayList<>(width);

		for (int i = 0; i < width; i++) {
			h = new Layer();

			if (i == 0) {
				h.setNeuronCount(hidden);
				h.makeLayers(input,hidden);

			} else if (i == width-1) {
				h.setNeuronCount(output);
				h.makeLayers(hidden,output);

			}
			else {
				h.setNeuronCount(hidden);
				h.makeLayers(hidden,hidden);

			}

			layers.add(h);
		}

		this.layers = layers;

		// Store layer counts.

		this.inputLayerCount = input;
		this.hiddenLayerCount = width;
		this.ouputLayerCount = output;

	}

	// =======================================================

	public void forwardPass(ArrayList<Double> x) {

		Neuron z;
		double sum;

		// Calculate the ouput of the first ouput & hidden layers

		for (int l = 0; l < layers.size(); l++) { // For each hidden layer

			if (l == 0) { // If it's the first hidden layer

				for (Neuron n : layers.get(l).getNeurons()) { // For each neuron in the first hidden layer

					sum = 0;

					for(int w = 0; w < n.getInput().size(); w++) { // For each weight in that layer

						sum += x.get(w) * n.getInput().get(w); // weight_w * input_w

					}

					n.getOutput().set(0, activationFunction(sum));

				}

			} else { // It's in the other layers

				for (Neuron n : layers.get(l).getNeurons()) { // For each neuron in that layer

					sum = 0;

					for(int w = 0; w < n.getInput().size(); w++) {

						z = layers.get(l-1).getNeurons().get(w);
						sum += z.getOutput().get(0) * n.getInput().get(w);

					}

					n.getOutput().set(0, activationFunction(sum));
				}

			}

		}

	}

	// =======================================================

	public double activationFunction(double value) {
		return 1 / (1 + Math.pow(Math.E, value));
	}

	public double activationFunctionDerivative(double value) {
		return activationFunction(value) * (1 - activationFunction(value));
	}

	// =======================================================

	public void backpropagate(double target) {

		Stack<ArrayList<Double>> delta = new Stack<ArrayList<Double>>();
		ArrayList<Double> tmp;
		Neuron z;
		double d;
		double output;
		double sum;
		double weight;

		for (int l = layers.size() - 1; l >= 0; l--) { // For each layer, from the last to first

			tmp = new ArrayList<>(layers.get(l).getNeurons().size());

			if(l == layers.size() - 1) { // If it's the last layer

				for(int n = 0; n < layers.get(l).getNeuronCount(); n++) { // For each neuron in the last layer

					z = layers.get(l).getNeurons().get(n);
					output = z.getOutput().get(0);
					d = output * (1 - output) * (output - target); // Calculate delta_k = O_k (1 - O_k) (O_k - t_k) for output
					tmp.add(d);

				} // End for loop

			} else { // It's the other layers

				for(int n = 0; n < layers.get(l).getNeuronCount(); n++) { // For each neuron

					z = layers.get(l).getNeurons().get(n);
					output = z.getOutput().get(0);
					d = output * (1-output);

					sum = 0;

					for(int k = 0; k < delta.peek().size(); k++) { // The summation of delta k

						sum += delta.peek().get(k) * z.getInput().get(k); // Calculate delta_j = O_j (1-O_j) * Summation delta_k * W_jk for hidden

					}

					tmp.add(sum);

				} // End for loop

			}

			delta.add(tmp);

		} // End for loop

		// Update the weights
		// -eta * delta_l * O_(l-1)

		for(int l = 0; l < layers.size(); l++) { // For each layer

			tmp = delta.pop();

			for(int n = 0; n < layers.get(l).getNeuronCount(); n++) { // For each neuron

				z = layers.get(l).getNeurons().get(n);

				for(int w = 0; w < z.getInput().size(); w++) { // For each weight

					weight = eta * tmp.get(n) * layers.get(l).getNeurons().get(n).getOutput().get(0);
					z.getInput().set(w, z.getInput().get(w) + weight);

				}

			}

		} // End for loop

	}

	/////////////////////////

	public ArrayList<Layer> getLayers() {
		return layers;
	}

	public void printWeights() {

		for(Layer l : layers) {

			for(Neuron n : l.getNeurons()) {

				for(Double w : n.getInput()) {

					System.out.printf("%4.2f ",w);

				}

				System.out.println();

			}
			System.out.println();

		}
	}

}