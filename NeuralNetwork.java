
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Stack;

public class NeuralNetwork {

	private ArrayList<Layer> bias;
	private ArrayList<Layer> layers;
	private int inputLayerCount;
	private int hiddenLayerCount;
	private int ouputLayerCount;
	double eta = -0.40;

	public ArrayList<Layer> getLayers() {
		return layers;
	}

	// -----------------------------------------------------

	public double activationFunction(double value) {
		return 1 / (1 + Math.pow(Math.E, value));
	}

	public double activationFunctionDerivative(double value) {
		return activationFunction(value) * (1 - activationFunction(value));
	}

	public double error(double output, double target) {
		return target - output;
	}

	/** Build a network with the given number of input, hidden, and output neurons in each layer.
			The input layer represents a separate input vector.
			Each node in the hidden layer has input weights, and an output.
	*/

	public void buildNetwork(int input, int hidden, int width, int output) {

		Layer b;
		Layer h;

		ArrayList<Layer> bLayers = new ArrayList<>(width);
		ArrayList<Layer> layers = new ArrayList<>(width);

		for (int i = 0; i < width; i++) {

			b = new Layer();
			h = new Layer();

			if (i == 0) {
				h.setNeuronCount(hidden);
				h.makeLayers(input,1);

			} else if (i == width-1) {
				h.setNeuronCount(output);
				h.makeLayers(hidden,1);

			}
			else {
				h.setNeuronCount(hidden);
				h.makeLayers(hidden,1);

			}

			b.setNeuronCount(1);
			b.makeLayers(hidden,1);

			bLayers.add(b);
			layers.add(h);

		}

		this.bias = bLayers;
		this.layers = layers;

		this.inputLayerCount = input;
		this.hiddenLayerCount = width;
		this.ouputLayerCount = output;

	}

	/** Conduct a forward pass through the network. This includes a bias node.

	*/

	public void forwardPass(ArrayList<Double> x) {

		Neuron z;
		double sum;

		for (int l = 0; l < layers.size(); l++) { // For each layer

			for(int n = 0; n < layers.get(l).getNeurons().size(); n++) { // For each neuron in that layer

				sum = 0;

				for(int w = 0; w < layers.get(l).getNeurons().get(n).getInput().size(); w++) { // For each weight in that neuron

					if(l == 0) { // It's in the first layer

							sum += x.get(w) * layers.get(l).getNeurons().get(n).getInput().get(w);

					} else { // It's in the other layers

							z = layers.get(l-1).getNeurons().get(w);
							sum += z.getOutput().get(0) * layers.get(l).getNeurons().get(n).getInput().get(w);

					}

				}

				sum += 1 * bias.get(l).getNeurons().get(0).getInput().get(n); // Add the bias to each node ouput
				layers.get(l).getNeurons().get(n).getOutput().set(0, activationFunction(sum));

			}

		}

	}

	/** Conduct the backpropagation phase, and update weights. */

	public void backpropagate(ArrayList<Double> input, ArrayList<Double> target) {

		Stack<ArrayList<Double>> delta = new Stack<ArrayList<Double>>();
		ArrayList<Double> tmp;
		Neuron b;
		Neuron z;
		double d;
		double output;
		double sum;
		double weight;

		for (int l = layers.size() - 1; l >= 0; l--) { // For each layer, from the last to first

			tmp = new ArrayList<>(layers.get(l).getNeurons().size());

			for(int n = 0; n < layers.get(l).getNeuronCount(); n++) { // For each neuron in that layer

				if(l == layers.size() - 1) { // If it's the last layer

					z = layers.get(l).getNeurons().get(n);
					output = z.getOutput().get(0);
					d = output * (1 - output) * error(output,target.get(n)); // Calculate delta_k = O_k (1 - O_k) (O_k - t_k) for output
					tmp.add(d);

				} else { // It's in the other layers.

					z = layers.get(l).getNeurons().get(n);
					output = z.getOutput().get(0);
					d = output * (1-output);

					sum = 0;

					for(int k = 0; k < delta.peek().size(); k++) { // For each k

						sum += delta.peek().get(k) * layers.get(l+1).getNeurons().get(k).getInput().get(n); // Calculate O_j (1- O_j) Summation delta_k * W_j+1 k

					}

					tmp.add(sum);

				}

			} // End for loop

			delta.add(tmp);

		} // End for loop

		// Update the weights

		for(int l = 0; l < layers.size(); l++) { // For each layer

			tmp = delta.pop();

			for(int n = 0; n < layers.get(l).getNeuronCount()-1; n++) { // For each neuron

				z = layers.get(l).getNeurons().get(n);

				for(int w = 0; w < z.getInput().size(); w++) { // For each weight in that neuron

					if(l == 0) {

						weight = eta * tmp.get(n) * input.get(w); // -eta * delta_l * x_i-1

					} else {

						weight = eta * tmp.get(n) * layers.get(l-1).getNeurons().get(w).getOutput().get(0);

					}

					z.getInput().set(w, z.getInput().get(w) + weight);

				}

				b = bias.get(l).getNeurons().get(0); // Update bias
				b.getInput().set(n,b.getInput().get(n) + (eta*tmp.get(n)));

			}

		} // End for loop

	}

	/** Print all of the weights in the neural network. */

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

	/** Print the output of a given layer. */

	public void printOutput(int layer) {

		for(Neuron n : layers.get(layer).getNeurons()) {

			System.out.printf("%4.4f\n",n.getOutput().get(0));

		}

	}

}
