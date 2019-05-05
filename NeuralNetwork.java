

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Stack;
import java.util.LinkedList;
import java.util.Random;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Serializable;

public class NeuralNetwork implements Serializable {

	private static final long serialVersionUID = 1L;
	private ArrayList<Layer> bias;
	private ArrayList<Layer> layers;
	private int inputLayerCount;
	private int hiddenLayerCount;
	private int ouputLayerCount;
	double eta = -0.10;

	public ArrayList<Layer> getLayers() {
		return layers;
	}

	public double activationFunction(double value) {
		return 1 / (1 + Math.pow(Math.E, value * -1));
	}

	public double activationFunctionDerivative(double a) {
		return a * (1 - a);
	}

	public double error(double target, double output) {
		return output-target;
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

		for (int l = 0; l < width; l++) {

			h = new Layer();

			b = new Layer();
			b.setNeuronCount(1);

			if (l == 0) {
				h.setNeuronCount(hidden);
				h.makeLayers(input);
				b.makeLayers(hidden);

			} else if (l == width-1) {
				h.setNeuronCount(output);
				h.makeLayers(hidden);
				b.makeLayers(output);

			}
			else {
				h.setNeuronCount(hidden);
				h.makeLayers(hidden);
				b.makeLayers(hidden);

			}

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

			for(int j = 0; j < layers.get(l).getNeurons().size(); j++) { // For each neuron in that layer

				sum = 0;

				for(int i = 0; i < layers.get(l).getNeurons().get(j).getW().size(); i++) { // For each weight in that neuron

					if(l == 0) { // It's in the first layer

						sum += x.get(i) * layers.get(l).getNeurons().get(j).getW().get(i);

					} else { // It's in the other layers

						z = layers.get(l-1).getNeurons().get(i);
						sum += z.getA() * layers.get(l).getNeurons().get(j).getW().get(i);

					}
				}
				sum += bias.get(l).getNeurons().get(0).getW().get(j); // Add the bias to each node ouput

				layers.get(l).getNeurons().get(j).setIn(sum);
				layers.get(l).getNeurons().get(j).setA(activationFunction(sum));

			}
		}
	}

	/** Conduct the backpropagation phase, and update weights. */

	public void backpropagate(ArrayList<Double> input, ArrayList<Double> target) {
		Stack<ArrayList<Double>> delta = new Stack<ArrayList<Double>>();
		ArrayList<Double> tmp;
		Neuron z;
		double d;
		double output;
		double sum;

		for (int l = layers.size() - 1; l >= 0; l--) { // For each layer, from the last to first

			tmp = new ArrayList<>(layers.get(l).getNeurons().size());

			for(int j = 0; j < layers.get(l).getNeuronCount(); j++) { // For each neuron in that layer

				z = layers.get(l).getNeurons().get(j);
				d = activationFunctionDerivative(z.getA());

				if(l == layers.size() - 1) { // If it's the last layer (output)

					tmp.add(d * error(target.get(j),z.getA())); // Calculate delta_k = O_k (1 - O_k) (O_k - t_k) for output

				} else { // It's in the other layers.

					sum = 0;

					for(int k = 0; k < delta.peek().size(); k++) { // For each k
						sum += delta.peek().get(k) * layers.get(l+1).getNeurons().get(k).getW().get(j);
					}

					tmp.add(d * sum); // Calculate O_j (1- O_j) Summation delta_k * W_j+1 k

				}
			} // End for loop

			delta.add(tmp);

		} // End for loop

		updateWeights(input,delta); // Update the weights

	}

	public void updateWeights(ArrayList<Double> input,Stack<ArrayList<Double>> delta) {
		ArrayList<Double> tmp;
		Neuron z;
		Neuron b;
		double change;

		for(int l = 0; l < layers.size(); l++) { // For each layer

			tmp = delta.pop();

			for(int j = 0; j < layers.get(l).getNeurons().size(); j++) { // For each neuron

				z = layers.get(l).getNeurons().get(j);

				for(int i = 0; i < z.getW().size(); i++) { // For each weight in that neuron

					if(l == 0) {
						change = tmp.get(j) * input.get(i);
					} else {
						change = tmp.get(j) * layers.get(l-1).getNeurons().get(i).getA();
					}

					z.getW().set(i, z.getW().get(i) + (eta * change));

				}

				b = bias.get(l).getNeurons().get(0); // Update bias
				b.getW().set(j, b.getW().get(j) + ( eta * tmp.get(j) ));

			}

		} // End for loop

	}

	/** Print all of the weights in the neural network to the console. */

	public void printWeights() {
		Neuron z;

		for(int l = 0; l < layers.size(); l++) {

			for(int n = 0; n < layers.get(l).getNeurons().size(); n++) {

				z = layers.get(l).getNeurons().get(n);

				for(Double w : z.getW()) {

					System.out.printf("%4.3f ",w);

				}
				System.out.printf("%4.3f \n",bias.get(l).getNeurons().get(0).getW().get(n));

			}
			System.out.println();

		}
	}

	/** Prints the weights to a file. Do not use this while training, unless you want a 2GB size file. */

	public void printWeights(BufferedWriter bw, int[] tGroup, int round, int epoch) throws IOException {

		for(int l = 0; l < layers.size(); l++) {

			for(int n = 0; n < layers.get(l).getNeuronCount(); n++) {

				for(int w = 0; w < layers.get(l).getNeurons().get(n).getW().size(); w++) {
					bw.write(tGroup[0]+"-"+tGroup[1]+","+round+","+epoch+","+l+","+n+",N,"+w+","+layers.get(l).getNeurons().get(n).getW().get(w)+"\n"); //"tGroup,round,epoch,layer,neuron,type,weight,value\n"
				}
			}

			for(int w = 0; w < bias.get(l).getNeurons().get(0).getW().size(); w++) {
				bw.write(tGroup[0]+"-"+tGroup[1]+","+round+","+epoch+","+l+","+0+",B,"+w+","+bias.get(l).getNeurons().get(0).getW().get(w)+"\n");
			}
		}
	}

	/** Print the output all layers. */

	public void printOutput() {
		Neuron z;

		for(Layer l : layers) {

			for(int n = 0; n < l.getNeurons().size(); n++) {
				z = l.getNeurons().get(n);

				System.out.printf("%4.3f %4.3f\n",z.getIn(),z.getA());
			}
			System.out.println();
		}
	}

	/** Print the output of just the last layer. */

	public void printLastOutput() {
		Neuron z;

		for(int n = 0; n < layers.get(layers.size()-1).getNeurons().size(); n++) {
			z = layers.get(layers.size()-1).getNeurons().get(n);

			if(n == 0) {
				System.out.printf("X: %2.2f \n",z.getA());
			} else if(n == 1) {
				System.out.printf("Y: %2.2f \n",z.getA());
			} else {
				System.out.printf("Z: %2.2f \n",z.getA());
			}

			/*
			if((n+1) % 8 == 0) {
				System.out.println();
			}
			*/
		}
		System.out.println();
	}

	public String getLastOutput() {
		String s = "";
		Neuron z;

		for(int n = 0; n < layers.get(layers.size()-1).getNeurons().size(); n++) {
			z = layers.get(layers.size()-1).getNeurons().get(n);

			s += String.format("%2.2f",z.getA());

			if(n < layers.get(layers.size()-1).getNeurons().size()-1) {
				s += ",";
			}
		}
		return s;
	}

}
