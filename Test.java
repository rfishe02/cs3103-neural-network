
import java.util.ArrayList;
import java.util.Random;

public class Test {

  public static void main(String[] args) {

    NeuralNetwork n = new NeuralNetwork();
    n.buildNetwork(8,4,5,1);

    n.printWeights();

    for(int i = 0; i < 10; i++) {

      ArrayList<Double> data = new ArrayList<>(8);

      for(int j = 0; j < 8; j++) {
        data.add(new Random().nextDouble() + 1);
      }
      n.forwardPass(data);
      n.backpropagate(7);
    }

    System.out.println("\nAFTER\n");
    n.printWeights();

  }

}
