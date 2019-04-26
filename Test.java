
import java.util.ArrayList;
import java.util.Random;

public class Test {

  public static void main(String[] args) {

    NeuralNetwork n = new NeuralNetwork();
    n.buildNetwork(8,4,4,8);

    //n.printWeights();

    for(int i = 0; i < 50; i++) {

      ArrayList<Double> data = new ArrayList<>(8);

      for(int j = 0; j < 8; j++) {
        data.add(new Random().nextDouble() + 1);
      }
      n.forwardPass(data);

      for(Double d : data) {

        if(d % 2 == 0) {
          n.backpropagate(new Random().nextDouble() + 1);
        } else {
          n.backpropagate(new Random().nextDouble() + 0);
        }

      }

      System.out.println();
      n.printWeights();

    }

    //System.out.println("\nAFTER\n");
    //n.printWeights();

  }

}
