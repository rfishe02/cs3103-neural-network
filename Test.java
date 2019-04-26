
import java.util.ArrayList;
import java.util.Random;

public class Test {

  public static void main(String[] args) {

    int input = 64;
    int output = 3;
    int hidden = 8;
    int width = 4;

    NeuralNetwork n = new NeuralNetwork();
    n.buildNetwork(input,hidden,width,output);
    n.printWeights();

    for(int i = 0; i < 50; i++) {

      ArrayList<Double> data = new ArrayList<>(input);

      for(int j = 0; j < input; j++) {
        data.add(new Random().nextDouble() + 1);
      }
      n.forwardPass(data);

      for(Double d : data) {

        if(d % 2 == 0) {
          n.backpropagate(new Random().nextDouble() + 4);
        } else {
          n.backpropagate(new Random().nextDouble() + 2);
        }

      }

    }

    System.out.println("\nAFTER\n");
    n.printWeights();

  }

}
