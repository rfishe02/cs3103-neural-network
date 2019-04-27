
import java.util.ArrayList;
import java.util.Random;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;

public class Test {

  public static void main(String[] args) {

    ArrayList<Double> x = getInput("x1.txt",56);

    int input = x.size()-1;
    int output = 3;
    int hidden = 8;
    int width = 4;

    NeuralNetwork n = new NeuralNetwork();
    n.buildNetwork(input,hidden,width,output);
    n.printWeights();

    n.forwardPass(x);
    n.backpropagate(1); // 1 for x, 2 for y, 3 for z

    System.out.println("\nAFTER\n");
    n.printWeights();

  }

  /** Read an 8 x 7 text file, with a letter written with the character X. */

  public static ArrayList<Double> getInput(String filename, int size) {

    ArrayList<Double> output = new ArrayList<>(size+1);

    try {
      BufferedReader br = new BufferedReader(new FileReader(filename));
      String read;

      while((read = br.readLine())!=null) {

        for(int i = 0; i < read.length(); i++) {

          if(Character.compare(read.charAt(i),'.') == 0) {
            output.add(0.0);
          } else {
            output.add(1.0);
          }

        }

      }

    } catch(IOException ex) {
      ex.printStackTrace();
    }

    output.add(1.0);

    return output;

  }

  /** Generate an input of a given size, with the added increment */

  public static ArrayList<Double> generateInput(int input, int inc) {

    ArrayList<Double> data = new ArrayList<>(input+1);

    for(int j = 0; j < input+1; j++) {
      data.add(new Random().nextDouble() + inc);
    }

    return data;

  }

}
