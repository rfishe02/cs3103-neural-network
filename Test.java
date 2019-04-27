
import java.util.ArrayList;
import java.util.Random;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;

public class Test {

  public static void main(String[] args) {

    ArrayList<Double> x = getInput("x1.txt",56);
    ArrayList<Double> x2 = getInput("y1.txt",56);

    int input = 56;
    int output =3;
    int hidden =28;
    int width = 14;

    NeuralNetwork n = new NeuralNetwork();
    n.buildNetwork(input,hidden,width,output);

    /*
    x = genInput(input,1);

    for(int i = 0; i < 25; i++) {
      n.forwardPass(x);
      n.backpropagate(0.2);
      n.printOutput();
      System.out.println();
    }
    */

    ArrayList<Double> xTarget = new ArrayList<>(3);
    xTarget.add(-1.0);
    xTarget.add(1.0);
    xTarget.add(1.0);

    for(int i = 0; i < 25; i++) {
      n.forwardPass(x);
      n.backpropagate(xTarget); // 1 for x, 2 for y, 3 for z
    }

    ArrayList<Double> yTarget = new ArrayList<>(3);
    yTarget.add(1.0);
    yTarget.add(-1.0);
    yTarget.add(1.0);

    for(int i = 0; i < 10; i++) {
      n.forwardPass(x2);
      n.backpropagate(yTarget); // 1 for x, 2 for y, 3 for z
    }

    n.forwardPass(x);
    n.printOutput();
    System.out.println();

    n.forwardPass(x2);
    n.printOutput();

  }

  /** Read an 8 x 7 text file, with a letter written with the character X. */

  public static ArrayList<Double> getInput(String filename, int size) {

    ArrayList<Double> output = new ArrayList<>(size);

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

    //output.add(1.0);

    return output;

  }

  /** Generate an input of a given size, with the added increment */

  public static ArrayList<Double> genInput(int input, int inc) {

    ArrayList<Double> data = new ArrayList<>(input);

    for(int j = 0; j < input; j++) {
      data.add(new Random().nextDouble() + inc);
    }

    return data;

  }

}
