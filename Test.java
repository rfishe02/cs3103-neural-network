
import java.util.ArrayList;
import java.util.Random;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;

public class Test {

  public static void main(String[] args) {


    ArrayList<Double> x = getInput("x1.txt",56);
    ArrayList<Double> x2 = getInput("y1.txt",56);
    ArrayList<Double> x3 = getInput("z1.txt",56);

    int input = 56;
    int output =3;
    int hidden =14;
    int width = 7;

    NeuralNetwork n = new NeuralNetwork();
    n.buildNetwork(input,hidden,width,output);

    ArrayList<Double> xTarget = new ArrayList<>(3);
    xTarget.add(1.0);
    xTarget.add(0.0);
    xTarget.add(0.0);

    for(int i = 0; i < 10; i++) {
      n.forwardPass(x);
      n.backpropagate(x,xTarget); // 1 for x, 2 for y, 3 for z
    }

    ArrayList<Double> yTarget = new ArrayList<>(3);
    yTarget.add(0.0);
    yTarget.add(1.0);
    yTarget.add(0.0);

    for(int i = 0; i < 10; i++) {
      n.forwardPass(x2);
      n.backpropagate(x2,yTarget); // 1 for x, 2 for y, 3 for z
    }

    ArrayList<Double> zTarget = new ArrayList<>(3);
    zTarget.add(0.0);
    zTarget.add(0.0);
    zTarget.add(1.0);

    for(int i = 0; i < 10; i++) {
      n.forwardPass(x3);
      n.backpropagate(x3,zTarget); // 1 for x, 2 for y, 3 for z
    }

    n.forwardPass(x);
    n.printOutput(n.getLayers().size()-1);
    System.out.println();

    n.forwardPass(x2);
    n.printOutput(n.getLayers().size()-1);


    /////////////////////

    /*
    ArrayList<Double> tData = new ArrayList<>();
    tData.add(1.0);
    tData.add(2.0);
    tData.add(3.0);

    NeuralNetwork t = new NeuralNetwork();
    t.buildNetwork(3,2,3,2);

    System.out.println();
    t.printWeights();
    System.out.println();

    t.forwardPass(tData);
    t.printOutput(0);
    System.out.println();
    t.printOutput(1);
    */

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
