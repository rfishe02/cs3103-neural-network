
import java.util.ArrayList;
import java.util.Random;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;

public class Test {

  public static void main(String[] args) {

    ArrayList<Double> x = getInput("x1.txt",56);
    ArrayList<Double> y = getInput("y1.txt",56);
    ArrayList<Double> z = getInput("z1.txt",56);

    int input = 56;
    int output =3;
    int hidden =30;
    int width = 3;

    NeuralNetwork n = new NeuralNetwork();
    n.buildNetwork(input,hidden,width,output);

    ArrayList<Double> xTarget = new ArrayList<>(3);
    xTarget.add(1.0);
    xTarget.add(0.0);
    xTarget.add(0.0);

    ArrayList<Double> yTarget = new ArrayList<>(3);
    yTarget.add(0.0);
    yTarget.add(1.0);
    yTarget.add(0.0);

    ArrayList<Double> zTarget = new ArrayList<>(3);
    zTarget.add(0.0);
    zTarget.add(0.0);
    zTarget.add(1.0);

    Neuron a;
    double correct = 0.0;
    int epochs = 1000;
    int i = 0;

    while(correct < 0.70 && i < epochs) {

      correct = 0;

      n.forwardPass(x);
      n.backpropagate(x,xTarget);

      n.forwardPass(y);
      n.backpropagate(y,yTarget);

      n.forwardPass(z);
      n.backpropagate(z,zTarget);

      n.forwardPass(x);
      if(n.getLayers().get(n.getLayers().size()-1).getNeurons().get(0).getA() > .70) {
        correct += 1.0;
      }
      n.printLastOutput();

      n.forwardPass(y);
      if(n.getLayers().get(n.getLayers().size()-1).getNeurons().get(1).getA() > .70) {
        correct += 1.0;
      }
      n.printLastOutput();

      n.forwardPass(z);
      if(n.getLayers().get(n.getLayers().size()-1).getNeurons().get(2).getA() > .70) {
        correct += 1.0;
      }
      n.printLastOutput();

      correct = correct / 3.0;
      System.out.println(correct);

      i++;

    }

    System.out.println(i);

    //train(n,x,100);
    //train(n,y,100);

    //n.printWeights();


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
    t.printOutput();
    */

  }

  public static void train(NeuralNetwork n, ArrayList<Double> x, int epochs) {

    for(int i = 0; i < epochs; i++) {
      n.forwardPass(x);
      n.backpropagate(x,x);
    }

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
