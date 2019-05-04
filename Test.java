
import java.util.ArrayList;
import java.util.Random;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;

public class Test {

  public static void main(String[] args) {

    int input = 56;
    int output =3;
    int hidden =30;
    int width = 3;

    getArray("test.txt");

    /*
    NeuralNetwork n = new NeuralNetwork();
    n.buildNetwork(input,hidden,width,output);

    ArrayList<Double> xTarget = new ArrayList<>(Arrays.asList(1.0,0.0,0.0));

    ArrayList<Double> yTarget = new ArrayList<>(Arrays.asList(0.0,1.0,0.0));

    ArrayList<Double> zTarget = new ArrayList<>(Arrays.asList(0.0,0.0,1.0));
    */

    Neuron a;
    double correct = 0.0;
    int epochs = 1000;
    int i = 0;

    /*
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

    }*/

    //System.out.println(i);

  }

  public static void train(NeuralNetwork n, ArrayList<Double> x, ArrayList<Double> y, int epochs) {

    for(int i = 0; i < epochs; i++) {
      n.forwardPass(x);
      n.backpropagate(x,y);
    }

  }

  /** Build an ArrayList for 3-fold cross validation. It accepts a file with a series of 8x7 characters separated by spaces.
      The target would be one of three outcomes, since this network is classifying one of three letters. The output layer should
      have three neurons.
  */

  public static ArrayList<ArrayList<Node>> getArray(String filename) {

    ArrayList<ArrayList<Node>> output = new ArrayList<>(3);
    output.add(new ArrayList<>());
    output.add(new ArrayList<>());
    output.add(new ArrayList<>());

    Node n;
    ArrayList<Double> tmp = new ArrayList<>();
    ArrayList<Double> target;
    Random rand = new Random();

    try {
      BufferedReader br = new BufferedReader(new FileReader(filename));
      String read;

      while((read = br.readLine())!=null) {

        if(read.length() < 2) {

          if(read.equals("X")) {
            target = new ArrayList<>(Arrays.asList(1.0,0.0,0.0));
          } else if(read.equals("Y")) {
            target = new ArrayList<>(Arrays.asList(0.0,1.0,0.0));
          } else {
            target = new ArrayList<>(Arrays.asList(0.0,0.0,1.0));
          }

          output.get(rand.nextInt(3)).add(new Node(tmp,target)); // Add at random to one of three ArrayLists.
          tmp = new ArrayList<>();

        } else {

          for(int i = 0; i < read.length(); i++) {

            if(Character.compare(read.charAt(i),'.') == 0) {
              tmp.add(0.0);
            } else {
              tmp.add(1.0);
            }

          }

        }

      }

      //System.out.println(output.get(0).size());
      //System.out.println(output.get(1).size());
      //System.out.println(output.get(2).size());

    } catch(IOException ex) {
      ex.printStackTrace();
    }

    return output;

  }

  public void printArrayList(ArrayList<ArrayList<Double>> output) {

    for(ArrayList<Double> a : output) {

      for(int i = 0; i < a.size(); i++) {

        System.out.print(Math.round(a.get(i))+" ");

        if((i+1)%8 == 0) {
          System.out.println();
        }

      }
      System.out.println();

    }

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
