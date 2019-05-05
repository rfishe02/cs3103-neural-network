
import java.util.ArrayList;
import java.util.Random;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

public class Test {

  public static void main(String[] args) {

    NeuralNetwork n;

    // Train the neural network an save it.

    //n = new NeuralNetwork();
    //n.buildNetwork(56,30,3,3); //input,hidden,width,output
    //trainNetwork(n,0);

    //saveNeuralNetwork(n,"trained-nn");

    // Load a trained neural network and test it with a single file.

    n = loadNeuralNetwork("trained-nn");

    ArrayList<Double> test = getLoneInput("z1.txt");

    n.forwardPass(test);
    n.printLastOutput();

  }

  /** Used to build a single arraylist from a file for testing. */

  public static ArrayList<Double> getLoneInput(String filename) {

    ArrayList<Double> output = new ArrayList<>(56);

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

    return output;

  }

  /** Save a trained neural network object to disk. */

  public static void saveNeuralNetwork(NeuralNetwork nn, String filename) {

    try {

      FileOutputStream fileOut = new FileOutputStream(filename);
      ObjectOutputStream objectOut = new ObjectOutputStream(fileOut);
      objectOut.writeObject(nn);
      objectOut.close();

    } catch (IOException ex) {
      ex.printStackTrace();
    }

  }

  /** Load a trained neural network object from disk. */

  public static NeuralNetwork loadNeuralNetwork(String filename) {

    NeuralNetwork nn = null;

    try {

      FileInputStream fileIn = new FileInputStream(filename);
      ObjectInputStream objectIn = new ObjectInputStream(fileIn);

      nn = (NeuralNetwork)objectIn.readObject();

      objectIn.close();

    } catch (IOException ex) {
      ex.printStackTrace();
    } catch (ClassNotFoundException ex) {
      ex.printStackTrace();
    }

    return nn;

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
            n = new Node(tmp,target,0);
          } else if(read.equals("Y")) {
            target = new ArrayList<>(Arrays.asList(0.0,1.0,0.0));
            n = new Node(tmp,target,1);
          } else {
            target = new ArrayList<>(Arrays.asList(0.0,0.0,1.0));
            n = new Node(tmp,target,2);
          }

          output.get(rand.nextInt(3)).add(n); // Add at random to one of three ArrayLists.
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

    } catch(IOException ex) {
      ex.printStackTrace();
    }

    return output;

  }

  public static void trainNetwork(NeuralNetwork n, int a) {
    int epochs = 700;

    ArrayList<ArrayList<Node>> tData = getArray("test.txt"); // This is the training & test data.

    ArrayList<int[]> tGroup = new ArrayList<>();
    tGroup.add(new int[]{1,2});
    tGroup.add(new int[]{0,2});
    tGroup.add(new int[]{1,0});

    int[] three = {
      0,1,2
    };

    train(tData,n,tGroup.get(a),epochs);
    test(tData,n,three[a]);

  }

  public static void train(ArrayList<ArrayList<Node>> a, NeuralNetwork n, int[] tGroup, int epochs) {

    Random rand = new Random();
    Node tNode;
    double[] outcome;
    int i = 0;
    int x;

    while(i < epochs) {

      // Train with the two selected groups. Select a random record to train with.
      x = rand.nextInt(tGroup.length);

      tNode = a.get(tGroup[x]).get(rand.nextInt(a.get(tGroup[x]).size()));
      n.forwardPass(tNode.getLetter());
      n.backpropagate(tNode.getLetter(),tNode.getTarget());

      outcome = getOutcome(n);

      i++;

    }

  }

  public static void test(ArrayList<ArrayList<Node>> a, NeuralNetwork n, int three) {

    double[] outcome;
    double correct = 0.0;

    for(Node test : a.get(three)) {

      n.forwardPass(test.getLetter()); // Pass the data through the network.

      outcome = getOutcome(n);

      if((int)outcome[0] == test.getN() && outcome[1] > .60) {
        correct += 1.0;
      } // The outcome needs to be at a sufficient level.

    }

    correct = correct / a.get(three).size();
    System.out.println(correct);

  }

  /** This methods finds the most likely outcome of the three output nodes. */

  public static double[] getOutcome(NeuralNetwork n) {

    double[] outcome = new double[2];
    double max = 0.0;
    int maxOutcome = 0;

    for(int c = 0; c < n.getLayers().get(n.getLayers().size()-1).getNeuronCount(); c++) {

      if(n.getLayers().get(n.getLayers().size()-1).getNeurons().get(c).getA() > max) {
        max = n.getLayers().get(n.getLayers().size()-1).getNeurons().get(c).getA();
        maxOutcome = c;
      }

    } // Find the neuron with the highest result. This should match the target of the case.

    outcome[0] = (double)maxOutcome;
    outcome[1] = max;

    return outcome;

  }

  //============================================================================
  // The following methods were used to test the neural
  // network and gather the results.
  //============================================================================

  /** Used to gather data in multiple trials, with different test sets. */

  public static void trainNetworkAndCollect() {
    int epochs = 700;

    ArrayList<ArrayList<Node>> tData = getArray("test.txt"); // This is the training & test data.

    ArrayList<int[]> tGroup = new ArrayList<>();
    tGroup.add(new int[]{1,2});
    tGroup.add(new int[]{0,2});
    tGroup.add(new int[]{1,0});

    int[] three = {
      0,1,2
    };

    try {

      BufferedWriter bTest = new BufferedWriter(new FileWriter("test-output-"+epochs+".csv"));
      BufferedWriter bTrain = new BufferedWriter(new FileWriter("training-output-"+epochs+".csv"));
      BufferedWriter bWeight = new BufferedWriter(new FileWriter("weights-"+epochs+".csv"));

      bTrain.write("tGroup,round,epoch,target,outcome,probX,probY,probZ\n");
      bTest.write("tGroup,round,target,outcome,probX,probY,probZ\n");
      bWeight.write("tGroup,round,epoch,layer,neuron,type,weight,value\n");

      for(int a = 0; a < 3; a++) {

        for(int i = 0; i < 10; i++) {
          NeuralNetwork n = new NeuralNetwork();
          n.buildNetwork(56,30,3,3);

          train(bTrain,tData,n,tGroup.get(a),epochs,i);
          test(bTest,tData,n,three[a],epochs,i);

          n.printWeights(bWeight,tGroup.get(a),i,epochs);
        }

      }

      bTest.close();
      bTrain.close();
      bWeight.close();

    } catch(IOException ex) {
      ex.printStackTrace();
    }

  }

  /** This method is used with the collectData method to train the network. */

  public static void train(BufferedWriter bw, ArrayList<ArrayList<Node>> a, NeuralNetwork n, int[] tGroup, int epochs, int round) throws IOException {

    Random rand = new Random();
    Node tNode;
    double[] outcome;
    int i = 0;
    int x;

    while(i < epochs) {

      // Train with the two selected groups. Select a random record to train with.
      x = rand.nextInt(tGroup.length);

      tNode = a.get(tGroup[x]).get(rand.nextInt(a.get(tGroup[x]).size()));
      n.forwardPass(tNode.getLetter());
      n.backpropagate(tNode.getLetter(),tNode.getTarget());

      outcome = getOutcome(n);
      bw.write(tGroup[0]+"-"+tGroup[1]+","+round+","+i+","+tNode.getN()+","+(int)outcome[0]+","+n.getLastOutput()+"\n"); //"tGroup,round,epoch,target,outcome,probX,probY,probZ\n"

      i++;

    }

  }

  /** This method is used with the collectData method to test the network. */

  public static void test(BufferedWriter bw, ArrayList<ArrayList<Node>> a, NeuralNetwork n, int three, int epochs, int round) throws IOException {

    double[] outcome;
    double correct = 0.0;

    for(Node test : a.get(three)) {

      n.forwardPass(test.getLetter()); // Pass the data through the network.

      outcome = getOutcome(n);

      if((int)outcome[0] == test.getN() && outcome[1] > .60) {
        correct += 1.0;
      } // The outcome needs to be at a sufficient level.

      bw.write(three+","+round+","+test.getN()+","+(int)outcome[0]+","+n.getLastOutput()+"\n"); //"tGroup,round,target,outcome,probX,probY,probZ\n"

    }

    correct = correct / a.get(three).size();
    System.out.println(correct);

  }

}
